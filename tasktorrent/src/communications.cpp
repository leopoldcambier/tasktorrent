#ifndef TTOR_SHARED

#include <list>
#include <vector>
#include <mutex>
#include <tuple>
#include <cassert>

#include "communications.hpp"
#include "mpi_utils.hpp"
#include "serialization.hpp"
#include "activemessage.hpp"

namespace ttor
{

int comm_rank()
{
    int world_rank;
    TASKTORRENT_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    return world_rank;
}

int comm_size()
{
    int world_size;
    TASKTORRENT_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    return world_size;
}

std::string processor_name()
{
    char name[MPI_MAX_PROCESSOR_NAME];
    int size;
    TASKTORRENT_MPI_CHECK(MPI_Get_processor_name(name, &size));
    return std::string(name);
}

/**
 * Communicator
 */

Communicator::Communicator(int verb_) : 
    verb(verb_), 
    logger(nullptr), 
    log(false), 
    messages_queued(0), 
    messages_processed(0) {
        TASKTORRENT_MPI_CHECK(MPI_Type_contiguous(static_cast<int>(mega), MPI_BYTE, &MPI_MEGABYTE));
        TASKTORRENT_MPI_CHECK(MPI_Type_commit(&MPI_MEGABYTE));
    }
    
std::unique_ptr<message> Communicator::make_active_message(int dest, size_t size)
{
    auto m = std::make_unique<message>(dest);
    size_t buffer_size = 0;
    int tag = 0;
    if(size > max_int_size) {
        buffer_size = mega * ((size + mega - 1) / mega); // pad so that the total size if a multiple of mega (2^20)
        tag = 1;
    } else {
        buffer_size = size;
        tag = 0;
    }
    m->buffer.resize(buffer_size);
    m->tag = tag;
    return m;
}

void Communicator::Isend_message(const std::unique_ptr<message> &m)
{
    if (verb > 1)
        printf("[%2d] -> %d: sending msg [tag %d], %zd B, rqst %p\n", comm_rank(), m->other, m->tag, m->buffer.size(), (void*)&(m->request));

    if(m->tag == 0) {
        size_t size = m->buffer.size();
        assert(size <= max_int_size);
        TASKTORRENT_MPI_CHECK(MPI_Isend(m->buffer.data(), static_cast<int>(size), MPI_BYTE, m->other, m->tag, MPI_COMM_WORLD, &(m->request)));
    } else if(m->tag == 1) {
        assert(m->buffer.size() > max_int_size);
        assert(m->buffer.size() % mega == 0);
        size_t size = m->buffer.size() / mega;
        if(size > max_int_size) {
            printf("Error in Communicator::Isend_message: requested message size of %zd larger than maximum of 2^31 MB\n", m->buffer.size());
            MPI_Finalize();
            exit(1);
        }
        TASKTORRENT_MPI_CHECK(MPI_Isend(m->buffer.data(), static_cast<int>(size), MPI_MEGABYTE, m->other, m->tag, MPI_COMM_WORLD, &(m->request)));
    } else {
        assert(false);
    }

    if (verb > 4)
        print_bytes(m->buffer);
}

void Communicator::Isend_queued_messages()
{
    std::list<std::unique_ptr<message>> to_Isend;
    {
        std::lock_guard<std::mutex> lock(messages_rdy_mtx);
        to_Isend.swap(messages_rdy);
        assert(messages_rdy.size() == 0);
    }
    const int self = comm_rank();
    for (auto &m : to_Isend)
    {
        if (m->other != self)
        {
            /* real MPI message to remote rank */
            Isend_message(m);
            messages_Isent.push_back(move(m));
        }
        else
        {
            /* This is a message to self.
             * Push directly to the list of received messages.
             */
            messages_Ircvd.push_back(move(m));
        }
    }
}

void Communicator::test_Isent_messages()
{
    std::list<std::unique_ptr<message>> messages_Isent_new;
    for (auto &m : messages_Isent)
    {
        int flag = 0;
        TASKTORRENT_MPI_CHECK(MPI_Test(&m->request, &flag, MPI_STATUS_IGNORE));
        if (flag) // operation completed
        {
            if (verb > 1)
                printf("[%2d] -> %d: msg [tag %d] sent, rqst %p completed\n", comm_rank(), m->other, m->tag, (void*)&m->request);
        }
        else
            messages_Isent_new.push_back(move(m));
    }
    messages_Isent.swap(messages_Isent_new);
}

// Return true if there is a message and we started an Irecv; false otherwise
bool Communicator::probe_Irecv_message(std::unique_ptr<message> &m)
{
    if (verb > 3)
        printf("[%2d] MPI probe\n", comm_rank());

    MPI_Status mpi_status;
    int mpi_size, mpi_flag; // MPI uses int for message count (mpi_size)
    TASKTORRENT_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_flag, &mpi_status));
    if (!mpi_flag)
        return false;

    int mpi_tag = mpi_status.MPI_TAG;
    int source = mpi_status.MPI_SOURCE;
    size_t buffer_size = 0;
    m = std::make_unique<message>(source);
    if(mpi_tag == 0) { // We are receiving MPI_BYTE
        TASKTORRENT_MPI_CHECK(MPI_Get_count(&mpi_status, MPI_BYTE, &mpi_size));
        buffer_size = static_cast<size_t>(mpi_size);
    } else if(mpi_tag == 1) { // We are receiving MPI_MEGABYTE
        TASKTORRENT_MPI_CHECK(MPI_Get_count(&mpi_status, MPI_MEGABYTE, &mpi_size));
        buffer_size = static_cast<size_t>(mpi_size) * mega;
    } else {
        assert(false);
    }
    m->buffer.resize(buffer_size);
    m->tag = mpi_tag;
    if (verb > 1)
        printf("[%2d] <- %d: receiving msg [tag %d], %zd B, rqst %p\n", comm_rank(), source, mpi_tag, buffer_size, (void*)&m->request);

    if(mpi_tag == 0) { // We are receiving MPI_BYTE
        TASKTORRENT_MPI_CHECK(MPI_Irecv(m->buffer.data(), mpi_size, MPI_BYTE, source, mpi_tag, MPI_COMM_WORLD, &m->request));
    } else if(mpi_tag == 1) {
        TASKTORRENT_MPI_CHECK(MPI_Irecv(m->buffer.data(), mpi_size, MPI_MEGABYTE, source, mpi_tag, MPI_COMM_WORLD, &m->request));
    } else {
        assert(false);
    }

    return true;
}

void Communicator::process_Ircvd_messages()
{
    std::list<std::unique_ptr<message>> messages_Ircvd_new;
    const int self = comm_rank();
    for (auto &m : messages_Ircvd)
    {
        int flag = 0;
        if (m->other != self)
        {
            /* Real remote message */
            TASKTORRENT_MPI_CHECK(MPI_Test(&m->request, &flag, MPI_STATUS_IGNORE));
        }
        else
        {
            // Message is local; it contains already all the data we need
            flag = 1;
        }
        if (flag)
        { // Message has completed
            if (verb > 1)
                printf("[%2d] -> %d: msg [tag %d] received, rqst %p complete\n", comm_rank(), m->other, m->tag, (void*)&m->request);

            std::unique_ptr<Event> e;
            if (log)
                e = std::make_unique<Event>("rank_" + std::to_string(comm_rank()) + ">lpc>" + "rank_" + std::to_string(m->other) + ">" + std::to_string(m->tag));

            // Process the message
            process_message(m);

            if (log)
                logger->record(std::move(e));
        }
        else
            messages_Ircvd_new.push_back(move(m));
    }
    messages_Ircvd.swap(messages_Ircvd_new);
}

void Communicator::process_message(const std::unique_ptr<message> &m)
{
    Serializer<int> s;
    std::tuple<int> tup = s.read_buffer(m->buffer.data(), m->buffer.size());
    int am_id = std::get<0>(tup);
    assert(am_id >= 0 && am_id < static_cast<int>(active_messages.size()));
    if (verb > 4)
    {
        printf("[%2d] <- %2d: lpc() ID %d, data received: ", comm_rank(), m->other, am_id);
        print_bytes(m->buffer);
    }
    else if (verb > 1)
    {
        printf("[%2d] <- %2d: running lpc() ID %d\n", comm_rank(), m->other, am_id);
    }

    {
        // Run the callback function in the message
        active_messages.at(am_id)->run(m->buffer.data(), m->buffer.size());

        // This must be done strictly after running the callback function
        // This ensures that all potential new tasks have been created before we increment messages_rcvd
        messages_processed++;
    }

    if (verb > 2)
        printf("[%2d] <- %d: msg ID %d, lpc() completed, %lu B\n", comm_rank(), m->other, am_id, m->buffer.size());
}

void Communicator::set_logger(Logger *logger_)
{
    log = true;
    logger = logger_;
}

void Communicator::recv_process()
{
    // (1) Try Irecv
    while (true)
    {
        std::unique_ptr<message> m;
        bool success = probe_Irecv_message(m);
        // Note: if probe_Irecv_message keep returning false we never exit
        if (success)
        {
            // (2) Wait and then process message
            TASKTORRENT_MPI_CHECK(MPI_Wait(&m->request, MPI_STATUS_IGNORE));
            process_message(m);
            break;
        }
    }
}

void Communicator::progress()
{
    // Keep checking for Irecv messages and insert into queue
    while (true)
    {
        std::unique_ptr<message> m;
        bool success = probe_Irecv_message(m);
        if (success)
            messages_Ircvd.push_back(std::move(m));
        else
            // Iprobe says there are no messages in the pipeline
            break;
    }

    process_Ircvd_messages();
    Isend_queued_messages();
    test_Isent_messages();
}

bool Communicator::is_done()
{
    bool ret;
    {
        std::lock_guard<std::mutex> lock(messages_rdy_mtx);
        ret = messages_rdy.empty() && messages_Isent.empty() && messages_Ircvd.empty();
    }
    return ret;
}

int Communicator::get_n_msg_processed()
{
    return messages_processed.load();
}

int Communicator::get_n_msg_queued()
{
    return messages_queued.load();
}

} // namespace ttor

#endif