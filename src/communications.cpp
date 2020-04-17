#include "communications.hpp"

using namespace std;

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

string processor_name()
{
    char name[MPI_MAX_PROCESSOR_NAME];
    int size;
    TASKTORRENT_MPI_CHECK(MPI_Get_processor_name(name, &size));
    return string(name);
}

/**
 * Communicator
 */

Communicator::Communicator(int verb_, int tag_) : 
    verb(verb_), 
    logger(nullptr), 
    log(false), 
    tag(tag_), 
    messages_queued(0), 
    messages_processed(0) {}
    
unique_ptr<message> Communicator::make_active_message(int dest, size_t size)
{
    auto m = make_unique<message>(dest);
    m->buffer.resize(size);
    m->tag = tag;
    return m;
}

void Communicator::Isend_message(const unique_ptr<message> &m)
{
    if (verb > 1)
        printf("[%2d] -> %d: sending msg [tag %d], %lu B, rqst %p\n", comm_rank(), m->other, m->tag, m->buffer.size(), (void*)&(m->request));

    size_t max_size = static_cast<size_t>(std::numeric_limits<int>::max());
    if(m->buffer.size() > max_size) {
        printf("Error in Communicator::Isend_message: requested message size of %zd larger than maximum int %d\n", m->buffer.size(), std::numeric_limits<int>::max());
        MPI_Finalize();
        exit(1);
    }
    TASKTORRENT_MPI_CHECK(MPI_Isend(m->buffer.data(), static_cast<int>(m->buffer.size()), MPI_BYTE, m->other, m->tag, MPI_COMM_WORLD, &(m->request)));

    if (verb > 4)
        print_bytes(m->buffer);
}

void Communicator::Isend_queued_messages()
{
    list<unique_ptr<message>> to_Isend;
    {
        lock_guard<mutex> lock(messages_rdy_mtx);
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
    list<unique_ptr<message>> messages_Isent_new;
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
bool Communicator::probe_Irecv_message(unique_ptr<message> &m)
{
    if (verb > 3)
        printf("[%2d] MPI probe on tag %d\n", comm_rank(), tag);

    MPI_Status status;
    int size, flag; // MPI uses int for message count (size)
    TASKTORRENT_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &status));
    if (!flag)
        return false;

    TASKTORRENT_MPI_CHECK(MPI_Get_count(&status, MPI_BYTE, &size));
    int mpi_tag = status.MPI_TAG;
    assert(mpi_tag == tag);
    int source = status.MPI_SOURCE;
    m = make_unique<message>(source);
    m->buffer.resize(size);
    m->tag = mpi_tag;
    if (verb > 1)
        printf("[%2d] <- %d: receiving msg [tag %d], %d B, rqst %p\n", comm_rank(), source, mpi_tag, size, (void*)&m->request);

    TASKTORRENT_MPI_CHECK(MPI_Irecv(m->buffer.data(), m->buffer.size(), MPI_BYTE, source, mpi_tag, MPI_COMM_WORLD, &m->request));

    return true;
}

void Communicator::process_Ircvd_messages()
{
    list<unique_ptr<message>> messages_Ircvd_new;
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

            unique_ptr<Event> e;
            if (log)
                e = make_unique<Event>("rank_" + to_string(comm_rank()) + ">lpc>" + "rank_" + to_string(m->other) + ">" + to_string(m->tag));

            // Process the message
            process_message(m);

            if (log)
                logger->record(move(e));
        }
        else
            messages_Ircvd_new.push_back(move(m));
    }
    messages_Ircvd.swap(messages_Ircvd_new);
}

void Communicator::process_message(const unique_ptr<message> &m)
{
    Serializer<int> s;
    tuple<int> tup = s.read_buffer(m->buffer.data(), m->buffer.size());
    int am_id = get<0>(tup);
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
        unique_ptr<message> m;
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
        unique_ptr<message> m;
        bool success = probe_Irecv_message(m);
        if (success)
            messages_Ircvd.push_back(move(m));
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
        lock_guard<mutex> lock(messages_rdy_mtx);
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