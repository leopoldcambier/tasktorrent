#ifndef TTOR_SHARED

#include <list>
#include <mutex>
#include <tuple>
#include <cassert>
#include <set>

#include "communications.hpp"
#include "mpi_utils.hpp"
#include "serialization.hpp"
#include "active_messages.hpp"

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

Communicator::Communicator(int verb_, size_t break_msg_size_) : 
    verb(verb_), 
    logger(nullptr), 
    log(false), 
    messages_queued(0), 
    messages_processed(0),
    break_msg_size(break_msg_size_) {
        TASKTORRENT_MPI_CHECK(MPI_Type_contiguous(static_cast<int>(mega), MPI_BYTE, &MPI_MEGABYTE));
        TASKTORRENT_MPI_CHECK(MPI_Type_commit(&MPI_MEGABYTE));
        assert(break_msg_size <= max_int_size);
        assert(break_msg_size >= static_cast<size_t>(mega));
    }
    
void Communicator::set_logger(Logger *logger_)
{
    log = true;
    logger = logger_;
}

std::unique_ptr<message> Communicator::make_active_message(int dest, size_t header_size)
{
    auto m = std::make_unique<message>();
    m->other = dest;
    size_t header_buffer_size = 0;
    if(header_size > break_msg_size) {
        header_buffer_size = mega * ((header_size + mega - 1) / mega); // pad so that the total size if a multiple of mega (2^20)
        m->header_tag = 1;
    } else {
        header_buffer_size = header_size;
        m->header_tag = 0;
    }
    m->header_buffer.resize(header_buffer_size);
    return m;
}

void Communicator::queue_message(std::unique_ptr<message> m)
{
    // Increment message counter
    messages_queued++;
    std::lock_guard<std::mutex> lock(messages_rdy_mtx);
    messages_rdy.push_back(std::move(m));
}

void Communicator::Isend_header_body(std::unique_ptr<message> &m)
{
    // Send the header
    if(m->header_tag == 0) {
        const size_t size = m->header_buffer.size();
        assert(size <= break_msg_size);
        TASKTORRENT_MPI_CHECK(MPI_Isend(m->header_buffer.data(), static_cast<int>(size), MPI_BYTE, m->other, m->header_tag, MPI_COMM_WORLD, &(m->header_request)));
    } else if(m->header_tag == 1) {
        assert(m->header_buffer.size() > break_msg_size);
        assert(m->header_buffer.size() % mega == 0);
        const size_t size = m->header_buffer.size() / mega;
        if(size > max_int_size) {
            printf("Error in Communicator::Isend_message: requested message size of %zd larger than maximum allowed\n", m->header_buffer.size());
            MPI_Finalize();
            exit(1);
        }
        TASKTORRENT_MPI_CHECK(MPI_Isend(m->header_buffer.data(), static_cast<int>(size), MPI_MEGABYTE, m->other, m->header_tag, MPI_COMM_WORLD, &(m->header_request)));
    } else {
        assert(false);
    }

    // Bodies info
    const int n_bodies = (m->body_size + break_msg_size - 1) / break_msg_size;
    m->body_requests.resize(n_bodies);
    if(m->header_tag == 0)
        m->body_tag = 2;
    else 
        m->body_tag = 3;

    // Send the bodies
    char* start = m->body_buffer;
    size_t size = m->body_size;
    if(start == nullptr) assert(n_bodies == 0 && size == 0);
    for(int i = 0; i < n_bodies; i++) {
        assert(size > 0);
        const size_t to_send = (size >= break_msg_size ? break_msg_size : size);
        assert(to_send <= max_int_size);
        TASKTORRENT_MPI_CHECK(MPI_Isend(start, static_cast<int>(to_send), MPI_BYTE, m->other, m->body_tag, MPI_COMM_WORLD, &(m->body_requests[i])));
        size -= to_send;
        start += to_send;
    }
    assert(size == 0);

    if (verb > 1)
        printf("[%3d] -> %3d: sending header & bodies [tags %d, %d], n_bodies %d, sizes %zd, %zd B\n", comm_rank(), m->other, m->header_tag, m->body_tag, n_bodies, m->header_buffer.size(), m->body_size);
}

// Return true if there is a message and we started an Irecv; false otherwise
bool Communicator::probe_Irecv_header(std::unique_ptr<message> &m)
{
    MPI_Status mpi_status;
    int mpi_size, mpi_flag; // MPI uses int for message count (mpi_size)
    for(int tag = 0; tag < 2; tag++) {
        TASKTORRENT_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &mpi_flag, &mpi_status));
        if(mpi_flag) {
            break;
        }
    }
    if (!mpi_flag) {
        return false;
    }

    int mpi_tag = mpi_status.MPI_TAG;
    int source = mpi_status.MPI_SOURCE;
    size_t buffer_size = 0;
    m = std::make_unique<message>();
    m->other = source;
    if(mpi_tag == 0) { // We are receiving MPI_BYTE
        TASKTORRENT_MPI_CHECK(MPI_Get_count(&mpi_status, MPI_BYTE, &mpi_size));
        buffer_size = static_cast<size_t>(mpi_size);
    } else if(mpi_tag == 1) { // We are receiving MPI_MEGABYTE
        TASKTORRENT_MPI_CHECK(MPI_Get_count(&mpi_status, MPI_MEGABYTE, &mpi_size));
        buffer_size = static_cast<size_t>(mpi_size) * mega;
    } else {
        assert(false);
    }
    m->header_buffer.resize(buffer_size);
    m->header_tag = mpi_tag;
    if (verb > 1)
        printf("[%3d] <- %3d: receiving header [tag %d], size %zd B\n", comm_rank(), source, mpi_tag, buffer_size);

    if(mpi_tag == 0) { // We are receiving MPI_BYTE
        TASKTORRENT_MPI_CHECK(MPI_Irecv(m->header_buffer.data(), mpi_size, MPI_BYTE, m->other, m->header_tag, MPI_COMM_WORLD, &m->header_request));
    } else if(mpi_tag == 1) {
        TASKTORRENT_MPI_CHECK(MPI_Irecv(m->header_buffer.data(), mpi_size, MPI_MEGABYTE, m->other, m->header_tag, MPI_COMM_WORLD, &m->header_request));
    } else {
        assert(false);
    }

    return true;
}

void Communicator::Irecv_body(std::unique_ptr<message> &m) {

    if(m->body_size == 0) return;

    const size_t n_bodies = (m->body_size + break_msg_size - 1) / break_msg_size;
    m->body_requests.resize(n_bodies);
    if(m->header_tag == 0) {
        m->body_tag = 2;
    } else if(m->header_tag == 1) {
        m->body_tag = 3;
    } else {
        assert(false);
    }

    char* start = m->body_buffer;
    size_t size = m->body_size;
    assert(size > 0);
    if (verb > 1) {
        printf("[%3d] <- %3d: receiving bodies [tag %d], size %zd B, num bodies %zd\n", comm_rank(), m->other, m->body_tag, m->body_size, n_bodies);
    }
    for(size_t i = 0; i < n_bodies; i++) {
        assert(size > 0);
        const size_t to_recv = (size >= break_msg_size ? break_msg_size : size);
        assert(to_recv <= max_int_size);
        TASKTORRENT_MPI_CHECK(MPI_Irecv(start, static_cast<int>(to_recv), MPI_BYTE, m->other, m->body_tag, MPI_COMM_WORLD, &m->body_requests[i]));
        size -= to_recv;
        start += to_recv;
    }
    assert(size == 0);
}

void Communicator::process_header(std::unique_ptr<message> &m) {
    assert(! m->header_processed);
    Serializer<size_t,size_t> s;
    std::tuple<size_t,size_t> tup = s.read_buffer(m->header_buffer.data(), m->header_buffer.size());
    const size_t am_id = std::get<0>(tup);
    const size_t body_size = std::get<1>(tup);
    m->body_size = body_size;
    assert(am_id >= 0 && am_id < active_messages.size());
    m->body_buffer = active_messages.at(am_id)->get_user_buffers(m->header_buffer.data(), m->header_buffer.size());
    m->header_processed = true;
}

void Communicator::process_body(std::unique_ptr<message> &m) {
    Serializer<size_t> s;
    std::tuple<size_t> tup = s.read_buffer(m->header_buffer.data(), m->header_buffer.size());
    const size_t am_id = std::get<0>(tup);
    assert(am_id >= 0 && am_id < active_messages.size());
    active_messages.at(am_id)->run(m->header_buffer.data(), m->header_buffer.size());
    messages_processed++;
}

void Communicator::Isend_queued_messages()
{
    std::list<std::unique_ptr<message>> to_Isend;
    {
        std::lock_guard<std::mutex> lock(messages_rdy_mtx);
        to_Isend.swap(messages_rdy);
        assert(messages_rdy.size() == 0);
    }
    for (auto &m : to_Isend)
    {
        Isend_header_body(m);
        messages_Isent.push_back(std::move(m));
    }
}

void Communicator::test_Isent_messages()
{
    std::list<std::unique_ptr<message>> messages_Isent_new;
    for (auto &m : messages_Isent)
    {
        int flag_header = 0;
        TASKTORRENT_MPI_CHECK(MPI_Test(&m->header_request, &flag_header, MPI_STATUS_IGNORE));
        int flag_bodies = 0;
        TASKTORRENT_MPI_CHECK(MPI_Testall(m->body_requests.size(), m->body_requests.data(), 
                                          &flag_bodies, MPI_STATUSES_IGNORE));
        if (flag_header && flag_bodies) { // Header and bodies sends are completed
            if (verb > 1)
                printf("[%3d] -> %3d: header and body sent [tags %d and %d], sizes %zd and %zd B\n", comm_rank(), m->other, m->header_tag, m->body_tag, m->header_buffer.size(), m->body_size);
        } else {
            messages_Isent_new.push_back(std::move(m));
        }
    }
    messages_Isent.swap(messages_Isent_new);
}

void Communicator::probe_Irecv_headers() 
{
    // Keep checking for Irecv messages and insert into queue
    while (true)
    {
        std::unique_ptr<message> m;
        const bool success = probe_Irecv_header(m);
        if (success) {
            headers_Ircvd.push_back(std::move(m));
        } else {
            // Iprobe says there are no messages in the pipeline
            break;
        }
    }
}

/**
 * Most of the complexity is here
 * 
 * We are receiving headers and bodies from all ranks, in order
 * For each quadruplet (source, dest, tag_header, tag_body), we have an ordered communication channel
 * We need to make sure we Irecv messages in the same order they are Isent
 */
void Communicator::test_process_Ircvd_headers_Irecv_bodies()
{
    std::set<int> first_with_body_seen; // if rank is present, it mean there is a message from rank before in the queue
    std::list<std::unique_ptr<message>> headers_Ircvd_new;
    for (auto &m : headers_Ircvd)
    {
        int header_flag = 0;
        TASKTORRENT_MPI_CHECK(MPI_Test(&m->header_request, &header_flag, MPI_STATUS_IGNORE));
        if (header_flag)
        {
            if(! m->header_processed) {
                if(verb > 1)
                    printf("[%3d] <- %3d: header receive completed [tag %d], size %zd B\n", comm_rank(), m->other, m->header_tag, m->header_buffer.size());
                process_header(m);
            }
            /**
             * If there is no body so order doesn't matter
             */
            if(m->body_size == 0) {
                bodies_Ircvd.push_back(std::move(m));
            /**
             * This is important !
             * All the complexity is from the fact that we need to order the Isent and Ircvd messages
             * Since bodies are always sent after the header
             * We can only receive the bodies in the same order the headers are received
             * FIXNE: this is too conservative, we can also discriminate per tags
             */
            } else if(first_with_body_seen.count(m->other) == 0) {
                first_with_body_seen.insert(m->other);
                Irecv_body(m);
                bodies_Ircvd.push_back(std::move(m));
            /**
             * If there _is_ a body coming later, but we are not the latest header from that particular rank, we have to wait before calling MPI_Irecv
             */
            } else {
                assert(first_with_body_seen.count(m->other) == 1);
                headers_Ircvd_new.push_back(std::move(m));
            }
        } else {
            first_with_body_seen.insert(m->other);
            headers_Ircvd_new.push_back(std::move(m));
        }
    }
    headers_Ircvd.swap(headers_Ircvd_new);
}

void Communicator::test_process_bodies()
{
    std::list<std::unique_ptr<message>> bodies_Ircvd_new;
    for(auto& m: bodies_Ircvd) {
        int body_flag = 0;
        assert(m->header_processed);
        TASKTORRENT_MPI_CHECK(MPI_Testall(m->body_requests.size(), m->body_requests.data(), &body_flag, MPI_STATUSES_IGNORE));
        if (body_flag) {
            if(verb > 1)
                printf("[%3d] <- %3d: body receive completed [tag %d], size %zd B\n", comm_rank(), m->other, m->body_tag, m->body_size);
            process_body(m);
        } else {
            bodies_Ircvd_new.push_back(std::move(m));
        }
    }
    bodies_Ircvd.swap(bodies_Ircvd_new);
}

void Communicator::progress()
{
    Isend_queued_messages();
    test_Isent_messages();
    probe_Irecv_headers();
    test_process_Ircvd_headers_Irecv_bodies();
    test_process_bodies();
}

// Blocking versions
void Communicator::blocking_send(std::unique_ptr<message> m) {
    messages_queued++;
    Isend_header_body(m);
    TASKTORRENT_MPI_CHECK(MPI_Wait(&m->header_request, MPI_STATUS_IGNORE));
    TASKTORRENT_MPI_CHECK(MPI_Waitall(m->body_requests.size(), m->body_requests.data(), MPI_STATUSES_IGNORE));
    if(verb > 1) 
        printf("[%3d] -> %3d: header and body sent completed [tags %d and %d], sizes %zd and %zd B\n", comm_rank(), m->other, m->header_tag, m->body_tag, m->header_buffer.size(), m->body_size);
}

void Communicator::recv_process() {
    while (true)
    {
        std::unique_ptr<message> m;
        bool success = probe_Irecv_header(m);
        if (success) {
            TASKTORRENT_MPI_CHECK(MPI_Wait(&m->header_request, MPI_STATUS_IGNORE));
            if(verb > 1) 
                printf("[%3d] <- %3d: header completed [tag %d], size %zd B\n", comm_rank(), m->other, m->header_tag, m->header_buffer.size());
            process_header(m);
            Irecv_body(m);
            TASKTORRENT_MPI_CHECK(MPI_Waitall(m->body_requests.size(), m->body_requests.data(), MPI_STATUSES_IGNORE));
            if(verb > 1)
                printf("[%3d] <- %3d: bodies completed [tag %d], size %zd B\n", comm_rank(), m->other, m->body_tag, m->body_size);
            process_body(m);
            break;
        }
    }
}

bool Communicator::is_done()
{
    bool ret;
    {
        std::lock_guard<std::mutex> lock(messages_rdy_mtx);
        ret = messages_rdy.empty() && messages_Isent.empty() && headers_Ircvd.empty() && bodies_Ircvd.empty();
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