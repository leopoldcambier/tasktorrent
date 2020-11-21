#ifdef TTOR_MPI

#include <list>
#include <mutex>
#include <tuple>
#include <cassert>
#include <set>

#include "communications_mpi.hpp"
#include "mpi_utils.hpp"
#include "serialization.hpp"
#include "util.hpp"
#include "message.hpp"

namespace ttor
{

/**
 * Communicator_MPI
 */

int Communicator_MPI::comm_rank() const
{
    return ttor::mpi_comm_rank(comm);
}

int Communicator_MPI::comm_size() const
{
    return ttor::mpi_comm_size(comm);
}

llint Communicator_MPI::get_n_msg_processed() const
{
    return messages_processed.load();
}

llint Communicator_MPI::get_n_msg_queued() const
{
    return messages_queued.load();
}

Communicator_MPI::Communicator_MPI(MPI_Comm comm_, int verb_, size_t break_msg_size_) : 
    Communicator_Base(verb_),
    my_rank(ttor::mpi_comm_rank(comm_)),
    comm(comm_),
    break_msg_size(break_msg_size_), messages_queued(0), messages_processed(0) {
        TASKTORRENT_MPI_CHECK(MPI_Type_contiguous(static_cast<int>(mega), MPI_BYTE, &MPI_MEGABYTE));
        TASKTORRENT_MPI_CHECK(MPI_Type_commit(&MPI_MEGABYTE));
        assert(break_msg_size <= max_int_size);
        assert(break_msg_size >= static_cast<size_t>(mega));
    }

void Communicator_MPI::self_Isend_header_body_process_complete(std::unique_ptr<message_MPI> &m)
{
    assert(m->source == my_rank);
    assert(m->dest == my_rank);
    // 1. Fetch location on receiver
    process_header(m);
    // 2. Copy the view (~Isend/Irecv for body)
    if(m->body_size > 0) { // memcpy cannot technically be called with nullptr source/dest even if size is 0
        memcpy(m->body_recv_buffer, m->body_send_buffer, m->body_size);
    }
    // 3. Run the AM
    process_body(m);
    // 4. Complete
    process_completed_body(m);
    if(verb > 1)
        printf("[%3d] -> %3d: header and body non-blocking local sent completed [tags %d and %d], sizes %zd and %zd B\n", my_rank, m->dest, m->header_tag, m->body_tag, m->header_buffer->size(), m->body_size);
}

void Communicator_MPI::Isend_header_body(std::unique_ptr<message_MPI> &m)
{
    // Send the header
    if(m->header_tag == 0) {
        const size_t size = m->header_buffer->size();
        assert(size <= break_msg_size);
        TASKTORRENT_MPI_CHECK(MPI_Isend(m->header_buffer->data(), static_cast<int>(size), MPI_BYTE, m->dest, m->header_tag, comm, &(m->header_request)));
    } else if(m->header_tag == 1) {
        assert(m->header_buffer->size() > break_msg_size);
        assert(m->header_buffer->size() % mega == 0);
        const size_t size = m->header_buffer->size() / mega;
        if(size > max_int_size) {
            printf("Error in Communicator_MPI::Isend_message: requested message size of %zd larger than maximum allowed\n", m->header_buffer->size());
            MPI_Finalize();
            exit(1);
        }
        TASKTORRENT_MPI_CHECK(MPI_Isend(m->header_buffer->data(), static_cast<int>(size), MPI_MEGABYTE, m->dest, m->header_tag, comm, &(m->header_request)));
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
    const char* start = m->body_send_buffer;
    size_t size = m->body_size;
    if(start == nullptr) assert(n_bodies == 0 && size == 0);
    for(int i = 0; i < n_bodies; i++) {
        assert(size > 0);
        const size_t to_send = (size >= break_msg_size ? break_msg_size : size);
        assert(to_send <= max_int_size);
        TASKTORRENT_MPI_CHECK(MPI_Isend(start, static_cast<int>(to_send), MPI_BYTE, m->dest, m->body_tag, comm, &(m->body_requests[i])));
        size -= to_send;
        start += to_send;
    }
    assert(size == 0);

    if (verb > 1)
        printf("[%3d] -> %3d: sending header & bodies [tags %d, %d], n_bodies %d, sizes %zd, %zd B\n", my_rank, m->dest, m->header_tag, m->body_tag, n_bodies, m->header_buffer->size(), m->body_size);
}

// Return true if there is a message and we started an Irecv; false otherwise
bool Communicator_MPI::probe_Irecv_header(std::unique_ptr<message_MPI> &m)
{
    MPI_Status mpi_status;
    int mpi_size, mpi_flag; // MPI uses int for message count (mpi_size)
    for(int tag = 0; tag < 2; tag++) {
        TASKTORRENT_MPI_CHECK(MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &mpi_flag, &mpi_status));
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
    m = std::make_unique<message_MPI>();
    m->source = source;
    m->dest = my_rank;
    if(mpi_tag == 0) { // We are receiving MPI_BYTE
        TASKTORRENT_MPI_CHECK(MPI_Get_count(&mpi_status, MPI_BYTE, &mpi_size));
        buffer_size = static_cast<size_t>(mpi_size);
    } else if(mpi_tag == 1) { // We are receiving MPI_MEGABYTE
        TASKTORRENT_MPI_CHECK(MPI_Get_count(&mpi_status, MPI_MEGABYTE, &mpi_size));
        buffer_size = static_cast<size_t>(mpi_size) * mega;
    } else {
        assert(false);
    }
    m->header_buffer->resize(buffer_size);
    m->header_tag = mpi_tag;
    if (verb > 1)
        printf("[%3d] <- %3d: receiving header [tag %d], size %zd B\n", my_rank, m->source, mpi_tag, buffer_size);

    if(mpi_tag == 0) { // We are receiving MPI_BYTE
        TASKTORRENT_MPI_CHECK(MPI_Irecv(m->header_buffer->data(), mpi_size, MPI_BYTE, m->source, m->header_tag, comm, &m->header_request));
    } else if(mpi_tag == 1) {
        TASKTORRENT_MPI_CHECK(MPI_Irecv(m->header_buffer->data(), mpi_size, MPI_MEGABYTE, m->source, m->header_tag, comm, &m->header_request));
    } else {
        assert(false);
    }

    return true;
}

void Communicator_MPI::Irecv_body(std::unique_ptr<message_MPI> &m) {

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

    char* start = m->body_recv_buffer;
    size_t size = m->body_size;
    assert(size > 0);
    if (verb > 1) {
        printf("[%3d] <- %3d: receiving bodies [tag %d], size %zd B, num bodies %zd\n", my_rank, m->source, m->body_tag, m->body_size, n_bodies);
    }
    for(size_t i = 0; i < n_bodies; i++) {
        assert(size > 0);
        const size_t to_recv = (size >= break_msg_size ? break_msg_size : size);
        assert(to_recv <= max_int_size);
        TASKTORRENT_MPI_CHECK(MPI_Irecv(start, static_cast<int>(to_recv), MPI_BYTE, m->source, m->body_tag, comm, &m->body_requests[i]));
        size -= to_recv;
        start += to_recv;
    }
    assert(size == 0);
}

void Communicator_MPI::process_header(std::unique_ptr<message_MPI> &m) {
    assert(! m->header_processed);
    Serializer<size_t,size_t> s;
    std::tuple<size_t,size_t> tup = s.read_buffer(m->header_buffer->data(), m->header_buffer->size());
    const size_t am_id = std::get<0>(tup);
    const size_t body_size = std::get<1>(tup);
    m->body_size = body_size;
    assert(am_id < active_messages.size());
    m->body_recv_buffer = active_messages.at(am_id)->get_user_buffers(m->header_buffer->data(), m->header_buffer->size());
    m->header_processed = true;
}

void Communicator_MPI::process_body(std::unique_ptr<message_MPI> &m) {
    Serializer<size_t> s;
    std::tuple<size_t> tup = s.read_buffer(m->header_buffer->data(), m->header_buffer->size());
    const size_t am_id = std::get<0>(tup);
    assert(am_id < active_messages.size());
    active_messages.at(am_id)->run(m->header_buffer->data(), m->header_buffer->size());
    messages_processed++;
}

void Communicator_MPI::process_completed_body(std::unique_ptr<message_MPI> &m) {
    Serializer<size_t> s;
    std::tuple<size_t> tup = s.read_buffer(m->header_buffer->data(), m->header_buffer->size());
    const size_t am_id = std::get<0>(tup);
    assert(am_id < active_messages.size());
    active_messages.at(am_id)->complete(m->header_buffer->data(), m->header_buffer->size());
}

void Communicator_MPI::Isend_queued_messages()
{
    std::list<std::unique_ptr<message_MPI>> to_Isend;
    {
        std::lock_guard<std::mutex> lock(messages_rdy_mtx);
        to_Isend.swap(messages_rdy);
        assert(messages_rdy.size() == 0);
    }
    for (auto &m : to_Isend)
    {
        assert(m->source == my_rank);
        if(m->dest == my_rank) {
            self_Isend_header_body_process_complete(m);
        } else {
            Isend_header_body(m);
            messages_Isent.push_back(std::move(m));
        }
    }
}

void Communicator_MPI::test_Isent_messages()
{
    std::list<std::unique_ptr<message_MPI>> messages_Isent_new;
    for (auto &m : messages_Isent)
    {
        int flag_header = 0;
        TASKTORRENT_MPI_CHECK(MPI_Test(&m->header_request, &flag_header, MPI_STATUS_IGNORE));
        int flag_bodies = 0;
        TASKTORRENT_MPI_CHECK(MPI_Testall(m->body_requests.size(), m->body_requests.data(), 
                                          &flag_bodies, MPI_STATUSES_IGNORE));
        if (flag_header && flag_bodies) { // Header and bodies sends are completed
            process_completed_body(m);
            if (verb > 1)
                printf("[%3d] -> %3d: header and body non-blocking sent completed [tags %d and %d], sizes %zd and %zd B\n", my_rank, m->dest, m->header_tag, m->body_tag, m->header_buffer->size(), m->body_size);
        } else {
            messages_Isent_new.push_back(std::move(m));
        }
    }
    messages_Isent.swap(messages_Isent_new);
}

void Communicator_MPI::probe_Irecv_headers() 
{
    // Keep checking for Irecv messages and insert into queue
    while (true)
    {
        std::unique_ptr<message_MPI> m;
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
void Communicator_MPI::test_process_Ircvd_headers_Irecv_bodies()
{
    std::set<int> first_with_body_seen; // if rank is present, it mean there is a message from rank before in the queue
    std::list<std::unique_ptr<message_MPI>> headers_Ircvd_new;
    for (auto &m : headers_Ircvd)
    {
        int header_flag = 0;
        TASKTORRENT_MPI_CHECK(MPI_Test(&m->header_request, &header_flag, MPI_STATUS_IGNORE));
        if (header_flag)
        {
            if(! m->header_processed) {
                if(verb > 1)
                    printf("[%3d] <- %3d: header receive completed [tag %d], size %zd B\n", my_rank, m->source, m->header_tag, m->header_buffer->size());
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
             * FIXME: this is too conservative, we can also discriminate per tags
             */
            } else if(first_with_body_seen.count(m->source) == 0) {
                first_with_body_seen.insert(m->source);
                Irecv_body(m);
                bodies_Ircvd.push_back(std::move(m));
            /**
             * If there _is_ a body coming later, but we are not the latest header from that particular rank, we have to wait before calling MPI_Irecv
             */
            } else {
                assert(first_with_body_seen.count(m->source) == 1);
                headers_Ircvd_new.push_back(std::move(m));
            }
        } else {
            first_with_body_seen.insert(m->source);
            headers_Ircvd_new.push_back(std::move(m));
        }
    }
    headers_Ircvd.swap(headers_Ircvd_new);
}

void Communicator_MPI::test_process_bodies()
{
    std::list<std::unique_ptr<message_MPI>> bodies_Ircvd_new;
    for(auto& m: bodies_Ircvd) {
        int body_flag = 0;
        assert(m->header_processed);
        TASKTORRENT_MPI_CHECK(MPI_Testall(m->body_requests.size(), m->body_requests.data(), &body_flag, MPI_STATUSES_IGNORE));
        if (body_flag) {
            if(verb > 1)
                printf("[%3d] <- %3d: body receive completed [tag %d], size %zd B\n", my_rank, m->source, m->body_tag, m->body_size);
            process_body(m);
        } else {
            bodies_Ircvd_new.push_back(std::move(m));
        }
    }
    bodies_Ircvd.swap(bodies_Ircvd_new);
}

void Communicator_MPI::progress()
{
    Isend_queued_messages();
    test_Isent_messages();
    probe_Irecv_headers();
    test_process_Ircvd_headers_Irecv_bodies();
    test_process_bodies();
}

bool Communicator_MPI::is_done() const
{
    bool ret;
    {
        std::lock_guard<std::mutex> lock(messages_rdy_mtx);
        ret = messages_rdy.empty() && messages_Isent.empty() && headers_Ircvd.empty() && bodies_Ircvd.empty();
    }
    return ret;
}

Communicator_MPI::~Communicator_MPI() = default;

/**
 * ActiveMsg_Base
 */

size_t Communicator_MPI::ActiveMsg_Base::get_id() const { return id_; }
Communicator_MPI *Communicator_MPI::ActiveMsg_Base::get_comm() const { return comm_; }
Communicator_MPI::ActiveMsg_Base::ActiveMsg_Base(Communicator_MPI *comm, size_t id) : id_(id), comm_(comm) {}
Communicator_MPI::ActiveMsg_Base::~ActiveMsg_Base() = default;

} // namespace ttor

#endif