#ifndef __TTOR_SRC_COMMUNICATIONS_MPI_HPP__
#define __TTOR_SRC_COMMUNICATIONS_MPI_HPP__

#ifdef TTOR_MPI

#include <utility>
#include <mutex>
#include <memory>
#include <functional>
#include <list>
#include <map>

#include <mpi.h>

#include "communications.hpp"
#include "views.hpp"
#include "mpi_utils.hpp"
#include "message.hpp"
#include "util_templates.hpp"
#include "serialization.hpp"
#include "apply_functions.hpp"

namespace ttor
{

struct message_MPI;

/**
 * \brief Handles all inter-ranks communications.
 * 
 * \details Object responsible for communications accross ranks.
 *          All MPI calls will be funneled through that object.
 */
class Communicator_MPI : public Communicator_Base
{

public:

    class ActiveMsg_Base;

    template <typename T, typename... Ps>
    class ActiveMsg;

private:

    const int my_rank;
    const static size_t mega = (1 << 20);
    const static size_t max_int_size = static_cast<size_t>(std::numeric_limits<int>::max());
    const MPI_Comm comm;
    MPI_Datatype MPI_MEGABYTE; // used to send large message larger than 4GB
    const size_t break_msg_size; // really mainly used for testing, where we make message artificially smaller, so we can actually test them

    std::vector<std::unique_ptr<ActiveMsg_Base>> active_messages;
    std::atomic<llint> messages_queued;    // queued messages
    std::atomic<llint> messages_processed; // received and processed messages

    /**
     * We have different channels for different kinds of messages
     *
     *  Two-steps message: Header           0 (count of 1 = 1 B)
     *                                      1 (count of 1 = 1 MB)
     *                     Body             2 (if header tag = 0) 
     *                                      3 (if header tag = 1)
     * 
     * The header Size (using B or MB) is encoded in the tag using 0 or 1
     * The bodies Size are encoded in the header, but use tag 2 or 3 depending on the header tag
     * 
     * We need those four tags to create different communication channels.
     * MPI messages are ordered between source and destination
     * With those tags we make sure that the MPI_Irecv and MPI_Isend are properly matching
     */

    /**
     * Where we store temporary data
     */

    // Sender side
    mutable std::mutex messages_rdy_mtx;
    // The messages pushed by the compute threads, protected by the above mutex
    std::list<std::unique_ptr<message_MPI>> messages_rdy;
    // The messages for which we called MPI_Isend
    std::list<std::unique_ptr<message_MPI>> messages_Isent;

    // Receiver side
    // The headers on which we called MPI_Irecv
    std::list<std::unique_ptr<message_MPI>> headers_Ircvd;
    // The bodies for which we called MPI_Irecv 
    std::list<std::unique_ptr<message_MPI>> bodies_Ircvd;

    /** 
     * Messages management Isend/Irecv
     */

    /**
     * Sender size
     */

    // Immediately Isend the message
    void Isend_header_body(std::unique_ptr<message_MPI> &m);

    // Send to myself, copies body, run AM and completion
    void self_Isend_header_body_process_complete(std::unique_ptr<message_MPI> &m);

    // Loop through the list of messages sent in by the task flow.
    // Isend all the messages in the ready queue.
    void Isend_queued_messages();

    // Test to see whether the Isent has completed or not
    // Free those Isent
    void test_Isent_messages();

    /**
     * Receiver side
     */
    
    // Probe for a message
    // If probe is true, then Irecv the header and returns true
    // Otherwise, returns false
    bool probe_Irecv_header(std::unique_ptr<message_MPI> &m);

    // Process the header
    void process_header(std::unique_ptr<message_MPI> &m);
    
    // Irecv the body
    void Irecv_body(std::unique_ptr<message_MPI> &m);

    // Process message
    void process_body(std::unique_ptr<message_MPI> &m);

    // Run the complete function when the body has been sent
    void process_completed_body(std::unique_ptr<message_MPI> &m);

    // Probe for incoming header
    // Starts MPI_Irecv for all probed headers
    void probe_Irecv_headers();

    // Process ready headers
    // If possible (taking ordering of MPI messages into account), Irecv their bodies
    void test_process_Ircvd_headers_Irecv_bodies();

    // Test completion of the bodies
    void test_process_bodies();

    // Sending AMs
    template <typename T, typename... Ps>
    std::unique_ptr<message_MPI> make_message_MPI(int dest, size_t am_id, const view<T>& body, Ps... ps);

    template <typename T, typename... Ps>
    friend class ActiveMsg;

    template <typename T, typename... Ps>
    void internal_send_large(int dest, size_t am_id, const view<T>& body, Ps... ps);

    template <typename T, typename... Ps>
    void internal_blocking_send_large(int dest, size_t am_id, const view<T>& body, Ps... ps);

public:

    /**
     * \brief Creates an Communicator_MPI.
     * 
     * \param[in] comm the MPI communicator to use in communications.
     * \param[in] verb the verbose level: 0 = no printing. > 0 = more and more printing.
     * \param[in] break_msg_size the size at which to break large messages into MPI messages. Mainly used for testing.
     * 
     * \pre `verb >= 0`.
     */
    Communicator_MPI(MPI_Comm comm = MPI_COMM_WORLD, int verb_ = 0, size_t break_msg_size_ = Communicator_MPI::max_int_size);

    /**
     * \brief Creates an active message
     */
    template <typename... Ps>
    ActiveMsg<char,Ps...> *make_active_msg(std::function<void(Ps...)> fun);

    /**
     * \brief Creates a large active message
     */
    template <typename T, typename... Ps>
    ActiveMsg<T,Ps...> *make_large_active_msg(std::function<void(Ps...)> fun, 
                                              std::function<T*(Ps...)> fun_ptr,
                                              std::function<void(Ps...)> fun_complete);

    /**
     * \brief Creates an active message
     */
    template<typename F>
    details::AM_t<ActiveMsg,decltype(&F::operator())> *make_active_msg(F f);

    /**
     * \brief Creates a large active message
     */
    template<typename F, typename G, typename H>
    details::Large_AM_t<ActiveMsg,decltype(&G::operator())> *make_large_active_msg(F f, G g, H h);

    /**
     * \brief Makes progress on the communications.
     * 
     * \details 
     * - Asynchronous (queue rpcs & in-flight lpcs) progress.
     * - Polls in Irecv and Isend request.
     * - Should be called from thread that called MPI_Init_Thread.
     * - Not thread safe.
     */
    virtual void progress() override;

    virtual bool is_done() const override;
    virtual llint get_n_msg_processed() const override;
    virtual llint get_n_msg_queued() const override;

    /**
     * \brief The rank within the MPI communicator
     * 
     * \return The MPI rank of the current processor
     */
    virtual int comm_rank() const override;

    /**
     * \brief The size of the MPI communicator
     * 
     * \return The number of MPI ranks within the communicator
     */
    virtual int comm_size() const override;

    virtual ~Communicator_MPI();

    class ActiveMsg_Base
    {

    private:

        const size_t id_;
        Communicator_MPI *comm_;

    public:

        size_t get_id() const;

        Communicator_MPI *get_comm() const;

        virtual void run(char *payload, size_t size) = 0;

        virtual char* get_user_buffers(char *payload, size_t size) = 0;

        virtual void complete(char *payload, size_t size) = 0;

        ActiveMsg_Base(Communicator_MPI *comm, size_t id);

        virtual ~ActiveMsg_Base();

    };

    template <typename T, typename... Ps>
    class ActiveMsg : public ActiveMsg_Base
    {

    private:

        const std::function<void(Ps...)> fun_;
        const std::function<T*(Ps...)> ptr_fun_;
        const std::function<void(Ps...)> complete_fun_;
        std::tuple<std::decay_t<Ps>...> get_payload(char *payload_raw, size_t size) const;

    public:

        ActiveMsg(std::function<void(Ps...)> fun, 
                  std::function<T*(Ps...)> ptr_fun, 
                  std::function<void(Ps...)> complete_fun, 
                  Communicator_MPI *comm,
                  size_t id);


        virtual void run(char *payload_raw, size_t size) override;
        
        virtual char* get_user_buffers(char *payload_raw, size_t size) override;

        virtual void complete(char *payload_raw, size_t size) override;

        void blocking_send(int dest, Ps... ps);
        
        void send(int dest, Ps... ps);

        void send_large(int dest, const view<T>& body, Ps... ps);

        void blocking_send_large(int dest, const view<T>& body, Ps... ps);

        virtual ~ActiveMsg();

    };

}; // Communicator_MPI

using Communicator = Communicator_MPI;

} // namespace ttor

/** Implementation **/

namespace ttor {

template<typename T, typename... Ps>
std::unique_ptr<message_MPI> Communicator_MPI::make_message_MPI(int dest, size_t am_id, const view<T>& body, Ps... ps) {

    auto m = std::make_unique<message_MPI>();

    m->source = my_rank;
    m->dest = dest;

    size_t body_size = body.size() * sizeof(T);
    m->body_send_buffer = (const char*)body.data();
    m->body_size = body_size;

    Serializer<size_t, size_t, Ps...> s;
    size_t header_size = s.size(am_id, body_size, ps...);
    size_t header_required_size = header_size;
    m->header_tag = 0;
    if(header_size > break_msg_size) {
        m->header_tag = 1;
        header_required_size = mega * ((header_size + mega - 1) / mega);
    }
    m->header_buffer->resize(header_required_size);
    s.write_buffer(m->header_buffer->data(), m->header_buffer->size(), am_id, body_size, ps...);

    return m;
}

template<typename T, typename... Ps>
void Communicator_MPI::internal_send_large(int dest, size_t am_id, const view<T>& body, Ps... ps) {
    // Increment message counter
    messages_queued++;
    auto m = make_message_MPI(dest, am_id, body, ps...);
    std::lock_guard<std::mutex> lock(messages_rdy_mtx);
    messages_rdy.push_back(std::move(m));
}

template<typename T, typename... Ps>
void Communicator_MPI::internal_blocking_send_large(int dest, size_t am_id, const view<T>& body, Ps... ps) {
    // Increment message counter
    messages_queued++;
    auto m = make_message_MPI(dest, am_id, body, ps...);
    Isend_header_body(m);
    TASKTORRENT_MPI_CHECK(MPI_Wait(&m->header_request, MPI_STATUS_IGNORE));
    TASKTORRENT_MPI_CHECK(MPI_Waitall(m->body_requests.size(), m->body_requests.data(), MPI_STATUSES_IGNORE));
    process_completed_body(m);
    if(verb > 1) 
        printf("[%3d] -> %3d: header and body blocking sent completed [tags %d and %d], sizes %zd and %zd B\n", my_rank, m->dest, m->header_tag, m->body_tag, m->header_buffer->size(), m->body_size);
}

// Create active messages
template <typename... Ps>
Communicator_MPI::ActiveMsg<char,Ps...> *Communicator_MPI::make_active_msg(std::function<void(Ps...)> fun)
{
    std::function<char*(Ps...)> fun_ptr = [](Ps...) {
        return nullptr;
    };
    std::function<void(Ps...)> fun_complete = [](Ps...) {
        return;
    };
    return make_large_active_msg(fun, fun_ptr, fun_complete);
}

// Create large active messages
template <typename T, typename... Ps>
Communicator_MPI::ActiveMsg<T,Ps...> *Communicator_MPI::make_large_active_msg(std::function<void(Ps...)> fun, 
                                                                              std::function<T*(Ps...)> fun_ptr,
                                                                              std::function<void(Ps...)> fun_complete)
{
    auto am = std::make_unique<Communicator_MPI::ActiveMsg<T,Ps...>>(fun, fun_ptr, fun_complete, this, active_messages.size());
    auto am_ = am.get();
    active_messages.push_back(move(am));

    if (verb > 0)
    {
        printf("[%2d]: created lpc() with ID %lu\n", comm_rank(), active_messages.size() - 1);
    }

    return am_;
}

template<typename F>
details::AM_t<Communicator_MPI::ActiveMsg,decltype(&F::operator())>
  *Communicator_MPI::make_active_msg(F f) {
    auto fun = details::GetStdFunction(f);
    return make_active_msg(fun);
}

template<typename F, typename G, typename H>
details::Large_AM_t<Communicator_MPI::ActiveMsg,decltype(&G::operator())>
  *Communicator_MPI::make_large_active_msg(F f, G g, H h) {
    auto fun = details::GetStdFunction(f);
    auto fun_ptr = details::GetStdFunction(g);
    auto fun_complete = details::GetStdFunction(h);
    return make_large_active_msg(fun, fun_ptr, fun_complete);
}

/**
 * ActiveMsg
 */

template <typename T, typename... Ps>
Communicator_MPI::ActiveMsg<T,Ps...>::ActiveMsg(std::function<void(Ps...)> fun, 
                                                std::function<T*(Ps...)> ptr_fun, 
                                                std::function<void(Ps...)> complete_fun, 
                                                Communicator_MPI *comm,
                                                size_t id) : ActiveMsg_Base(comm, id), fun_(fun), ptr_fun_(ptr_fun), complete_fun_(complete_fun) {}

template <typename T, typename... Ps>
std::tuple<std::decay_t<Ps>...> Communicator_MPI::ActiveMsg<T, Ps...>::get_payload(char *payload_raw, size_t size) const
{
    // ID, body size, header args...
    Serializer<size_t, size_t, std::decay_t<Ps>...> s;
    auto tup = s.read_buffer(payload_raw, size);
    assert(std::get<0>(tup) == get_id());
    return tail(tail(tup));
}

template <typename T, typename... Ps>
void Communicator_MPI::ActiveMsg<T, Ps...>::run(char *payload_raw, size_t size)
{
    auto args = get_payload(payload_raw, size);
    apply_fun(this->fun_, args);
}

template <typename T, typename... Ps>
void Communicator_MPI::ActiveMsg<T, Ps...>::complete(char *payload_raw, size_t size)
{
    auto args = get_payload(payload_raw, size);
    apply_fun(this->complete_fun_, args);
}

template <typename T, typename... Ps>
char* Communicator_MPI::ActiveMsg<T, Ps...>::get_user_buffers(char *payload_raw, size_t size)
{
    auto args = get_payload(payload_raw, size);
    return (char*) apply_fun(this->ptr_fun_, args);
}

template <typename T, typename... Ps>
void Communicator_MPI::ActiveMsg<T, Ps...>::blocking_send(int dest, Ps... ps)
{
    view<T> body;
    this->get_comm()->internal_blocking_send_large(dest, this->get_id(), body, ps...);
}

template <typename T, typename... Ps>
void Communicator_MPI::ActiveMsg<T, Ps...>::send(int dest, Ps... ps)
{
    view<T> body;
    this->get_comm()->internal_send_large(dest, this->get_id(), body, ps...);
}

template <typename T, typename... Ps>
void Communicator_MPI::ActiveMsg<T, Ps...>::send_large(int dest, const view<T>& body, Ps... ps)
{
    this->get_comm()->internal_send_large(dest, this->get_id(), body, ps...);
}

template <typename T, typename... Ps>
void Communicator_MPI::ActiveMsg<T, Ps...>::blocking_send_large(int dest, const view<T>& body, Ps... ps)
{
    this->get_comm()->internal_blocking_send_large(dest, this->get_id(), body, ps...);
}

template <typename T, typename... Ps>
Communicator_MPI::ActiveMsg<T, Ps...>::~ActiveMsg() = default;

} // namespace ttor

#endif

#endif
