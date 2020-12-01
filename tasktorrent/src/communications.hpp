#ifndef __TTOR_SRC_COMMUNICATIONS_HPP__
#define __TTOR_SRC_COMMUNICATIONS_HPP__

#ifndef TTOR_SHARED

#include <utility>
#include <mutex>
#include <memory>
#include <functional>
#include <list>
#include <map>

#include <mpi.h>

#include "util.hpp"
#include "message.hpp"
#include "mpi_utils.hpp"
#include "functional_extra.hpp"

namespace ttor
{

class ActiveMsgBase;

template <typename... Ps>
class ActiveMsg;

/**
 * Extract the type of a std::function or lambda 
 * and return ActiveMsg<Args...>
 */
template <typename T>
struct ActiveMsg_type
{
    using type = void;
};

template <typename Ret, typename Class, typename... Args>
struct ActiveMsg_type<Ret (Class::*)(Args &...) const>
{
    using type = ActiveMsg<Args...>;
};

/**
 * \brief Handles all inter-ranks communications.
 * 
 * \details Object responsible for communications accross ranks.
 *          All MPI calls will be funneled through that object.
 */
class Communicator
{

private:

    const static size_t mega = (1 << 20);
    const static size_t max_int_size = static_cast<size_t>(std::numeric_limits<int>::max());
    const MPI_Comm comm;
    const int my_rank;
    const int verb;
    Logger *logger;
    bool log;
    std::vector<std::unique_ptr<ActiveMsgBase>> active_messages;
    std::atomic<int> messages_queued;    // queued messages
    std::atomic<int> messages_processed; // received and processed messages
    MPI_Datatype MPI_MEGABYTE; // used to send large message larger than 4GB
    const size_t break_msg_size; // really mainly used for testing, where we make message artificially smaller, so we can actually test them
    size_t sleep_time_us;

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
    std::mutex messages_rdy_mtx;
    // The messages pushed by the compute threads, protected by the above mutex
    std::list<std::unique_ptr<message>> messages_rdy;
    // The messages for which we called MPI_Isend
    std::list<std::unique_ptr<message>> messages_Isent;

    // Receiver side
    // The headers on which we called MPI_Irecv
    std::list<std::unique_ptr<message>> headers_Ircvd;
    // The bodies for which we called MPI_Irecv 
    std::list<std::unique_ptr<message>> bodies_Ircvd;

    /** 
     * Messages management Isend/Irecv
     */

    /**
     * Sender size
     */

    // Immediately Isend the message
    void Isend_header_body(std::unique_ptr<message> &m);

    // Send to myself, copies body, run AM and completion
    void self_Isend_header_body_process_complete(std::unique_ptr<message> &m);

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
    bool probe_Irecv_header(std::unique_ptr<message> &m);

    // Process the header
    void process_header(std::unique_ptr<message> &m);
    
    // Irecv the body
    void Irecv_body(std::unique_ptr<message> &m);

    // Process message
    void process_body(std::unique_ptr<message> &m);

    // Run the complete function when the body has been sent
    void process_completed_body(std::unique_ptr<message> &m);

    // Probe for incoming header
    // Starts MPI_Irecv for all probed headers
    void probe_Irecv_headers();

    // Process ready headers
    // If possible (taking ordering of MPI messages into account), Irecv their bodies
    void test_process_Ircvd_headers_Irecv_bodies();

    // Test completion of the bodies
    void test_process_bodies();

    /**
     * Active Message have access to those functions
     */
    template <typename... Ps>
    friend class ActiveMsg;

    /**
     * Blocking-send a message
     * Should be called from thread that called MPI_Init_Thread
     */
    void blocking_send(std::unique_ptr<message> m);

    /**
     * Queue a message in RPCComm internal message queue
     * Message will be Isent later
     * Thread-safe
     */
    void queue_message(std::unique_ptr<message> m);

    /**
     * Queue a message in the internal message queue. Name is used to annotate the message. Message will be Isent later.
     * Thread-safe
     * \param name the message name
     * \param m a message
     */
    std::unique_ptr<message> make_active_message(int dest, size_t header_size);

public:

    /**
     * \brief Creates an Communicator.
     * 
     * \param[in] comm the MPI communicator to use in communications.
     * \param[in] verb the verbose level: 0 = no printing. > 0 = more and more printing.
     * \param[in] break_msg_size the size at which to break large messages into MPI messages. Mainly used for testing.
     * 
     * \pre `verb >= 0`.
     */
    Communicator(MPI_Comm comm = MPI_COMM_WORLD, int verb_ = 0, size_t break_msg_size_ = Communicator::max_int_size);

    /**
     * \brief Creates an active message tied to function fun.
     * 
     * \param[in] fun the active function to be run on the receiver rank.
     * 
     * \return A pointer to the active message. The active message is stored in `this` and should not be freed by the user.
     */
    template <typename... Ps>
    ActiveMsg<Ps...> *make_active_msg(std::function<void(Ps &...)> fun);

    /**
     * \brief Creates an active message tied to function fun and body pointer function fun_ptr.
     * 
     * \param[in] fun the active function to be run on the receiver rank.
     * \param[in] fun_ptr the active function to be run on the receiver rank to retreive the body buffer location.
     * \param[in] fun_complete function to be run on the sender when the send operation has complete.
     * 
     * \return A pointer to the active message. The active message is stored in `this` and should not be freed by the user.
     */
    template <typename T, typename... Ps>
    ActiveMsg<Ps...> *make_large_active_msg(std::function<void(Ps &...)> fun, 
                                            std::function<T*(Ps &...)> fun_ptr,
                                            std::function<void(Ps &...)> fun_complete);

    /**
     * \brief Creates an active message tied to function fun.
     * 
     * \param[in] fun the active function to be run on the receiver rank.
     * 
     * \return A pointer to the active message. The active message is stored in `this` and should not be freed by the user.
     */
    template <typename F>
    typename ActiveMsg_type<decltype(&F::operator())>::type *make_active_msg(F fun);

    /**
     * \brief Creates an active message tied to function fun and body pointer function fun_ptr.
     * 
     * \param[in] fun the active function to be run on the receiver rank.
     * \param[in] fun_ptr the active function to be run on the receiver rank to retreive the body buffer location.
     * \param[in] fun_complete function to be run on the sender when the send operation has complete.
     * 
     * \return A pointer to the active message. The active message is stored in `this` and should not be freed by the user.
     */
    template <typename F, typename G, typename H>
    typename ActiveMsg_type<decltype(&F::operator())>::type *make_large_active_msg(F fun,
                                                                                   G fun_ptr,
                                                                                   H fun_complete);

    /**
     * \brief Set the logger.
     * 
     * \param[in] logger a pointer to the logger. The logger is not owned by `this`.
     * 
     * \pre `logger` should be a pointer to a valid `Logger`, that should not be destroyed while `this` is in use.
     */
    void set_logger(Logger *logger);

    /** 
     * \brief Blocking-receive & process a message. 
     * 
     * \details Should be called from thread that called MPI_Init_Thread.
     *          Not thread safe.
     */
    void recv_process();

    /**
     * \brief Makes progress on the communications.
     * 
     * \details Asynchronous (queue rpcs & in-flight lpcs) progress.
     *          Polls in Irecv and Isend request.
     *          Should be called from thread that called MPI_Init_Thread.
     *          Not thread safe.
     */
    void progress();

    /**
     * \brief Check for local completion.
     * 
     * \return `true` if all queues are empty, `false` otherwise.
     */
    bool is_done();

    /**
     * \brief Number of locally processed active messages.
     * 
     * \details An active message is processed on the receiver when the associated LPC has finished running.
     * 
     * \return The number of processed active message. 
     */
    int get_n_msg_processed();

    /**
     * \brief Number of locally queued active messages.
     * 
     * \details An active message is queued on the sender after a call to `am->send(...)`.
     * 
     * \return The number of queued active message.
     */
    int get_n_msg_queued();

    /**
     * \brief The rank within the communicator
     * 
     * \return The MPI rank of the current processor
     */
    int comm_rank();

    /**
     * \brief The size of the communicator
     * 
     * \return The number of MPI ranks within the communicator
     */
    int comm_size();
};

} // namespace ttor

/**
 * Implementations
 */

namespace ttor {

// Create active messages
template <typename... Ps>
ActiveMsg<Ps...> *Communicator::make_active_msg(std::function<void(Ps &...)> fun)
{
    std::function<char*(Ps &...)> fun_ptr = [](Ps &...) {
        return nullptr;
    };
    std::function<void(Ps &...)> fun_complete = [](Ps &...) {
        return;
    };
    return make_large_active_msg(fun, fun_ptr, fun_complete);
}

// Create large active messages
template <typename T, typename... Ps>
ActiveMsg<Ps...> *Communicator::make_large_active_msg(std::function<void(Ps &...)> fun, 
                                                      std::function<T*(Ps &...)> fun_ptr,
                                                      std::function<void(Ps &...)> fun_complete)
{
    auto am = std::make_unique<ActiveMsg<Ps...>>(fun, fun_ptr, fun_complete, this, active_messages.size());
    auto am_ = am.get();
    active_messages.push_back(move(am));

    if (verb > 0)
    {
        printf("[%2d]: created lpc() with ID %lu\n", comm_rank(), active_messages.size() - 1);
    }

    return am_;
}

// TODO: Simplify the return type
template <typename F>
typename ActiveMsg_type<decltype(&F::operator())>::type *Communicator::make_active_msg(F f)
{
    auto fun = GetStdFunction(f);
    return make_active_msg(fun);
}

template <typename F, typename G, typename H>
typename ActiveMsg_type<decltype(&F::operator())>::type *Communicator::make_large_active_msg(F f, G g, H h)
{
    auto fun = GetStdFunction(f);
    auto fun_ptr = GetStdFunction(g);
    auto fun_complete = GetStdFunction(h);
    return make_large_active_msg(fun, fun_ptr, fun_complete);
}

} // namespace ttor

#endif

#endif
