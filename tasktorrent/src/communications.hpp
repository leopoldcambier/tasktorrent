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

int comm_rank();
int comm_size();
std::string processor_name();

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

class Communicator
{

private:
    const static size_t mega = (1 << 20);
    const static size_t max_int_size = static_cast<size_t>(std::numeric_limits<int>::max());
    const int verb;
    Logger *logger;
    bool log;
    std::vector<std::unique_ptr<ActiveMsgBase>> active_messages;
    std::atomic<int> messages_queued;    // queued messages
    std::atomic<int> messages_processed; // received and processed messages
    MPI_Datatype MPI_MEGABYTE; // used to send large message larger than 4GB
    const size_t break_msg_size; // really mainly used for testing, where we make message artificially smaller, so we can actually test them

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

    // Probe for incoming header
    // Starts MPI_Irecv for all probed headers
    void probe_Irecv_headers();

    // Process ready headers
    // If possible (taking ordering of MPI messages into account), Irecv their bodies
    void test_process_Ircvd_headers_Irecv_bodies();

    // Test completion of the bodies
    void test_process_bodies();


public:
    /**
     * Create a message tailored for an ActiveMsg of a given size to dest
     * Message can later be filled with the data to be sent
     * TODO: This should be hidden from public
     */
    std::unique_ptr<message> make_active_message(int dest, size_t header_size);

public:

    /**
     * Creates an Communicator
     * - verb_ is the verbose level: 0 = no printing. 4 = lots of printing.
     * - break_msg_size is used to send smaller message, used mainly for testing and should not be used by the user
     */
    Communicator(int verb_ = 0, size_t break_msg_size_ = Communicator::max_int_size);

    /**
     * Creates an active message tied to function fun
     */
    template <typename... Ps>
    ActiveMsg<Ps...> *make_active_msg(std::function<void(Ps &...)> fun);

    template <typename T, typename... Ps>
    ActiveMsg<Ps...> *make_large_active_msg(std::function<void(Ps &...)> fun, std::function<T*(Ps &...)> fun_ptr);

    /**
     * Creates an active message tied to function fun
     * fun can be a lambda function
     */
    template <typename F>
    typename ActiveMsg_type<decltype(&F::operator())>::type *make_active_msg(F f);
    template <typename F, typename G>
    typename ActiveMsg_type<decltype(&F::operator())>::type *make_large_active_msg(F f, G g);


    /**
     * Set the logger
     */
    void set_logger(Logger *logger_);

    /**
     * Queue a message in RPCComm internal message queue
     * Message will be Isent later
     * Thread-safe
     */
    void queue_message(std::unique_ptr<message> m);

    /**
     * Blocking-send a message
     * Should be called from thread that called MPI_Init_Thread
     */
    void blocking_send(std::unique_ptr<message> m);

    /** 
     * Blocking-recv & process a message 
     * Should be called from thread that called MPI_Init_Thread
     */
    void recv_process();

    /**
     * Asynchronous (queue_rpc & in-flight lpcs) Progress
     * Polls in Irecv and Isend request
     * Should be called from thread that called MPI_Init_Thread
     */
    void progress();

    /**
     * Returns true is all queues are empty
     * Returns false otherwise
     */
    bool is_done();

    /**
     * Returns the number of received and processed messages
     */
    int get_n_msg_processed();

    /**
     * Returns the number of queued (or sent) messages
     */
    int get_n_msg_queued();
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
    return make_large_active_msg(fun, fun_ptr);
}

// Create large active messages
template <typename T, typename... Ps>
ActiveMsg<Ps...> *Communicator::make_large_active_msg(std::function<void(Ps &...)> fun, std::function<T*(Ps &...)> fun_ptr)
{
    auto am = std::make_unique<ActiveMsg<Ps...>>(fun, fun_ptr, this, active_messages.size());
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

template <typename F, typename G>
typename ActiveMsg_type<decltype(&F::operator())>::type *Communicator::make_large_active_msg(F f, G g)
{
    auto fun = GetStdFunction(f);
    auto fun_ptr = GetStdFunction(g);
    return make_large_active_msg(fun, fun_ptr);
}

} // namespace ttor

#endif

#endif