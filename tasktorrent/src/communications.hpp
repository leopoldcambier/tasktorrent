#ifndef __TTOR_SRC_COMMUNICATIONS_HPP__
#define __TTOR_SRC_COMMUNICATIONS_HPP__

#ifndef TTOR_SHARED

#include <utility>
#include <mutex>
#include <memory>
#include <functional>
#include <list>

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
    std::atomic<int> messages_queued; // queued messages
    std::atomic<int> messages_processed; // received and processed messages
    MPI_Datatype MPI_MEGABYTE; // used to send large message larger than 4GB

    /** Small messages                        
     *  This class maintains three lists to handle "small" messages, for which we allocate memory internally.
     *  messages_rdy is a list containing all messages "ready" to be send (memory allocated and buffer ready).
     *      Any thread can add to this list. messages_rdy_mtx protects access.
     *  messages_Isent is a list containing all messages sent (Isent called, message pending). This is only manipulated by the master thread
     *  messages_Ircvd is a list containing all messages recv (Irecv called, message pending). This is only manipulated by the master thread
     */
    std::list<std::unique_ptr<message>> messages_rdy;
    std::mutex messages_rdy_mtx;
    std::list<std::unique_ptr<message>> messages_Isent;
    std::list<std::unique_ptr<message>> messages_Ircvd;

    /** 
     * Messages management Isend/Irecv
     */
    void Isend_message(const std::unique_ptr<message> &m);
    // Loop through the list of messages to send in the task flow.
    // Isend all the messages in the ready queue.
    void Isend_queued_messages();
    // Test to see whether the Isent has completed or not
    void test_Isent_messages();
    // Probe for message received
    // If probe is true, then Irecv the message
    bool probe_Irecv_message(std::unique_ptr<message> &m);
    // Run all lpcs that have been received
    void process_Ircvd_messages();

    /** 
     * Process message (only Active message management so far)
     */
    void process_message(const std::unique_ptr<message> &m);

public:
    /**
     * Create a message tailored for an ActiveMsg of a given size to dest
     * Message can later be filled with the data to be sent
     * TODO: This should be hidden from public
     */
    std::unique_ptr<message> make_active_message(int dest, size_t size);

public:

    /**
     * Creates an Communicator
     * - verb_ is the verbose level: 0 = no printing. 4 = lots of printing.
     */
    Communicator(int verb_ = 0);

    /**
     * Creates an active message tied to function fun
     */
    template <typename... Ps>
    ActiveMsg<Ps...> *make_active_msg(std::function<void(Ps &...)> fun);

    /**
     * Creates an active message tied to function fun
     * fun can be a lambda function
     */
    template <typename F>
    typename ActiveMsg_type<decltype(&F::operator())>::type *make_active_msg(F f);

    /**
     * Set the logger
     */
    void set_logger(Logger *logger_);

    /**
     * Queue a message in RPCComm internal message queue
     * Name is used to annotate the message
     * Message will be Isent later
     * Thread-safe
     */
    template <typename... Ps>
    void queue_named_message(std::string name, std::unique_ptr<message> m);

    /**
     * Queue a message in RPCComm internal message queue
     * Message will be Isent later
     * Thread-safe
     */
    template <typename... Ps>
    void queue_message(std::unique_ptr<message> m);

    /**
     * Blocking-send a message
     * Should be called from thread that called MPI_Init_Thread
     */
    template <typename... Ps>
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

template <typename... Ps>
void Communicator::queue_named_message(std::string name, std::unique_ptr<message> m)
{
    // Increment message counter
    messages_queued++;

    std::unique_ptr<Event> e;
    if (log)
    {
        e = std::make_unique<Event>();
        e->name = "rank_" + std::to_string(comm_rank()) + ">qrpc>" + "rank_" + std::to_string(m->other) + ">" + std::to_string(m->tag) + ">" + name;
    }

    {
        std::lock_guard<std::mutex> lock(messages_rdy_mtx);
        if (verb > 2)
        {
            printf("[%2d] -> %d: %d pushed, %lu total\n", comm_rank(), m->other, m->tag, messages_rdy.size() + 1);
        }
        messages_rdy.push_back(std::move(m));
    }

    if (log)
        logger->record(std::move(e));
}

template <typename... Ps>
void Communicator::queue_message(std::unique_ptr<message> m)
{
    queue_named_message("_", move(m));
}

// Blocking send
template <typename... Ps>
void Communicator::blocking_send(std::unique_ptr<message> m)
{
    // Increment message counter
    messages_queued++;

    Isend_message(m);
    TASKTORRENT_MPI_CHECK(MPI_Wait(&m->request, MPI_STATUS_IGNORE));
}

// Create active messages
template <typename... Ps>
ActiveMsg<Ps...> *Communicator::make_active_msg(std::function<void(Ps &...)> fun)
{
    auto am = std::make_unique<ActiveMsg<Ps...>>(fun, this, active_messages.size());
    ActiveMsg<Ps...> *am_ = am.get();
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

} // namespace ttor

#endif

#endif
