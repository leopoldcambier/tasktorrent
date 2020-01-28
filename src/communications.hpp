#ifndef __TTOR_COMMUNICATIONS_HPP__
#define __TTOR_COMMUNICATIONS_HPP__

#include <iostream>
#include <tuple>
#include <utility>
#include <thread>
#include <unordered_map>
#include <array>
#include <deque>
#include <mutex>
#include <functional>
#include <utility>
#include <string>
#include <sstream>
#include <atomic>
#include <queue>
#include <memory>
#include <list>
#include <algorithm>

#include <mpi.h>

#include "serialization.hpp"
#include "util.hpp"
#include "views.hpp"
#include "apply_functions.hpp"
#include "functional_extra.hpp"

namespace ttor
{

using std::list;
using std::make_unique;
using std::move;
using std::queue;
using std::to_string;
using std::unique_ptr;
using std::vector;

int comm_rank();
int comm_size();
string processor_name();

struct message
{
public:
    std::vector<char> buffer;
    MPI_Request request;
    int other;
    int tag;
    char *start_buffer;
    message(int other) : other(other){};
};

class Communicator;

/**
 * Base Active Message class
 */
class ActiveMsgBase
{
public:
    virtual void run(char *) = 0;
    virtual ~ActiveMsgBase(){};
};

/**
 * Implementation of Active Message
 * An active message is a pair of
 * (1) A local function
 * (2) A remote payload
 * tied to an Communicator instance
 */
template <typename... Ps>
class ActiveMsg : public ActiveMsgBase
{
private:
    Communicator *comm_;
    std::function<void(Ps &...)> fun_;
    unique_ptr<message> make_message(int dest, Ps &... ps);

public:
    /**
     * Create an active message tied to given function fun and that feeds into Communicator comm
     */
    ActiveMsg(std::function<void(Ps &...)> fun, Communicator *comm);
    /**
     * Deserialize payload_raw and run active message through the RPCComm
     */
    virtual void run(char *payload_raw);
    /**
     * Immediately send the payload to be sent to the destination
     * Should be called from the same thread calling MPI_Init_Thread(...)
     */
    void blocking_send(int dest, Ps &... ps);
    /**
     * Queue the payload to be send later to dest
     * Thread-safe; can be called from any thread
     */
    void send(int dest, Ps &... ps);
    /**
     * Queue the payload to be send later to dest
     * Annotate the message with name for logging purposes
     * Thread-safe; can be called from any thread
     */
    void named_send(int dest, string name, Ps &... ps);
    virtual ~ActiveMsg();
};

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
    const int verb;
    Logger *logger;
    bool log;
    const int tag;
    std::vector<unique_ptr<ActiveMsgBase>> active_messages;
    std::atomic<int> messages_queued; // queued messages
    std::atomic<int> messages_processed; // received and processed messages

    /** Small messages                        
     *  This class maintains three lists to handle "small" messages, for which we allocate memory internally.
     *  messages_rdy is a list containing all messages "ready" to be send (memory allocated and buffer ready).
     *      Any thread can add to this list. messages_rdy_mtx protects access.
     *  messages_Isent is a list containing all messages sent (Isent called, message pending). This is only manipulated by the master thread
     *  messages_Ircvd is a list containing all messages recv (Irecv called, message pending). This is only manipulated by the master thread
     */
    list<unique_ptr<message>> messages_rdy;
    std::mutex messages_rdy_mtx;
    list<unique_ptr<message>> messages_Isent;
    list<unique_ptr<message>> messages_Ircvd;

    /** 
     * Messages management Isend/Irecv
     */
    void Isend_message(const unique_ptr<message> &m);
    // Loop through the list of messages to send in the task flow.
    // Isend all the messages in the ready queue.
    void Isend_queued_messages();
    // Test to see whether the Isent has completed or not
    void test_Isent_messages();
    // Probe for message received
    // If probe is true, then Irecv the message
    bool probe_Irecv_message(unique_ptr<message> &m);
    // Run all lpcs that have been received
    void process_Ircvd_messages();

    /** 
     * Process message (only Active message management so far)
     */
    void process_message(const unique_ptr<message> &m);

public:
    /**
     * Create a message tailored for an ActiveMsg of a given size to dest
     * Message can later be filled with the data to be sent
     * TODO: This should be hidden from public
     */
    unique_ptr<message> make_active_message(ActiveMsgBase *am, int dest, int size);

public:

    /**
     * Creates an Communicator
     * - verb_ is the verbose level: 0 = no printing. 4 = lots of printing.
     */
    Communicator(int verb_ = 0, int tag_ = 0);

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
    void queue_named_message(string name, unique_ptr<message> m);

    /**
     * Queue a message in RPCComm internal message queue
     * Message will be Isent later
     * Thread-safe
     */
    template <typename... Ps>
    void queue_message(unique_ptr<message> m);

    /**
     * Blocking-send a message
     * Should be called from thread that called MPI_Init_Thread
     */
    template <typename... Ps>
    void blocking_send(unique_ptr<message> m);

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

/**
 * Implementations
 */

/**
 * Active Messages
 */
template <typename... Ps>
ActiveMsg<Ps...>::ActiveMsg(function<void(Ps &...)> fun, Communicator *comm) : comm_(comm), fun_(fun) {}

template <typename... Ps>
void ActiveMsg<Ps...>::run(char *payload_raw)
{
    Serializer<Ps...> s;
    tuple<Ps...> tup = s.read_buffer(payload_raw);
    apply_fun(fun_, tup);
}

template <typename... Ps>
unique_ptr<message> ActiveMsg<Ps...>::make_message(int dest, Ps &... ps)
{
    Serializer<Ps...> s;
    int size = s.size(ps...);
    unique_ptr<message> m = comm_->make_active_message(this, dest, size);
    s.write_buffer(m->start_buffer, ps...);
    return m;
}

template <typename... Ps>
void ActiveMsg<Ps...>::blocking_send(int dest, Ps &... ps)
{
    auto m = make_message(dest, ps...);
    comm_->blocking_send(move(m));
}

template <typename... Ps>
void ActiveMsg<Ps...>::send(int dest, Ps &... ps)
{
    named_send(dest, "_", ps...);
}

template <typename... Ps>
void ActiveMsg<Ps...>::named_send(int dest, string name, Ps &... ps)
{
    auto m = make_message(dest, ps...);
    comm_->queue_named_message(name, move(m));
}

template <typename... Ps>
ActiveMsg<Ps...>::~ActiveMsg(){};

template <typename... Ps>
ActiveMsg<Ps...> *Communicator::make_active_msg(function<void(Ps &...)> fun)
{
    auto am = make_unique<ActiveMsg<Ps...>>(fun, this);
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

template <typename... Ps>
void Communicator::queue_named_message(string name, unique_ptr<message> m)
{
    // Increment message counter
    messages_queued++;

    unique_ptr<Event> e;
    if (log)
    {
        e = make_unique<Event>();
        e->name = "rank_" + to_string(comm_rank()) + ">qrpc>" + "rank_" + to_string(m->other) + ">" + to_string(m->tag) + ">" + name;
    }

    {
        lock_guard<mutex> lock(messages_rdy_mtx);
        if (verb > 2)
        {
            printf("[%2d] -> %d: %d pushed, %lu total\n", comm_rank(), m->other, m->tag, messages_rdy.size() + 1);
        }
        messages_rdy.push_back(move(m));
    }

    if (log)
        logger->record(move(e));
}

template <typename... Ps>
void Communicator::queue_message(unique_ptr<message> m)
{
    queue_named_message("_", move(m));
}

// Blocking send
template <typename... Ps>
void Communicator::blocking_send(unique_ptr<message> m)
{
    // Increment message counter
    messages_queued++;

    Isend_message(m);
    int err = MPI_Wait(&m->request, MPI_STATUS_IGNORE);
    assert(err == MPI_SUCCESS);
}

} // namespace ttor

#endif
