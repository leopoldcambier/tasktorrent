#ifndef __TTOR_SRC_ACTIVEMESSAGE_HPP__
#define __TTOR_SRC_ACTIVEMESSAGE_HPP__

#ifndef TTOR_SHARED

#include <utility>
#include <mutex>
#include <memory>
#include <cassert>
#include <functional>

#include "apply_functions.hpp"
#include "serialization.hpp"
#include "message.hpp"

namespace ttor {

class Communicator;

/**
 * Base Active Message class
 */
class ActiveMsgBase
{
private:
    int id_;
public:
    int get_id() const;
    virtual void run(char *, size_t) = 0;
    ActiveMsgBase(int id);
    virtual ~ActiveMsgBase();
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
    std::unique_ptr<message> make_message(int dest, Ps &... ps);

public:
    /**
     * Create an active message tied to given function fun and that feeds into Communicator comm
     */
    ActiveMsg(std::function<void(Ps &...)> fun, Communicator *comm, int id);
    /**
     * Deserialize payload_raw and run active message through the RPCComm
     */
    virtual void run(char *payload_raw, size_t size);
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
    void named_send(int dest, std::string name, Ps &... ps);
    virtual ~ActiveMsg();
};

} // namespace ttor

/**
 * Implementations
 */

#include "communications.hpp"

namespace ttor {

template <typename... Ps>
ActiveMsg<Ps...>::ActiveMsg(std::function<void(Ps &...)> fun, Communicator *comm, int id) : ActiveMsgBase(id), comm_(comm), fun_(fun) {}

template <typename... Ps>
void ActiveMsg<Ps...>::run(char *payload_raw, size_t size)
{
    Serializer<int, Ps...> s;
    auto tup = s.read_buffer(payload_raw, size);
    assert(get_id() == std::get<0>(tup));
    auto args = tail(tup);
    apply_fun(fun_, args);
}

template <typename... Ps>
std::unique_ptr<message> ActiveMsg<Ps...>::make_message(int dest, Ps &... ps)
{
    Serializer<int, Ps...> s;
    int id = get_id();
    size_t size = s.size(id, ps...);
    std::unique_ptr<message> m = comm_->make_active_message(dest, size);
    s.write_buffer(m->buffer.data(), m->buffer.size(), id, ps...);
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
void ActiveMsg<Ps...>::named_send(int dest, std::string name, Ps &... ps)
{
    auto m = make_message(dest, ps...);
    comm_->queue_named_message(name, move(m));
}

template <typename... Ps>
ActiveMsg<Ps...>::~ActiveMsg(){}

} // namespace ttor

#endif

#endif