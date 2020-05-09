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
    size_t id_;
public:
    size_t get_id() const;
    virtual void run(char *, size_t) = 0;
    virtual char* get_user_buffers(char*, size_t) = 0;
    ActiveMsgBase(size_t id);
    virtual ~ActiveMsgBase();
};

/**
 * Implementation of Active Message
 * An active message is a pair of
 * (1) A local function
 * (2) A remote payload, made of a header (type Ps...) and a body (a buffer of T). The body can be empty
 * tied to an Communicator instance
 */
template <typename... Ps>
class ActiveMsg : public ActiveMsgBase
{
private:
    Communicator *comm_;
    std::function<void(Ps &...)> fun_;
    std::function<char*(Ps &...)> ptr_fun_;

    /**
     * Create the message
     */
    template<typename T>
    std::unique_ptr<message> make_message(int dest, view<T> body, Ps &... ps);

public:
    /**
     * Create an active message tied to given function fun and that feeds into Communicator comm
     */
    template<typename T>
    ActiveMsg(std::function<void(Ps &...)> fun, std::function<T*(Ps &...)> ptr_fun, Communicator *comm, size_t id) : ActiveMsgBase(id), comm_(comm), fun_(fun) {
        ptr_fun_ = [ptr_fun](Ps &... ps) {
            char* ptr = (char*)ptr_fun(ps...);
            return ptr;
        };
    }

    /**
     * Deserialize payload_raw and run active message through the RPCComm
     */
    virtual void run(char *payload_raw, size_t size);
    /**
     * Get user buffers
     */
    virtual char* get_user_buffers(char *payload_raw, size_t size);
    /**
     * Immediately send the payload to be sent to the destination
     * Should be called from the same thread calling MPI_Init_Thread(...)
     */
    void blocking_send(int dest, Ps &... ps);
    /**
     * Queue the payload to be send later to dest
     * Thread-safe
     */
    void send(int dest, Ps &... ps);
    /**
     * Queue the payload to be send later, with a body
     * Thread-safe
     */
    template<typename T>
    void send_large(int dest, view<T> body, Ps &... ps);

    template<typename T>
    void blocking_send_large(int dest, view<T> body, Ps &... ps);

    virtual ~ActiveMsg();
};

} // namespace ttor

/**
 * Implementations
 */

#include "communications.hpp"

namespace ttor {

template <typename... Ps>
void ActiveMsg<Ps...>::run(char *payload_raw, size_t size)
{
    // ID, body size, header args...
    Serializer<size_t, size_t, Ps...> s;
    auto tup = s.read_buffer(payload_raw, size);
    assert(std::get<0>(tup) == get_id());
    assert(std::get<1>(tup) >= 0);
    auto args = tail(tail(tup));
    apply_fun(fun_, args);
}

template <typename... Ps>
char* ActiveMsg<Ps...>::get_user_buffers(char *payload_raw, size_t size)
{
    // ID, body size, header args...
    Serializer<size_t, size_t, Ps...> s;
    auto tup = s.read_buffer(payload_raw, size);
    assert(std::get<0>(tup) == get_id());
    assert(std::get<1>(tup) >= 0);
    auto args = tail(tail(tup));
    return apply_fun(ptr_fun_, args);
}

template <typename... Ps>
template <typename T>
std::unique_ptr<message> ActiveMsg<Ps...>::make_message(int dest, view<T> body, Ps &... ps)
{
    // ID, body size, header args...
    Serializer<size_t, size_t, Ps...> s;
    size_t id = get_id();
    size_t body_size = body.size() * sizeof(T);
    size_t header_size = s.size(id, body_size, ps...);
    std::unique_ptr<message> m = comm_->make_active_message(dest, header_size);
    s.write_buffer(m->header_buffer.data(), m->header_buffer.size(), id, body_size, ps...);
    m->body_buffer = (char*)body.data();
    m->body_size = body_size;
    return m;
}

template <typename... Ps>
void ActiveMsg<Ps...>::blocking_send(int dest, Ps &... ps)
{
    view<char> body(nullptr, 0);
    blocking_send_large(dest, body, ps...);
}

template <typename... Ps>
void ActiveMsg<Ps...>::send(int dest, Ps &... ps)
{
    view<char> body(nullptr, 0);
    send_large(dest, body, ps...);
}

template <typename... Ps>
template <typename T>
void ActiveMsg<Ps...>::send_large(int dest, view<T> body, Ps &... ps)
{
    auto m = make_message(dest, body, ps...);
    comm_->queue_message(move(m));
}

template <typename... Ps>
template <typename T>
void ActiveMsg<Ps...>::blocking_send_large(int dest, view<T> body, Ps &... ps)
{
    auto m = make_message(dest, body, ps...);
    comm_->blocking_send(move(m));
}

template <typename... Ps>
ActiveMsg<Ps...>::~ActiveMsg(){}

} // namespace ttor

#endif

#endif