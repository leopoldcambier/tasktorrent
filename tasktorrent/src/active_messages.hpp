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
 * An active message is two things:
 * - A function 
 * - A payload
 * The function is serialized accross ranks using its ID
 * The payload is send as a buffer of bytes
 */
class ActiveMsgBase
{
private:
    int id_;
public:
    /**
     * Return the ID of the active message.
     * \return The global ID of the active message
     */
    int get_id() const;

    /**
     * Deserialize the payload and run the associated function.
     * \param payload a pointer to the payload 
     * \param size the number of bytes in the payload
     */
    virtual void run(char * payload, size_t size) = 0;

    /**
     * Creates an active message with corresponding global ID `id`.
     * \param id the global id of that active message. ID should be unique for every active message (on a given rank).
     */
    ActiveMsgBase(int id);

    /**
     * Destroys the active message.
     */
    virtual ~ActiveMsgBase();
};

/**
 * Implementation of Active Message for a payload of type `Ps...`.
 * An active message is a pair of
 * - A function
 * - A payload
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
     * Create an active message with ID `id` tied to function `fun` using communicator `comm` for communications
     * \param fun the function to be run on the receiver
     * \param comm the communicator instance to use for communications
     * \param id the active message unique ID. User is responsible to never reuse ID's.
     */
    ActiveMsg(std::function<void(Ps &...)> fun, Communicator *comm, int id);
    
    virtual void run(char *payload_raw, size_t size);
    
    /**
     * Immediately sends payload to destination
     * The function returns when the payload has been sent
     * This is not thread safe and can only be called by the MPI master thread
     * \param dest the destination rank
     * \param ps the payload
     */
    void blocking_send(int dest, Ps &... ps);
    
    /**
     * Queue the payload `ps` to be send later to `dest`
     * This is thread-safe and can be called by any thread
     * \param dest the destination rank
     * \param ps is the payload
     */
    void send(int dest, Ps &... ps);

    /**
     * Queue the payload `ps` to be send later to `dest` and associated name `name` for profiling purposes
     * This is thread-safe and can be called by any thread
     * \param dest the destination rank
     * \param name is the name to associate to this send operation
     * \param ps is the payload
     */
    void named_send(int dest, std::string name, Ps &... ps);

    /**
     * Destroys the active message
     */
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