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
 * \brief Base Active Message class
 * 
 * \details An active message is two things:
 *          - A function 
 *          - A payload
 *          The function is serialized accross ranks using its ID
 *          The payload is send as a buffer of bytes
 */
class ActiveMsgBase
{

private:

    int id_;

public:

    /**
     * \brief Return the ID of the active message.
     * 
     * \return The global ID of the active message
     */
    int get_id() const;

    /**
     * \brief Deserialize the payload and run the associated function.
     * 
     * \param[in] payload a pointer to the payload 
     * \param[in] size the number of bytes in the payload
     */
    virtual void run(char * payload, size_t size) = 0;

    /**
     * \brief Creates an active message
     * 
     * \param id the global id of that active message. ID should be unique for every active message (on a given rank).
     */
    ActiveMsgBase(int id);

    /**
     * \brief Destroys the active message.
     */
    virtual ~ActiveMsgBase();
};

/**
 * \brief Implementation of Active Message for a payload of type `Ps...`.
 * 
 * \details An active message is a pair of
 *          - A function
 *          - A payload
 *          tied to an Communicator instance
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
     * \brief Create an active message
     * 
     * \param[in] fun the function to be run on the receiver
     * \param[in] comm the communicator instance to use for communications
     * \param[in] id the active message unique ID. User is responsible to never 
     *           reuse ID's, and all ranks should use the same ID's to refer
     *           to the same active function
     */
    ActiveMsg(std::function<void(Ps &...)> fun, Communicator *comm, int id);
    
    virtual void run(char *payload_raw, size_t size);
    
    /**
     * \brief Immediately sends payload to destination.
     * 
     * \details The function returns when the payload has been sent.
     *          This is not thread safe and can only be called by the MPI master thread.
     * 
     * \param[in] dest the destination rank
     * \param[in] ps the payload
     */
    void blocking_send(int dest, Ps &... ps);
    
    /**
     * \brief Queue the payload to be send later
     * 
     * \details This is thread-safe and can be called by any thread
     * 
     * \param[in] dest the destination rank
     * \param[in] ps the payload
     */
    void send(int dest, Ps &... ps);

    /**
     * \brief Queue the payload to be send later
     * 
     * \details This is thread-safe and can be called by any thread
     * 
     * \param[in] dest the destination rank
     * \param[in] name the name to associate to this send operation (for logging purposes)
     * \param[in] ps the payload
     */
    void named_send(int dest, std::string name, Ps &... ps);

    /**
     * \brief Destroys the active message
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