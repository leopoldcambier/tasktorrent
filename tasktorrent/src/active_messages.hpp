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
 * \details An active message is two (or four) things:
 *          - A payload (header) to be send from the sender to the receiver rank.
 *          - [Optional] A payload (body) to be send from the sender to the receiver rank, without any temporary copy.
 *          - [Optional] When using a body, a function to be run on the receiver rank indicating where to store the body
 *          - A function to be run on the receiver rank, when the header and the (optional) body have arrived
 * 
 *          The function is serialized accross ranks using its ID.
 *          The payload (header) is sent as a buffer of bytes, using an intermediary copy where the payload is serialized.
 *          The payload (body) is directly send (without any intermadiary copy)
 */
class ActiveMsgBase
{

private:

    size_t id_;

public:

    /**
     * \brief Return the ID of the active message.
     * 
     * \return The global ID of the active message
     */
    size_t get_id() const;

    /**
     * \brief Deserialize the (header) payload and run the associated function.
     * 
     * \param[in] payload a pointer to the payload 
     * \param[in] size the number of bytes in the payload
     * 
     * \pre `payload` should be a valid buffer of `size` bytes.
     */
    virtual void run(char *payload, size_t size) = 0;

    /**
     * \brief Returns the location of where the body should be stored
     * 
     * \param[in] payload a pointer to the payload (header)
     * \param[in] size the number of bytes in the payload (header)
     * 
     * \pre `payload` should be a valid buffer of `size` bytes.
     */
    virtual char* get_user_buffers(char *payload, size_t size) = 0;

    /**
     * \brief Creates an active message
     * 
     * \param id the global id of that active message. 
     * 
     * \pre `id` should be a unique id for that active message, and should be the same for that active message accross all ranks.
     */
    ActiveMsgBase(size_t id);

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
 *          - A payload (header)
 *          tied to an Communicator instance.
 *          The active message also had an optional payload (body) and a function to indicate where to store the body on the receiver.
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
     * \brief Creates an active message.
     * 
     * \param[in] fun the function to be run on the receiver.
     * \param[in] ptr_fun the function to be run on the receiver, giving the location of where the body should be stored.
     * \param[in] comm the communicator instance to use for communications.
     *            The active message does not take ownership of `comm`.
     * \param[in] id the active message unique ID. User is responsible to never 
     *           reuse ID's, and all ranks should use the same ID's to refer
     *           to the same active function
     * 
     * \pre `comm` should be a valid pointer to a `Communicator`, which should not be destroyed while the
     *      active message is in used.
     */
    template<typename T>
    ActiveMsg(std::function<void(Ps &...)> fun, std::function<T*(Ps &...)> ptr_fun, Communicator *comm, size_t id) : ActiveMsgBase(id), comm_(comm), fun_(fun) {
        ptr_fun_ = [ptr_fun](Ps &... ps) {
            char* ptr = (char*)ptr_fun(ps...);
            return ptr;
        };
    }

    virtual void run(char *payload_raw, size_t size);
    
    virtual char* get_user_buffers(char *payload_raw, size_t size);

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
     * \brief Queue the payload to be send later.
     * 
     * \details This is thread-safe and can be called by any thread.
     * 
     * \param[in] dest the destination rank
     * \param[in] ps the payload
     */
    void send(int dest, Ps &... ps);

    /**
     * \brief Queue the payload to be send later, with an accompanying body
     * 
     * \details This is thread-safe and can be called by any thread.
     * 
     * \param[in] dest the destination rank
     * \param[in] body a view to the body
     * \param[in] ps the payload
     */
    template<typename T>
    void send_large(int dest, view<T> body, Ps &... ps);

    /**
     * \brief Immediately send the payload, with an accompanying body
     * 
     * \details The function returns when the payload (the header, not the body) has been sent.
     *          This is not thread safe and can only be called by the MPI master thread.
     * 
     * \param[in] dest the destination rank
     * \param[in] body a view to the body
     * \param[in] ps the payload
     */
    template<typename T>
    void blocking_send_large(int dest, view<T> body, Ps &... ps);

    /**
     * \brief Destroys the ActiveMsg
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