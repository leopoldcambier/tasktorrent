#ifndef __TTOR_SRC_COMMUNICATIONS_UPCXX_HPP__
#define __TTOR_SRC_COMMUNICATIONS_UPCXX_HPP__

#ifdef TTOR_UPCXX

#include <utility>
#include <mutex>
#include <memory>
#include <functional>
#include <iostream>

#include <upcxx/upcxx.hpp>

#include "communications.hpp"
#include "views.hpp"
#include "util_templates.hpp"
#include "apply_functions.hpp"

namespace ttor
{

class Communicator_UPCXX;

class Communicator_UPCXX : public Communicator_Base
{

public:

    class ActiveMsg_Base;

    template <typename T, typename... Ps>
    class ActiveMsg;

private:

    upcxx::dist_object<Communicator_UPCXX*> dcomm;
    std::vector<std::unique_ptr<ActiveMsg_Base>> active_messages;

    std::atomic<llint> messages_queued;
    std::atomic<llint> messages_processed;

public:

    Communicator_UPCXX(int verb_ = 0);

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

    virtual llint get_n_msg_processed() const override;

    virtual llint get_n_msg_queued() const override;

    virtual void progress() override;

    virtual bool is_done() const override;

    /**
     * \brief The rank within the world team
     * 
     * \return The UPC++ rank of the current processor within the world team
     */
    virtual int comm_rank() const override;

    /**
     * \brief The size of the world team
     * 
     * \return The size of the UPC++ world team
     */
    virtual int comm_size() const override;

    virtual ~Communicator_UPCXX();

    class ActiveMsg_Base {
    public:
        virtual ~ActiveMsg_Base();
    };

    template<typename T, typename... Ps>
    class ActiveMsg : public ActiveMsg_Base
    {

    private:

        Communicator_UPCXX *comm_;

        upcxx::dist_object<std::function<void(Ps...)>> dfun;
        upcxx::dist_object<std::function<T*(Ps...)>> dfun_ptr;
        std::function<void(Ps...)> lfun_complete;

    public:
        ActiveMsg(Communicator_UPCXX* comm, 
            std::function<void(Ps...)> fun, 
            std::function<T*(Ps...)> fun_ptr, 
            std::function<void(Ps...)> fun_complete);
        void blocking_send(int dest, Ps... ps) const;
        void send(int dest, Ps... ps) const;
        void send_large(int dest, const view<T>& body, Ps... ps) const;
        void blocking_send_large(int dest, const view<T>& body, Ps... ps) const;
        virtual ~ActiveMsg();

    };

}; // Communicator_UPCXX

using Communicator = Communicator_UPCXX;

} // namespace ttor

/**
 * Communicator_UPCXX
 */

namespace ttor {

template<typename F>
details::AM_t<Communicator_UPCXX::ActiveMsg,decltype(&F::operator())>
  *Communicator_UPCXX::make_active_msg(F f) {
    auto fun = details::GetStdFunction(f);
    return make_active_msg(fun);
}

template<typename F, typename G, typename H>
details::Large_AM_t<Communicator_UPCXX::ActiveMsg,decltype(&G::operator())>
  *Communicator_UPCXX::make_large_active_msg(F f, G g, H h) {
    auto fun = details::GetStdFunction(f);
    auto fun_ptr = details::GetStdFunction(g);
    auto fun_complete = details::GetStdFunction(h);
    return make_large_active_msg(fun, fun_ptr, fun_complete);
}

template <typename... Ps>
Communicator_UPCXX::ActiveMsg<char, Ps...> *Communicator_UPCXX::make_active_msg(std::function<void(Ps...)> fun) {
    std::function<char*(Ps...)> fun_ptr = [](Ps...) {
        return nullptr;
    };
    std::function<void(Ps...)> fun_complete = [](Ps...) {
        return;
    };
    return make_large_active_msg(fun, fun_ptr, fun_complete);
}

template <typename T, typename... Ps>
Communicator_UPCXX::ActiveMsg<T,Ps...> *Communicator_UPCXX::make_large_active_msg(std::function<void(Ps...)> fun, 
                                                                                  std::function<T*(Ps...)> fun_ptr,
                                                                                  std::function<void(Ps...)> fun_complete)
{
    auto am = std::make_unique<ActiveMsg<T,Ps...>>(this, fun, fun_ptr, fun_complete);
    auto am_ = am.get();
    active_messages.push_back(move(am));
    return am_;
}

/**
 * ActiveMsg
 */

template<typename T, typename... Ps>
Communicator_UPCXX::ActiveMsg<T,Ps...>::ActiveMsg(Communicator_UPCXX* comm, 
                                                  std::function<void(Ps...)> fun, 
                                                  std::function<T*(Ps...)> fun_ptr, 
                                                  std::function<void(Ps...)> fun_complete) : 
    comm_(comm), dfun(fun), dfun_ptr(fun_ptr), lfun_complete(fun_complete) {}

template<typename T, typename... Ps>
void Communicator_UPCXX::ActiveMsg<T,Ps...>::blocking_send(int dest, Ps... ps) const {
    view<T> body;
    this->blocking_send_large(dest, body, ps...);
}
    
template<typename T, typename... Ps>
void Communicator_UPCXX::ActiveMsg<T,Ps...>::send(int dest, Ps... ps) const {
    view<T> body;
    this->blocking_send_large(dest, body, ps...);
}
    
template<typename T, typename... Ps>
void Communicator_UPCXX::ActiveMsg<T,Ps...>::send_large(int dest, const view<T>& body, Ps... ps) const {
    this->blocking_send_large(dest, body, ps...);
}

namespace details {

    // This is used to handle the fact that UPCXX has different types after deserialization
    // This converts a deserialized view<T> (which in upcxx is a upcxx::view<T,T*>) back 
    // into a view<T> (which, for us, is a ttor::details::view<T> = upcxx::view<T,const T*>)
    
    template<typename T>
    auto upcxx_convert_views(T&& t) {
        return t;
    };

    template<typename T>
    auto upcxx_convert_views(const upcxx::view<T,T*>& v) {
        return upcxx::make_view(v.cbegin(), v.cend());
    };
}

template<typename T, typename... Ps>
void Communicator_UPCXX::ActiveMsg<T,Ps...>::blocking_send_large(int dest, const view<T>& body_view, Ps... payload) const {
    // Count queued messages
    comm_->messages_queued++;
    // Send the RPC to the destination
    upcxx::rpc_ff(dest, [](
        upcxx::dist_object<std::function<void(Ps...)>> &lfun, 
        upcxx::dist_object<std::function<T*(Ps...)>> &lfun_ptr, 
        upcxx::dist_object<Communicator_UPCXX*> &lcomm, 
        const upcxx::deserialized_type_t<view<T>>&               local_body_view, // Whenever https://bitbucket.org/berkeleylab/upcxx/issues/431 is fixed
        const upcxx::deserialized_type_t<std::decay_t<Ps>>&...   local_payload)   // those two can become const auto&
    {
        // (1) Find pointer
        T* ptr = (*lfun_ptr)(details::upcxx_convert_views(local_payload)...);
        // (2) Copy body data
        std::copy(local_body_view.begin(), local_body_view.end(), ptr);
        // (3) Run usual function
        (*lfun)(details::upcxx_convert_views(local_payload)...);
        // (4) Count processed messages
        (*lcomm)->messages_processed++;
    }, this->dfun, this->dfun_ptr, comm_->dcomm, body_view, payload...);
    this->lfun_complete(payload...);
}

template<typename T, typename... Ps>
Communicator_UPCXX::ActiveMsg<T,Ps...>::~ActiveMsg() = default;

}


#endif

#endif
