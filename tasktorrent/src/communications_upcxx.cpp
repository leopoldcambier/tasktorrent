#ifdef TTOR_UPCXX

#include "communications_upcxx.hpp"

namespace ttor
{

/**
 * Communicator_UPCXX
 */

Communicator_UPCXX::Communicator_UPCXX(int verb_) : Communicator_Base(verb_), dcomm(this), messages_queued(0), messages_processed(0) {}

void Communicator_UPCXX::progress() {
    upcxx::progress();
}

bool Communicator_UPCXX::is_done() const {
    return true;
}

int Communicator_UPCXX::comm_rank() const {
    return upcxx::rank_me();
}

int Communicator_UPCXX::comm_size() const {
    return upcxx::rank_n();
}

llint Communicator_UPCXX::get_n_msg_processed() const {
    return messages_processed.load();
}

llint Communicator_UPCXX::get_n_msg_queued() const {
    return messages_queued.load();
}

Communicator_UPCXX::~Communicator_UPCXX() = default;

/**
 * ActiveMsg_UPCXX_Base
 */

Communicator_UPCXX::ActiveMsg_Base::~ActiveMsg_Base() = default;

}

#endif