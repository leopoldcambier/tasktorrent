#ifndef TTOR_SHARED

#include "active_messages.hpp"

namespace ttor {

size_t ActiveMsgBase::get_id() const { return id_; }
ActiveMsgBase::ActiveMsgBase(size_t id) : id_(id) {}
ActiveMsgBase::~ActiveMsgBase(){}

}

#endif