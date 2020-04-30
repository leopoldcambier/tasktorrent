#ifndef TTOR_SHARED

#include "activemessage.hpp"

namespace ttor {

int ActiveMsgBase::get_id() const { return id_; };
ActiveMsgBase::ActiveMsgBase(int id) : id_(id) {};
ActiveMsgBase::~ActiveMsgBase(){};

}

#endif