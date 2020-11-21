#include <cassert>

#include "communications.hpp"

namespace ttor
{

Communicator_Base::Communicator_Base(int verb_) : 
    verb(verb_), 
    log(false), 
    logger(nullptr) {}
    
void Communicator_Base::set_logger(Logger *logger_)
{
    log = true;
    logger = logger_;
}

Communicator_Base::~Communicator_Base() = default;

}