#ifndef TTOR_SHARED

#include "message.hpp"

namespace ttor {

    message::message() {
        this->other = -1;
        this->header_processed = false;
        this->header_tag = -1;
        this->body_tag = -1;
        this->body_buffer = nullptr;
        this->body_size = 0;
    }
}

#endif