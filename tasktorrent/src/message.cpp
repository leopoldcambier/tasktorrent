#ifndef TTOR_SHARED

#include "message.hpp"

namespace ttor {

    message::message() {
        this->source = -1;
        this->dest = -1;
        this->header_processed = false;
        this->header_tag = -1;
        this->body_tag = -1;
        this->body_send_buffer = nullptr;
        this->body_recv_buffer = nullptr;
        this->body_size = 0;
    }
}

#endif