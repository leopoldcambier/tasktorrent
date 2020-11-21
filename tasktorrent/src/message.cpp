#ifdef TTOR_MPI

#include <memory>

#include "message.hpp"

namespace ttor {

    message_MPI::message_MPI() {
        this->source = -1;
        this->dest = -1;
        this->header_processed = false;
        this->header_tag = -1;
        this->body_tag = -1;
        this->body_send_buffer = nullptr;
        this->body_recv_buffer = nullptr;
        this->body_size = 0;
        this->header_buffer = std::make_unique<std::vector<char>>(0);
    }
}

#endif