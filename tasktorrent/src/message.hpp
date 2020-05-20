#ifndef __TTOR_SRC_MESSAGE_HPP__
#define __TTOR_SRC_MESSAGE_HPP__

#ifndef TTOR_SHARED

#include <vector>
#include <mpi.h>

namespace ttor {

struct message
{
public:

    // This is used for two-steps messages
    // We first send a header (in exactly 1 MPI messages
    // Followed by potentially multiple MPI messages for the body

    // Header is encoded as
    // AM ID (size_t) | Body_size (size_t) | Arguments ...
    // Body is a user provided buffer

    // Source and destination
    int source;
    int dest;

    // Header
    bool header_processed;
    int header_tag;                         // 0 if count in B, 1 if count in MB
    std::vector<char> header_buffer;        // Header buffer (internally allocated/deallocated)
    MPI_Request header_request;             // Associated request

    // Body
    int body_tag;                           // Tag used to communicate the body
    char* body_send_buffer;                 // "Large" body buffer (user provided) on the sender
    char* body_recv_buffer;                 // "Large" body buffer (user provided) on the receiver
    size_t body_size;                       // Sizes
    std::vector<MPI_Request> body_requests; // Associated requests (multiple needed if message > 2 GB since we sent multiple messages)

    message();
};

}

#endif

#endif