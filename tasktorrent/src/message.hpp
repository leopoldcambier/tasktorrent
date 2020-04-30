#ifndef __TTOR_SRC_MESSAGE_HPP__
#define __TTOR_SRC_MESSAGE_HPP__

#ifndef TTOR_SHARED

#include <vector>
#include <mpi.h>

namespace ttor {

struct message
{
public:
    std::vector<char> buffer;
    MPI_Request request;
    int other;
    int tag;
    message(int other);
};

}

#endif

#endif