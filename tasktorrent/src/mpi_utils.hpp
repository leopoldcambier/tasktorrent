#ifndef __TTOR_SRC_MPI_UTILS_HPP__
#define __TTOR_SRC_MPI_UTILS_HPP__

#ifndef TTOR_SHARED

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string>

#define TASKTORRENT_MPI_CHECK( call ) do { \
    int err = call; \
    if (err != MPI_SUCCESS) { \
        fprintf(stderr, "TaskTorrent: MPI error %d in file %s at line %i in function %s\n", \
              err, __FILE__, __LINE__, __func__); \
        MPI_Finalize(); \
        exit(1); \
    }   } while(0)

namespace ttor {

/**
 * Return this processor's rank within comm
 * 
 * \param[in] comm the MPI communicator
 * 
 * \return the rank of this processor within comm
 */
int comm_rank(MPI_Comm comm = MPI_COMM_WORLD);

/**
 * Return comm's size
 * 
 * \param[in] comm the MPI communicator
 * 
 * \return the number of processors in comm
 */
int comm_size(MPI_Comm comm = MPI_COMM_WORLD);

/**
 * Return the hostname
 * 
 * \return the hostname of this processor
 */
std::string processor_name();

}

#endif

#endif