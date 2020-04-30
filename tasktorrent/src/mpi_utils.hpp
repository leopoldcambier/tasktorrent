#ifndef __TTOR_SRC_MPI_UTILS_HPP__
#define __TTOR_SRC_MPI_UTILS_HPP__

#ifndef TTOR_SHARED

#include <mpi.h>
#include <stdio.h>

#define TASKTORRENT_MPI_CHECK( call ) do { \
    int err = call; \
    if (err != MPI_SUCCESS) { \
        fprintf(stderr, "TaskTorrent: MPI error %d in file %s at line %i in function %s\n", \
              err, __FILE__, __LINE__, __func__); \
        MPI_Finalize(); \
        exit(1); \
    }   } while(0)

#endif

#endif