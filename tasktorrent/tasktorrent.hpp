#ifndef __TTOR_TASKTORRENT_HPP__
#define __TTOR_TASKTORRENT_HPP__

/**
 * The public interface to tasktorrent
 */

#if (!defined(TTOR_MPI)) && (!defined(TTOR_UPCXX)) && (!defined(TTOR_SHARED))
    static_assert(false , "Define either TTOR_MPI, TTOR_UPCXX or TTOR_SHARED. For instance, add -DTTOR_MPI to your compiler flags to use TaskTorrent with MPI");
#endif

#include "src/threadpool_shared.hpp"
#include "src/threadpool_dist.hpp"
#include "src/communications.hpp"
#include "src/communications_mpi.hpp"
#include "src/communications_upcxx.hpp"
#include "src/taskflow.hpp"
#include "src/views.hpp"
#include "src/mpi_utils.hpp"
#include "src/communications_extra.hpp"

#endif