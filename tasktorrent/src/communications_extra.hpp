#ifndef __TTOR_SRC_COMMUNICATIONS_EXTRA_HPP__
#define __TTOR_SRC_COMMUNICATIONS_EXTRA_HPP__

#if defined(TTOR_MPI) || defined(TTOR_UPCXX)

#include <memory>
#include "communications.hpp"
#include "communications_mpi.hpp"
#include "communications_upcxx.hpp"

/**
 * The TaskTorrent namespace
 */
namespace ttor
{

    /**
     * \brief Initializes communications
     * 
     * \details Calls `MPI_Init_Thread(...)` or `upcxx::init()`, depending on whether `TTOR_MPI` or `TTOR_UPCXX` are defined
     */
    void comms_init();

    /**
     * \brief Finalizes communications
     * 
     * \details Calls `MPI_Finalize()` or `upcxx::finalize()`, depending on whether `TTOR_MPI` or `TTOR_UPCXX` are defined
     */
    void comms_finalize();

    /**
     * \brief Communication (world) barrier
     * 
     * \details Calls `MPI_Barrier(MPI_COMM_WORLD)` or `upcxx::barrier()`, depending on whether `TTOR_MPI` or `TTOR_UPCXX` are defined
     */
    void comms_world_barrier();

    /**
     * \brief Returns the (world) rank
     * 
     * \details Calls `MPI_Comm_rank(MPI_COMM_WORLD)` or `upcxx::rank_me()`, depending on whether `TTOR_MPI` or `TTOR_UPCXX` are defined
     * 
     * \return This processor's rank in the world communicator
     * 
     * \post `0 <= comms_world_rank() < comms_world_size()`
     */
    int comms_world_rank();

    /**
     * \brief Returns the (world) size
     * 
     * \details Calls `MPI_Comm_size(MPI_COMM_WORLD)` or `upcxx::rank_n()`, depending on whether `TTOR_MPI` or `TTOR_UPCXX` are defined
     * 
     * \return The size of the world communicator
     * 
     * \post `0 <= comms_world_rank() < comms_world_size()`
     * 
     */
    int comms_world_size();

    /**
     * \brief Returns the hostname
     * 
     * \details This is only informative and may not return anything useful.
     *          If `TTOR_MPI` is defined, uses `MPI_Get_processor_name()`.
     *          Otherwise, on Linux and Mac, uses POSIX's `gethostname()`.
     *          Otherwise, returns "Unknown".
     */
    std::string comms_hostname();

    /**
     * \brief Creates a communicator
     * 
     * \details Creates a `Communicator_MPI` or `Communicator_UPCXX`, depending on whether `TTOR_MPI` or `TTOR_UPCXX` are defined
     * 
     * \param[in] verb The level of verbosity. 0 is none, > 0 is more and more.
     * 
     * \pre `verb >= 0`
     */
    std::unique_ptr<Communicator> make_communicator_world(int verb = 0);

} // namespace ttor

#endif

#endif
