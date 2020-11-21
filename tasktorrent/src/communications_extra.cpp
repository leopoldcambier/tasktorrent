#if defined(TTOR_MPI) || defined(TTOR_UPCXX)

#include <cassert>
#include <memory>

#if defined(TTOR_MPI)
#include <mpi.h>
#include "mpi_utils.hpp"
#elif defined(TTOR_UPCXX)
#include <upcxx/upcxx.hpp>
#endif

#include "communications.hpp"
#include "communications_mpi.hpp"
#include "communications_extra.hpp"

namespace ttor
{

void comms_init() {
#if defined(TTOR_MPI)
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;
    MPI_Init_thread(NULL, NULL, req, &prov);
    assert(prov == req);
#elif defined(TTOR_UPCXX)
    upcxx::init();
#endif
}

void comms_finalize() {
#if defined(TTOR_MPI)
    MPI_Finalize();
#elif defined(TTOR_UPCXX)
    upcxx::finalize();
#endif
}

void comms_world_barrier() {
#if defined(TTOR_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
#elif defined(TTOR_UPCXX)
    upcxx::barrier();
#endif
}

int comms_world_rank() {
#if defined(TTOR_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#elif defined(TTOR_UPCXX)
    return upcxx::rank_me();
#endif

}

int comms_world_size() {
#if defined(TTOR_MPI)
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
#elif defined(TTOR_UPCXX)
    return upcxx::rank_n();
#endif
}

// TODO: improve this, that seems pretty unreliable
#if defined(TTOR_UPCXX) && (defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)))
#include <unistd.h>
#endif


std::string comms_hostname() {
    #if defined(TTOR_MPI)
        return mpi_processor_name();
    #elif defined(TTOR_UPCXX)
        #if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
            char name[256];
            name[255] = '\0';
            int err = gethostname(name, 255);
            if(err != 0) return "Error";
            else return name;
        #elif
            return "Unknown";
        #endif
#endif
}

std::unique_ptr<Communicator> make_communicator_world(int verb) {
#if defined(TTOR_MPI)
    return std::make_unique<Communicator_MPI>(MPI_COMM_WORLD, verb);
#elif defined(TTOR_UPCXX)
    return std::make_unique<Communicator_UPCXX>(verb);
#endif
}

} // namespace ttor

#endif