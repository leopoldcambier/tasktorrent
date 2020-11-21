enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

set(SOURCES 
    tasks.cpp
    threadpool_shared.cpp
    util.cpp
)

if(${TTOR_KIND} MATCHES "SHARED")
    # Enable shared mode only
    message(STATUS "Building TaskTorrent in shared memory mode")
    add_library(TaskTorrent STATIC ${SOURCES})
    target_compile_options(TaskTorrent PUBLIC "-DTTOR_SHARED")
else()
    list(APPEND SOURCES communications.cpp communications_extra.cpp threadpool_dist.cpp)
    if(${TTOR_KIND} MATCHES "MPI")
        message(STATUS "Building TaskTorrent in distributed memory mode using MPI")
        # Ttor + MPI
        list(APPEND SOURCES communications_mpi.cpp message.cpp mpi_utils.cpp)
        add_library(TaskTorrent STATIC ${SOURCES})
        target_compile_options(TaskTorrent PUBLIC "-DTTOR_MPI")
        # Find MPI
        find_package(MPI REQUIRED)
        target_link_libraries(TaskTorrent PUBLIC MPI::MPI_CXX)
    elseif(${TTOR_KIND} MATCHES "UPCXX")
        message(STATUS "Building TaskTorrent in distributed memory mode using UPCXX")
        # Ttor + UPCXX
        list(APPEND SOURCES communications_upcxx.cpp)
        add_library(TaskTorrent STATIC ${SOURCES})
        target_compile_options(TaskTorrent PUBLIC "-DTTOR_UPCXX")
        # Find UPCXX
        find_package(UPCXX REQUIRED)
        target_link_libraries(TaskTorrent PUBLIC UPCXX::upcxx)
    else()
    message(FATAL_ERROR "Need to set TTOR_KIND to SHARED, MPI or UPCXX")
    endif()
endif()
target_include_directories(TaskTorrent PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../")
