enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

set(SOURCES 
    tasks.cpp
    threadpool_shared.cpp
    util.cpp
)

if(TTOR_SHARED)
    # Enable shared mode only
    message(STATUS "Building TaskTorrent in shared memory mode")
    add_compile_options(-DTTOR_SHARED)
    add_library(TaskTorrent STATIC ${SOURCES})
    target_compile_options(TaskTorrent PUBLIC "-DTTOR_SHARED")
else()
    # Adding communications routines
    message(STATUS "Building TaskTorrent in distributed memory mode")    
    list(APPEND SOURCES communications.cpp threadpool_dist.cpp active_messages.cpp message.cpp mpi_utils.cpp)
    add_library(TaskTorrent STATIC ${SOURCES})
    find_package(MPI REQUIRED)
    target_link_libraries(TaskTorrent PUBLIC MPI::MPI_CXX)
endif()
target_include_directories(TaskTorrent PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../")
