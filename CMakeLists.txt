cmake_minimum_required(VERSION 3.12.4)
project(TaskTorrentTests)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(AddGoogleTest)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src)
if(TTOR_SHARED)
    message("Building TaskTorrent in shared memory mode")
    add_compile_options(-DTTOR_SHARED)
    add_subdirectory(tests/shared)
else()
    message("Building TaskTorrent in distributed memory mode")
    find_package(MPI REQUIRED)
    add_subdirectory(tests/mpi)
endif()

