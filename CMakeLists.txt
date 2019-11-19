cmake_minimum_required(VERSION 3.13.0)
project(TaskTorrent)

enable_language(CXX)
set (CMAKE_CXX_STANDARD 14)

if(TTOR_SHARED)
    message("Building TaskTorrent in shared memory mode")
    add_compile_options(-DTTOR_SHARED)
    set(SOURCES src/serialization.cpp src/util.cpp)
else()
    message("Building TaskTorrent in distributed memory mode")
    find_package(MPI REQUIRED)
    set(SOURCES src/communications.cpp src/serialization.cpp src/util.cpp)
endif()

include_directories(src/)

add_library(TaskTorrent STATIC ${SOURCES})


