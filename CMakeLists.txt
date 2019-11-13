cmake_minimum_required(VERSION 3.13.0)
project(TaskTorrent)

enable_language(CXX)
set (CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

include_directories(src/)
set(SOURCES src/communications.cpp src/serialization.cpp src/util.cpp)

add_library(taskTorrent STATIC ${SOURCES})


