cmake_minimum_required(VERSION 3.12.4)

project(TaskTorrentTests VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(AddGoogleTest)

add_subdirectory(src)
if(TTOR_SHARED)
    add_subdirectory(tests/shared)
else()
    add_subdirectory(tests/mpi)
endif()

