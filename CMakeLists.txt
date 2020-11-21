cmake_minimum_required(VERSION 3.12.4)

project(TaskTorrentTests VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(AddGoogleTest)

if(NOT TTOR_SAN)
  set(TTOR_SAN "OFF")
endif()

if(${TTOR_SAN} MATCHES "ADDRESS")
    message(STATUS "Adding -fsanitize=address")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -O2")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()
if(${TTOR_SAN} MATCHES "THREAD")
    message(STATUS "Adding -fsanitize=thread")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -O2") # Absurdingly slow without -O2
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=thread")
endif()
if(${TTOR_SAN} MATCHES "UB")
    message(STATUS "Adding -fsanitize=undefined")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -O2")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=undefined")
endif()

if(NOT TTOR_KIND)
  set(TTOR_KIND "MPI")
endif()

add_subdirectory(tasktorrent/src)
if(${TTOR_KIND} MATCHES "SHARED")
    add_subdirectory(tests/shared)
elseif(${TTOR_KIND} MATCHES "MPI")
    add_subdirectory(tests/distributed)
elseif(${TTOR_KIND} MATCHES "UPCXX")
    add_subdirectory(tests/distributed)
else()
    message(FATAL_ERROR "Need to set TTOR_KIND to SHARED, MPI or UPCXX")
endif()

