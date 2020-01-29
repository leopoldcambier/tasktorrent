enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

if(TTOR_SHARED)
    message("Building TaskTorrent in shared memory mode")
    add_compile_options(-DTTOR_SHARED)
    set(SOURCES serialization.cpp util.cpp)
else()
    message("Building TaskTorrent in distributed memory mode")
    set(SOURCES communications.cpp serialization.cpp util.cpp)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
endif()

include_directories(./)

add_library(TaskTorrent STATIC ${SOURCES})
