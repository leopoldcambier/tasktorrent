enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

set(SOURCES 
    serialization.cpp 
    util.cpp 
)

if(TTOR_SHARED)
    # Enable shared mode only
    message("Building TaskTorrent in shared memory mode")
    add_compile_options(-DTTOR_SHARED)
    add_library(TaskTorrent STATIC ${SOURCES})
    target_compile_options(TaskTorrent PUBLIC "-DTTOR_SHARED")
    target_include_directories(TaskTorrent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
else()
    # Adding communications routines
    message("Building TaskTorrent in distributed memory mode")    
    list(APPEND SOURCES communications.cpp)# communications.hpp)
    add_library(TaskTorrent STATIC ${SOURCES})
    target_include_directories(TaskTorrent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    find_package(MPI REQUIRED)
    target_link_libraries(TaskTorrent PUBLIC MPI::MPI_CXX)
endif()


