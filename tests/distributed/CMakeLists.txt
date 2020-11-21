message(STATUS "Adding distributed memory tests")

set(TESTS
    tests_active_msg 
    tests_cholesky 
    tests_communicator 
    tests_completion 
    tests_ddot 
    tests_random_graph
)

if(${TTOR_KIND} MATCHES "MPI")
    message(STATUS "Adding mpi tests")
    list(APPEND TESTS tests_active_msg_mpi tests_mpi)
endif()

find_package(Eigen3 3.3 REQUIRED)
foreach (test ${TESTS})
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE TaskTorrent)
    target_include_directories(${test} PRIVATE ${EIGEN3_INCLUDE_DIR})
    add_gtest(${test})
endforeach()
