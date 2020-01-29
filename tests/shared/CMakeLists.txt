include_directories("../../src/")
include_directories(${EIGEN_INCLUDE_DIRS})

foreach (test tests tests_serialize)
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE TaskTorrent)
    add_gtest(${test})
endforeach()
