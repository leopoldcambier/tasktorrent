#!/bin/bash

# First argument in the command line is the number of iterations in the test
# Recommended ~ 4000.

mpirun -mca shmem posix -mca btl ^tcp -oversubscribe -n 4 ./tests_communicator 2 0 --gtest_repeat=$1 --gtest_break_on_failure --gtest_filter=-ttor.critical

mpirun -mca shmem posix -mca btl ^tcp -oversubscribe -n 1 ./tests_communicator 2 0 --gtest_repeat=32 --gtest_break_on_failure --gtest_filter=ttor.critical