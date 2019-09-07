#!/bin/bash

make all || exit 1

CMD_OSBS="mpirun -mca shmem posix -mca btl ^tcp -oversubscribe"

for nrank in 1 2 4
do
    printf "number of processors assigned: ${nrank}\n"
    $CMD_OSBS -n ${nrank} ./tests_communicator 2 0 --gtest_repeat=64 --gtest_break_on_failure --gtest_filter=ttor.mini

    if [ $? != "0" ]
    then
        exit 1
    fi 

    $CMD_OSBS -n ${nrank} ./tests_communicator 2 0 --gtest_repeat=64 --gtest_break_on_failure --gtest_filter=ttor.mini,ttor.sparse_graph

    if [ $? != "0" ]
    then
        exit 1
    fi           
done