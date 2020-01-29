#!/bin/bash

cd ${TTOR_ROOT}/build
cmake .. -DEIGEN_INCLUDE_DIRS=${TTOR_ROOT}/eigen/ -DTTOR_SHARED=OFF
make
cd ${TTOR_ROOT}/build/tests/mpi

CMD="mpirun -mca shmem posix -mca btl ^tcp -n 2 ./tests_communicator"
CMD_OSBS="mpirun -mca shmem posix -mca btl ^tcp -oversubscribe"

$CMD_OSBS -n 4 ./tests_comms_internals --gtest_repeat=10 --gtest_break_on_failure

if [ $? != "0" ]
then
    exit 1
fi   

for nrank in 1 2 3 4
do
    $CMD_OSBS -n ${nrank} ./tests_completion 2 0 --gtest_repeat=32 --gtest_break_on_failure

    if [ $? != "0" ]
    then
        exit 1
    fi    
done

for nrank in 1 2 3 4
do
    $CMD_OSBS -n ${nrank} ./tests_communicator 1 0 --gtest_filter=*mini

    if [ $? != "0" ]
    then
        exit 1
    fi

    $CMD_OSBS -n ${nrank} ./tests_communicator 1 0 --gtest_filter=*sparse_graph

    if [ $? != "0" ]
    then
        exit 1
    fi
done

$CMD 1 0 --gtest_filter=*ring

if [ $? != "0" ]
then
    exit 1
fi

$CMD 1 0 --gtest_filter=*pingpong

if [ $? != "0" ]
then
    exit 1
fi

mpirun -mca shmem posix -mca btl ^tcp -n 1 ./tests_communicator 2 0 --gtest_filter=*critical

if [ $? != "0" ]
then
    exit 1
fi

for nrank in 4
do
  for nthread in 1 2
  do
    $CMD_OSBS -n ${nrank} ./tests_communicator ${nthread} 0
    if [ $? != "0" ]
    then
        exit 1
    fi

    $CMD_OSBS -n ${nrank} ./ddot_test ${nthread} 2 0
    if [ $? != "0" ]
    then
        exit 1
    fi    
  done
done

$CMD_OSBS -n 4 ./tests_communicator 2 0 --gtest_repeat=4 --gtest_break_on_failure

if [ $? != "0" ]
then
    exit 1
fi
