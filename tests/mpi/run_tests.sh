#!/bin/bash

if [[ -z "${TTOR_MPIRUN}" ]]; then
  CMD_MPIRUN="mpirun"
else
  CMD_MPIRUN=${TTOR_MPIRUN}
fi

echo "Using ${CMD_MPIRUN} as mpirun command"

$CMD_MPIRUN -n 2 ./cholesky 2 5 10 1 2 --gtest_repeat=5 --gtest_break_on_failure

if [ $? != "0" ]
then
    exit 1
fi

$CMD_MPIRUN -n 4 ./cholesky 1 5 32 2 2 --gtest_repeat=5 --gtest_break_on_failure

if [ $? != "0" ]
then
    exit 1
fi

for nrank in 1 2 3 4
do
    $CMD_MPIRUN -n ${nrank} ./tests_active_msg_large --gtest_break_on_failure  --gtest_filter=-*large*

    if [ $? != "0" ]
    then
        exit 1
    fi
done

$CMD_MPIRUN -n 4 ./tests_comms_internals --gtest_repeat=10 --gtest_break_on_failure --gtest_filter=-*large*

if [ $? != "0" ]
then
    exit 1
fi   

for nrank in 1 2 3 4
do
    $CMD_MPIRUN -n ${nrank} ./tests_completion 2 0 --gtest_repeat=32 --gtest_break_on_failure

    if [ $? != "0" ]
    then
        exit 1
    fi    
done

for nrank in 1 2 3 4
do
    $CMD_MPIRUN -n ${nrank} ./tests_communicator 1 0 --gtest_filter=*mini

    if [ $? != "0" ]
    then
        exit 1
    fi

    $CMD_MPIRUN -n ${nrank} ./tests_communicator 1 0 --gtest_filter=*sparse_graph

    if [ $? != "0" ]
    then
        exit 1
    fi
done

$CMD_MPIRUN -n 2 ./tests_communicator 1 0 --gtest_filter=*ring

if [ $? != "0" ]
then
    exit 1
fi

$CMD_MPIRUN -n 2 ./tests_communicator 1 0 --gtest_filter=*pingpong

if [ $? != "0" ]
then
    exit 1
fi

$CMD_MPIRUN -n 1 ./tests_communicator 2 0 --gtest_filter=*critical

if [ $? != "0" ]
then
    exit 1
fi

for nrank in 4
do
  for nthread in 1 2
  do
    $CMD_MPIRUN -n ${nrank} ./tests_communicator ${nthread} 0
    if [ $? != "0" ]
    then
        exit 1
    fi

    $CMD_MPIRUN -n ${nrank} ./ddot_test ${nthread} 2 0
    if [ $? != "0" ]
    then
        exit 1
    fi    
  done
done

$CMD_MPIRUN -n 4 ./tests_communicator 2 0 --gtest_repeat=4 --gtest_break_on_failure

if [ $? != "0" ]
then
    exit 1
fi
