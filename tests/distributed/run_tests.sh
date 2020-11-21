#!/bin/bash

TTOR_LAUNCHER="${TTOR_LAUNCHER:-mpirun}"
TTOR_LARGE_TESTS="${TTOR_LARGE_TESTS:-ON}"
TTOR_MANY_RANKS="${TTOR_MANY_RANKS:-ON}"

echo "TTOR_LAUNCHER = ${TTOR_LAUNCHER}"
echo "TTOR_LARGE_TESTS = ${TTOR_LARGE_TESTS}"
echo "TTOR_MANY_RANKS = ${TTOR_MANY_RANKS}"

if [[ "ON" = ${TTOR_MANY_RANKS} ]]; then
    echo "Using from 1 to 8 ranks"
    ranks=(1 2 3 4 6 8)
else
    echo "Using from 1 to 4 ranks"
    ranks=(1 2 3 4)
fi

if [[ "ON" = ${TTOR_LARGE_TESTS} ]]; then
    echo "Testing large messages"
    extraargs="--gtest_break_on_failure"
else
    echo "NOT testing very large messages"
    extraargs="--gtest_break_on_failure --gtest_filter=-*large*"
fi

##### Cholesky

for nranks in "${ranks[@]}"
do
  for nthreads in 1 2
  do
    $TTOR_LAUNCHER -n ${nranks} ./tests_cholesky ${nthreads} ${extraargs}
    if [ $? != "0" ]
    then
        exit 1
    fi    
  done
done

##### Ddot

for nranks in "${ranks[@]}"
do
  for nthreads in 1 2
  do
    $TTOR_LAUNCHER -n ${nranks} ./tests_ddot ${nthreads} ${extraargs}
    if [ $? != "0" ]
    then
        exit 1
    fi    
  done
done

##### Random graph

for nranks in "${ranks[@]}"
do
    $TTOR_LAUNCHER -n ${nranks} ./tests_random_graph ${extraargs}
    if [ $? != "0" ]
    then
        exit 1
    fi
done

##### Communicator

for nranks in "${ranks[@]}"
do
    for nthreads in 1 2 3 4
    do
        $TTOR_LAUNCHER -n ${nranks} ./tests_communicator ${nthreads} --gtest_repeat=5  ${extraargs}
        if [ $? != "0" ]
        then
            exit 1
        fi
    done
done

##### Active Msg

for nranks in "${ranks[@]}"
do
    $TTOR_LAUNCHER -n ${nranks} ./tests_active_msg ${extraargs}
    if [ $? != "0" ]
    then
        exit 1
    fi   
done

if [[ "MPI" = ${TTOR_MANY_RANKS} ]]; then
    for nranks in "${ranks[@]}"
    do
        $TTOR_LAUNCHER -n ${nranks} ./tests_active_msg_mpi ${extraargs}
        if [ $? != "0" ]
        then
            exit 1
        fi   
    done
fi

##### MPI

if [[ "MPI" = ${TTOR_MANY_RANKS} ]]; then
    for nranks in "${ranks[@]}"
    do
        $TTOR_LAUNCHER -n ${nranks} ./tests_mpi ${extraargs}
        if [ $? != "0" ]
        then
            exit 1
        fi   
    done
fi

##### Completion

for nranks in "${ranks[@]}"
do
    $TTOR_LAUNCHER -n ${nranks} ./tests_completion --gtest_repeat=100 ${extraargs}
    if [ $? != "0" ]
    then
        exit 1
    fi   
done