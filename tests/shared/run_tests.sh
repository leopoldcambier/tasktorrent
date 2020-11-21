#!/bin/bash

TTOR_LARGE_TESTS="${TTOR_LARGE_TESTS:-ON}"
echo "TTOR_LARGE_TESTS = ${TTOR_LARGE_TESTS}"

if [[ "ON" = ${TTOR_LARGE_TESTS} ]]; then
    echo "Testing large buffers"
    extraargs="--gtest_break_on_failure"
else
    echo "NOT testing very large buffers"
    extraargs="--gtest_break_on_failure --gtest_filter=-*large*"
fi

./tests_serialize ${extraargs}
if [ $? != "0" ]
then
    exit 1
fi

./tests 8 0 0 0 --gtest_break_on_failure --gtest_filter=graph.mini --gtest_repeat=100
if [ $? != "0" ]
then
        exit 1
fi

./tests 8 0 0 0 --gtest_break_on_failure --gtest_filter=reduction.* --gtest_repeat=100
if [ $? != "0" ]
then
        exit 1
fi

for nthreads in 2 8 32 
do
    ./tests ${nthreads} 1 32 0 ${extraargs}
    if [ $? != "0" ]
    then
            exit 1
    fi
done