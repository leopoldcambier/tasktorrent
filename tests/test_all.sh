#!/bin/bash

# Tutorial test

printf "\nTesting tutorial\n"
dir=$1
cd $dir/tutorial

make clean
make run

if [ $? != "0" ]
then
    exit 1
fi

# Distributed test

printf "\nTesting distributed mode\n"
mkdir -p $dir/build
cd $dir/build
rm -rf ./*
cmake .. -DTTOR_SHARED=OFF
cmake --build .
cp $dir/tests/mpi/run_tests.sh $dir/build/tests/mpi/run_tests.sh
cd $dir/build/tests/mpi

./run_tests.sh

if [ $? != "0" ]
then
    exit 1
fi

# Shared test

printf "\nTesting shared memory mode\n"
mkdir -p $dir/build
cd $dir/build
rm -rf ./*
cmake .. -DTTOR_SHARED=ON
cmake --build .
cp $dir/tests/shared/run_tests.sh $dir/build/tests/shared/run_tests.sh
cd $dir/build/tests/shared

./run_tests.sh

if [ $? != "0" ]
then
    exit 1
fi

printf "\nAll test runs are complete\n"