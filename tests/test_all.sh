#!/bin/bash

dir=$1
cd $dir/tutorial

make clean
make run

if [ $? != "0" ]
then
    exit 1
fi

cd ../tests/mpi

./run_tests.sh

if [ $? != "0" ]
then
    exit 1
fi

cd ../shared

make clean
./run_tests.sh

if [ $? != "0" ]
then
    exit 1
fi

printf "\nAll test runs are complete\n"