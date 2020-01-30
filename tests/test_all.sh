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

# Shared and distributed tests
for SHARED in OFF ON
do
    # With and without sanitizers
    for SAN in OFF ADDRESS THREAD UB
    do
        printf "\n\nTesting SHARED = ${SHARED}, SAN = ${SAN}\n\n"
        mkdir -p $dir/build
        cd $dir/build
        rm -rf ./*

        printf "Building ...\n"
        cmake -DTTOR_SHARED=${SHARED} -DCMAKE_BUILD_TYPE=Debug -DTTOR_SAN=${SAN} ..
        cmake --build .
        if [[ $SHARED -eq "OFF" ]]
        then
            cp $dir/tests/mpi/run_tests.sh $dir/build/tests/mpi/run_tests.sh
            cd $dir/build/tests/mpi
        else
            cp $dir/tests/shared/run_tests.sh $dir/build/tests/shared/run_tests.sh
            cd $dir/build/tests/shared
        fi

        printf "Testing ...\n"
        ./run_tests.sh

        if [ $? != "0" ]
        then
            exit 1
        fi
    done
done

printf "\n\nAll test runs are complete\n"