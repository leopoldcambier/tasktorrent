#!/bin/bash

# Get ttor's source
dir=$1
if [ -z "$dir" ]
then
    echo "You need to pass ttor's source as first argument. Aborting tests."
    exit 1
fi
cmakeexist="${dir}/CMakeLists.txt"
echo $cmakeexist
if [[ -f "$cmakeexist" ]];
then
    echo "Found CMakeLists.txt"
else
    echo "You need to pass ttor's source as first argument, which should contain CMakeLists.txt. Aborting tests."
    exit 1
fi
printf "TTOR's source set to ${dir}"

# Tutorial test
printf "\nTesting tutorial\n"

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
        builddir="${dir}/build_${SHARED}_${SAN}"
        mkdir -p $builddir
        cd $builddir

        printf "Building in ${builddir}...\n"
        cmake -DTTOR_SHARED=${SHARED} -DCMAKE_BUILD_TYPE=Debug -DTTOR_SAN=${SAN} ..
        cmake --build .
        if [[ $SHARED -eq "OFF" ]]
        then
            cp $dir/tests/mpi/run_tests.sh $builddir/tests/mpi/run_tests.sh
            cd $builddir/tests/mpi
        else
            cp $dir/tests/shared/run_tests.sh $builddir/tests/shared/run_tests.sh
            cd $builddir/tests/shared
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