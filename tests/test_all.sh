#!/bin/bash

# Get ttor's source
dir="${TTOR_ROOT}"
if [ -z "$dir" ]
then
    echo "You need to pass ttor's source as TTOR_ROOT. Aborting tests."
    exit 1
fi
cmakeexist="${dir}/CMakeLists.txt"
echo $cmakeexist
if [[ -f "$cmakeexist" ]];
then
    echo "Found CMakeLists.txt"
else
    echo "You need to pass ttor's source as TTOR_ROOT, which should contain CMakeLists.txt. Aborting tests."
    exit 1
fi
printf "TTOR's source set to ${dir}"

printf "\n\n"
echo "Config: TTOR_ROOT=${TTOR_ROOT}"
echo "        EIGEN3_ROOT=${EIGEN3_ROOT}"
echo "        ASAN_OPTIONS=${ASAN_OPTIONS}"
echo "        TTOR_LAUNCHER=${TTOR_LAUNCHER}"
echo "        TTOR_TEST=${TTOR_TEST}"
echo "        TTOR_KIND=${TTOR_KIND}"
echo "        TTOR_SAN=${TTOR_SAN}"
echo "        TTOR_LARGE_TESTS=${TTOR_LARGE_TESTS}"
echo "        TTOR_MANY_RANKS=${TTOR_MANY_RANKS}"
echo "        TTOR_BUILD_TYPE=${TTOR_BUILD_TYPE}"
echo "        TTOR_UPCXX_PREFIX=${TTOR_UPCXX_PREFIX}"
echo "        TTOR_CXX=${TTOR_CXX}"
printf "\n\n"

TTOR_KIND="${TTOR_KIND:-MPI}"
TTOR_SAN="${TTOR_SAN:-OFF}"
TTOR_BUILD_TYPE="${TTOR_BUILD_TYPE:-Debug}"

# Shared and distributed tests
builddir="${dir}/build_${TTOR_BUILD_TYPE}_${TTOR_KIND}_${TTOR_SAN}"
mkdir -p $builddir
cd $builddir

printf "Building in ${builddir}...\n"
if [[ "${TTOR_KIND}" == "UPCXX" ]]; then
    UPCXX_THREADMODE=par cmake -DTTOR_KIND=${TTOR_KIND} -DCMAKE_BUILD_TYPE=${TTOR_BUILD_TYPE} -DTTOR_SAN=${TTOR_SAN} -DCMAKE_PREFIX_PATH=${TTOR_UPCXX_PREFIX} -DCMAKE_CXX_COMPILER=${TTOR_CXX} ..
else
    cmake -DTTOR_KIND=${TTOR_KIND} -DCMAKE_BUILD_TYPE=${TTOR_BUILD_TYPE} -DTTOR_SAN=${TTOR_SAN} ..
fi
cmake --build . -j
if [[ "${TTOR_KIND}" == "SHARED" ]]; then
    cp $dir/tests/shared/run_tests.sh $builddir/tests/shared/run_tests.sh
    cd $builddir/tests/shared
else
    cp $dir/tests/distributed/run_tests.sh $builddir/tests/distributed/run_tests.sh
    cd $builddir/tests/distributed
fi

printf "Testing ...\n"
./run_tests.sh

if [ $? != "0" ]
then
    exit 1
fi

printf "\n\nAll test runs are complete\n"
