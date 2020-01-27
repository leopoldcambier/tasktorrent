#!/bin/bash

cd ${TTOR_ROOT}/build
cmake .. -DEIGEN_INCLUDE_DIRS=${TTOR_ROOT}/eigen-eigen-323c052e1731/ -DTTOR_SHARED=ON
make
cd ${TTOR_ROOT}/build/tests/shared

./tests_serialize

if [ $? != "0" ]
then
    exit 1
fi

./tests 32 1 32 0
