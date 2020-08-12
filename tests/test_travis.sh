#!/bin/bash

echo "Config: TTOR_ROOT=${TTOR_ROOT}, EIGEN3_ROOT=${EIGEN3_ROOT}, ASAN_OPTIONS=${ASAN_OPTIONS}, TTOR_MPIRUN=${TTOR_MPIRUN}"
echo "        TEST=${TEST}, SHARED=${SHARED}, SAN=${SAN}"

if [ "${TEST}" == "MINIAPPS" ]
then
  cd ${TTOR_ROOT}/tutorial && make clean && make run
  if [ $? != "0" ]
  then
    exit 1
  fi
  cd ${TTOR_ROOT}/miniapp/dense_cholesky && cp Makefile.conf.travis Makefile.conf && make clean && make 3d_cholesky && mpirun -n 8 ./3d_cholesky
  if [ $? != "0" ]
  then
    exit 1
  fi
  cd ${TTOR_ROOT}/miniapp/3d_gemm && cp Makefile.conf.travis Makefile.conf && make clean && make 3d_gemm && mpirun --oversubscribe -n 8 ./3d_gemm
  if [ $? != "0" ]
  then
    exit 1
  fi
  cd ${TTOR_ROOT}/miniapp/sparse_cholesky && cp Makefile.conf.travis Makefile.conf && make clean && make snchol && mpirun -n 2 ./snchol neglapl_2_32.mm 10 2 0 5 0 NONE 5
  if [ $? != "0" ]
  then
    exit 1
  fi
else
  cd ${TTOR_ROOT} && ./tests/test_all.sh
  if [ $? != "0" ]
  then
    exit 1
  fi
fi
