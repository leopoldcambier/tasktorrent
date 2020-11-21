#!/bin/bash

if [[ "${TTOR_TEST}" == "TESTSUITE" ]]; then
    echo "Running test suite"
    if [[ "${TTOR_KIND}" == "MPI" ]]; then
        export TTOR_LAUNCHER="mpirun --allow-run-as-root"
        ./tests/test_all.sh
        if [ $? != "0" ]
        then
            exit 1
        fi
    elif [[ "${TTOR_KIND}" == "UPCXX" ]]; then
        export TTOR_LAUNCHER="upcxx-run" 
        ./tests/test_all.sh
        if [ $? != "0" ]
        then
            exit 1
        fi
    else
        export TTOR_LAUNCHER="none" 
        ./tests/test_all.sh
        if [ $? != "0" ]
        then
            exit 1
        fi
    fi
else
    echo "Running miniapps"
    TTOR_LAUNCHER="mpirun -oversubscribe --allow-run-as-root"

    # Tutorial
    cd ${TTOR_ROOT}/tutorial && make clean && make && $TTOR_LAUNCHER -n 2 ./tuto && $TTOR_LAUNCHER -n 2 ./tuto_large_am
    if [[ $? != "0" ]]; then
        exit 1
    fi

    # Miniapps
    cp ${TTOR_ROOT}/miniapp/Makefile.conf.github ${TTOR_ROOT}/miniapp/Makefile.conf

    cd ${TTOR_ROOT}/miniapp/dense_cholesky && make clean && make 2d_cholesky && $TTOR_LAUNCHER -n 4 ./2d_cholesky
    if [[ $? != "0" ]]; then
        exit 1
    fi

    cd ${TTOR_ROOT}/miniapp/dense_cholesky && make clean && make 3d_cholesky && $TTOR_LAUNCHER -n 8 ./3d_cholesky
    if [[ $? != "0" ]]; then
        exit 1
    fi

    cd ${TTOR_ROOT}/miniapp/gemm && make clean && make 2d_gemm && $TTOR_LAUNCHER -n 4 ./2d_gemm
    if [[ $? != "0" ]]; then
        exit 1
    fi

    cd ${TTOR_ROOT}/miniapp/gemm && make clean && make 3d_gemm && $TTOR_LAUNCHER -n 8 ./3d_gemm
    if [[ $? != "0" ]]; then
        exit 1
    fi

    cd ${TTOR_ROOT}/miniapp/sparse_cholesky && make clean && make snchol && $TTOR_LAUNCHER -n 2 ./snchol
    if [[ $? != "0" ]]; then
        exit 1
    fi
fi