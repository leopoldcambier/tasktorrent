#!/bin/bash

for large in 1
do
    NROWS=1 NCOLS=1 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 1 run_gemm_2d_effvsconc.sh
    NROWS=1 NCOLS=1 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 1 run_gemm_2d_effvsconc.sh

    NROWS=1 NCOLS=2 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 2 run_gemm_2d_effvsconc.sh
    NROWS=1 NCOLS=2 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 2 run_gemm_2d_effvsconc.sh

    NROWS=2 NCOLS=2 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 4 run_gemm_2d_effvsconc.sh
    NROWS=2 NCOLS=2 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 4 run_gemm_2d_effvsconc.sh

    NROWS=4 NCOLS=2 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 8 run_gemm_2d_effvsconc.sh
    NROWS=4 NCOLS=2 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 8 run_gemm_2d_effvsconc.sh
    NROWS=4 NCOLS=2 MATRIX_SIZE=32768 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 8 run_gemm_2d_effvsconc.sh

    NROWS=4 NCOLS=4 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 16 run_gemm_2d_effvsconc.sh
    NROWS=4 NCOLS=4 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 16 run_gemm_2d_effvsconc.sh
    NROWS=4 NCOLS=4 MATRIX_SIZE=32768 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 16 run_gemm_2d_effvsconc.sh

    NROWS=4 NCOLS=8 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 32 run_gemm_2d_effvsconc.sh
    NROWS=4 NCOLS=8 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 32 run_gemm_2d_effvsconc.sh
    NROWS=4 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 32 run_gemm_2d_effvsconc.sh

    NROWS=8 NCOLS=8 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d_effvsconc.sh
    NROWS=8 NCOLS=8 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d_effvsconc.sh
    NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d_effvsconc.sh
    NROWS=8 NCOLS=8 MATRIX_SIZE=65536 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d_effvsconc.sh
done
