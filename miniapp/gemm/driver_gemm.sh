#!/bin/bash

for large in 0 1
do
    NROWS=1 NCOLS=1 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 1 run_gemm_2d.sh
    NROWS=1 NCOLS=1 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 1 run_gemm_2d.sh

    NROWS=4 NCOLS=2 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 8 run_gemm_2d.sh
    NROWS=4 NCOLS=2 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 8 run_gemm_2d.sh
    NROWS=4 NCOLS=2 MATRIX_SIZE=32768 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 8 run_gemm_2d.sh

    NROWS=8 NCOLS=8 MATRIX_SIZE=8192  BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d.sh
    NROWS=8 NCOLS=8 MATRIX_SIZE=16384 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d.sh
    NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d.sh
    NROWS=8 NCOLS=8 MATRIX_SIZE=65536 BLOCK_SIZE=256 LARGE=${large} sbatch -c 32 -n 64 run_gemm_2d.sh
done



OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=8192  BLOCK_SIZE=256 sbatch -c 32 -n 1 run_gemm_3d.sh
OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=16384 BLOCK_SIZE=256 sbatch -c 32 -n 1 run_gemm_3d.sh

OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=8192  BLOCK_SIZE=256 sbatch -c 32 -n 8 run_gemm_3d.sh
OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=16384 BLOCK_SIZE=256 sbatch -c 32 -n 8 run_gemm_3d.sh
OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=32768 BLOCK_SIZE=256 sbatch -c 32 -n 8 run_gemm_3d.sh

OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=8192  BLOCK_SIZE=256 sbatch -c 32 -n 64 run_gemm_3d.sh
OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=16384 BLOCK_SIZE=256 sbatch -c 32 -n 64 run_gemm_3d.sh
OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=32768 BLOCK_SIZE=256 sbatch -c 32 -n 64 run_gemm_3d.sh
OMP_NUM_THREADS=1 NUM_THREADS=16 MATRIX_SIZE=65536 BLOCK_SIZE=256 sbatch -c 32 -n 64 run_gemm_3d.sh


OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=8192  BLOCK_SIZE=8192  sbatch -c 32 -n 1 run_gemm_3d.sh
OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=16384 BLOCK_SIZE=16384 sbatch -c 32 -n 1 run_gemm_3d.sh

OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=8192  BLOCK_SIZE=4096  sbatch -c 32 -n 8 run_gemm_3d.sh
OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=16384 BLOCK_SIZE=8192  sbatch -c 32 -n 8 run_gemm_3d.sh
OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=32768 BLOCK_SIZE=16384 sbatch -c 32 -n 8 run_gemm_3d.sh

OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=8192  BLOCK_SIZE=2048 sbatch -c 32 -n 64 run_gemm_3d.sh
OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=16384 BLOCK_SIZE=4096 sbatch -c 32 -n 64 run_gemm_3d.sh
OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=32768 BLOCK_SIZE=8192 sbatch -c 32 -n 64 run_gemm_3d.sh
OMP_NUM_THREADS=16 NUM_THREADS=1 MATRIX_SIZE=65536 BLOCK_SIZE=8192 sbatch -c 32 -n 64 run_gemm_3d.sh



NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=64   LARGE=0 sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=64   LARGE=1 sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=128  LARGE=1 sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=256  LARGE=1 sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=512  LARGE=1 sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=1024 LARGE=1 sbatch -c 32 -n 64 run_gemm_2d.sh
NROWS=8 NCOLS=8 MATRIX_SIZE=32768 BLOCK_SIZE=2048 LARGE=1 sbatch -c 32 -n 64 run_gemm_2d.sh
