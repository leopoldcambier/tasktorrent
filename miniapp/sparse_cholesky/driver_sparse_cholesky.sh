#!/bin/bash

FOLDER="/fastscratch/lcambier/laplacians"

MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=1  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=2  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=4  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=8  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 2  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 4  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 8  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_64.mm NLEVELS=11 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 16 ./run_sparse_cholesky.sh

MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=1  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=2  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=4  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=8  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 2  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 4  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 8  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_96.mm NLEVELS=12 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 16 ./run_sparse_cholesky.sh

MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=1  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=2  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=4  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=8  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 2  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 4  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 8  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_128.mm NLEVELS=13 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 16 ./run_sparse_cholesky.sh

MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=4  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=8  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 2  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 4  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 8  ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 16 ./run_sparse_cholesky.sh
MATRIX=${FOLDER}/neglapl_3_160.mm NLEVELS=14 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 32 ./run_sparse_cholesky.sh

# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=4  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=8  sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 1  ./run_sparse_cholesky.sh
# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 2  ./run_sparse_cholesky.sh
# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 4  ./run_sparse_cholesky.sh
# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 8  ./run_sparse_cholesky.sh
# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 16 ./run_sparse_cholesky.sh
# MATRIX=${FOLDER}/neglapl_3_192.mm NLEVELS=15 BLOCK_SIZE=256 N_THREADS=16 sbatch -c 32 -n 32 ./run_sparse_cholesky.sh
