#!/bin/bash
#SBATCH --output=ttor_gemm_2d_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
OMP_NUM_THREADS=1 mpirun -n ${SLURM_NTASKS} ./2d_gemm --matrix_size=${MATRIX_SIZE} --block_size=${BLOCK_SIZE} --n_threads=16 --nprows=${NROWS} --npcols=${NCOLS} --test=false --use_large=${LARGE}
