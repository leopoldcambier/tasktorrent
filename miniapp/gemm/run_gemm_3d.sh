#!/bin/bash
#SBATCH --output=ttor_gemm_3d_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
mpirun -n ${SLURM_NTASKS} ./3d_gemm --matrix_size=${MATRIX_SIZE} --block_size=${BLOCK_SIZE} --n_threads=${NUM_THREADS} --test=0
