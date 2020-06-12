#!/bin/bash
#SBATCH --output=ttor_gemm_3d_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
mpirun -n ${SLURM_NTASKS} ./3d_gemm ${MATRIX_SIZE} ${BLOCK_SIZE} ${NUM_THREADS} NONE 0 0
