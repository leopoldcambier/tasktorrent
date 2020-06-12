#!/bin/bash
#SBATCH --output=ttor_gemm_2d_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname
OMP_NUM_THREADS=1 mpirun -n ${SLURM_NTASKS} ./2d_gemm ${MATRIX_SIZE} ${BLOCK_SIZE} 16 ${NROWS} ${NCOLS} 0 ${LARGE}
