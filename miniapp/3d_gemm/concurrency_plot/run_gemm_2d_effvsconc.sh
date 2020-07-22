#!/bin/bash
#SBATCH --output=ttor_gemm_2d_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname


CORE[0]=1
CORE[1]=2
CORE[2]=4
CORE[3]=8
CORE[4]=16

for i in {0..4}
do
    OMP_NUM_THREADS=1 mpirun -n ${SLURM_NTASKS} ./../2d_gemm ${MATRIX_SIZE} ${BLOCK_SIZE} ${CORE[i]} ${NROWS} ${NCOLS} 0 ${LARGE}
done
