#!/bin/bash
#SBATCH -o ttor_spchol_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname

OMP_NUM_THREADS=1 mpirun -n ${SLURM_NTASKS} ./snchol --matrix=${MATRIX} --nlevels=${NLEVELS} --block_size=${BLOCK_SIZE} --n_threads=${N_THREADS}