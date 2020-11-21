#!/bin/bash
#SBATCH -o ttor_chol_%j.out

hostname
lscpu

mpirun -n ${SLURM_NTASKS} hostname

echo ${KIND}
echo ${RANDOM_SIZES}

# ./cholesky block_size num_blocks n_threads verb nprows npcols kind log depslog test accumulate random_sizes
OMP_NUM_THREADS=1 mpirun -n ${SLURM_NTASKS} ./2d_cholesky --block_size=${BLOCK_SIZE} --num_blocks=${NUM_BLOCKS} --n_threads=${N_THREADS} --nprows=${NPROWS} --npcols=${NPCOLS} --kind=${KIND} --test=false  --upper_block_size=${RANDOM_SIZES}
