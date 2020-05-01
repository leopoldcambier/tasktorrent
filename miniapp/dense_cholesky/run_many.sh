#!/bin/bash
#SBATCH -o ttor_cholesky_blocksize_%j.out

hostname
lscpu

BLOCK_SIZE[0]=32
BLOCK_SIZE[1]=64
BLOCK_SIZE[2]=128
BLOCK_SIZE[3]=256
BLOCK_SIZE[4]=512
BLOCK_SIZE[5]=1024
BLOCK_SIZE[6]=2048
    
mpirun -n ${SLURM_NTASKS} hostname

for i in 6
do

    echo ${MATRIX_SIZE}
    echo ${BLOCK_SIZE[i]}
    let NUM_BLOCKS=$((${MATRIX_SIZE}/${BLOCK_SIZE[i]}))
    echo ${NUM_BLOCKS}

    # ./cholesky block_size num_blocks n_threads verb nprows npcols kind log depslog test accumulate
    mpirun -n ${SLURM_NTASKS} ./cholesky ${BLOCK_SIZE[i]} ${NUM_BLOCKS} ${N_THREADS} 0 ${NPROWS} ${NPCOLS} ${KIND} 0 0 0 0
    echo "=================================="

done
