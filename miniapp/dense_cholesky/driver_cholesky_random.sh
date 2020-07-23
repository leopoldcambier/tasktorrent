#!/bin/bash

RANDOM_SIZES=64  NPROWS=8 NPCOLS=8 NUM_BLOCKS=1024 BLOCK_SIZE=64 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=80  NPROWS=8 NPCOLS=8 NUM_BLOCKS=1024 BLOCK_SIZE=64 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=96  NPROWS=8 NPCOLS=8 NUM_BLOCKS=1024 BLOCK_SIZE=64 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=112 NPROWS=8 NPCOLS=8 NUM_BLOCKS=1024 BLOCK_SIZE=64 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=128 NPROWS=8 NPCOLS=8 NUM_BLOCKS=1024 BLOCK_SIZE=64 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh

RANDOM_SIZES=128 NPROWS=8 NPCOLS=8 NUM_BLOCKS=512 BLOCK_SIZE=128 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=160 NPROWS=8 NPCOLS=8 NUM_BLOCKS=512 BLOCK_SIZE=128 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=192 NPROWS=8 NPCOLS=8 NUM_BLOCKS=512 BLOCK_SIZE=128 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=224 NPROWS=8 NPCOLS=8 NUM_BLOCKS=512 BLOCK_SIZE=128 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=256 NPROWS=8 NPCOLS=8 NUM_BLOCKS=512 BLOCK_SIZE=128 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh

RANDOM_SIZES=256 NPROWS=8 NPCOLS=8 NUM_BLOCKS=256 BLOCK_SIZE=256 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=320 NPROWS=8 NPCOLS=8 NUM_BLOCKS=256 BLOCK_SIZE=256 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=384 NPROWS=8 NPCOLS=8 NUM_BLOCKS=256 BLOCK_SIZE=256 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=448 NPROWS=8 NPCOLS=8 NUM_BLOCKS=256 BLOCK_SIZE=256 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh
RANDOM_SIZES=512 NPROWS=8 NPCOLS=8 NUM_BLOCKS=256 BLOCK_SIZE=256 N_THREADS=16 KIND=2 sbatch -c 32 -n 64 run_cholesky.sh