#!/bin/bash

for nt in 16 31
do
    for kind in 0 1 2 3
    do
        NPROWS=1 NPCOLS=1 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch -c 32 -n 1 run_many.sh
        NPROWS=1 NPCOLS=1 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch -c 32 -n 1 run_many.sh
        NPROWS=1 NPCOLS=1 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch -c 32 -n 1 run_many.sh
        
        NPROWS=1 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch -c 32 -n 2 run_many.sh
        NPROWS=1 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch -c 32 -n 2 run_many.sh
        NPROWS=1 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch -c 32 -n 2 run_many.sh
        
        NPROWS=2 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch -c 32 -n 4 run_many.sh
        NPROWS=2 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch -c 32 -n 4 run_many.sh
        NPROWS=2 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch -c 32 -n 4 run_many.sh
        
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch -c 32 -n 8 run_many.sh
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch -c 32 -n 8 run_many.sh
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch -c 32 -n 8 run_many.sh
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch -c 32 -n 8 run_many.sh
        
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch -c 32 -n 16 run_many.sh
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch -c 32 -n 16 run_many.sh
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch -c 32 -n 16 run_many.sh
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch -c 32 -n 16 run_many.sh
        
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch -c 32 -n 32 run_many.sh
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch -c 32 -n 32 run_many.sh
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch -c 32 -n 32 run_many.sh
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch -c 32 -n 32 run_many.sh
        
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=65536 sbatch -c 32 -n 64 run_many.sh
    done
done
