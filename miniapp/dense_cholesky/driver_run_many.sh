#!/bin/bash

for nt in 16
do
    for kind in 3
    do
        NPROWS=1 NPCOLS=1 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch --nodelist=compute-1-1 -c 32 -n 1 run_many.sh
        NPROWS=1 NPCOLS=1 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch --nodelist=compute-1-2 -c 32 -n 1 run_many.sh
        NPROWS=1 NPCOLS=1 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch --nodelist=compute-1-3 -c 32 -n 1 run_many.sh
        
        NPROWS=1 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[30-31] -c 32 -n 2 run_many.sh
        NPROWS=1 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[30-31] -c 32 -n 2 run_many.sh
        NPROWS=1 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[30-31] -c 32 -n 2 run_many.sh
        
        NPROWS=2 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[30-33] -c 32 -n 4 run_many.sh
        NPROWS=2 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[30-33] -c 32 -n 4 run_many.sh
        NPROWS=2 NPCOLS=2 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[30-33] -c 32 -n 4 run_many.sh
        
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh
        NPROWS=2 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[17-24] -c 32 -n 8 run_many.sh
        
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh
        NPROWS=4 NPCOLS=4 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[1-16] -c 32 -n 16 run_many.sh
        
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh
        NPROWS=4 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[1-32] -c 32 -n 32 run_many.sh
        
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=4096  sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=8192  sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=16384 sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=32768 sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
        NPROWS=8 NPCOLS=8 KIND=${kind} N_THREADS=${nt} MATRIX_SIZE=65536 sbatch --nodelist=compute-1-[1-34],compute-3-[1-30] -c 32 -n 64 run_many.sh
    done
done
