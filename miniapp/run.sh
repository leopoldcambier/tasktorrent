#!/bin/bash
#
#SBATCH --time=00:01:00
#SBATCH --mem=64G
#SBATCH -J dmf
#SBATCH -p mc,normal
#SBATCH -o ttor.%j.out
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1

module load tbb scotch imkl openmpi gcc/8.1.0

export UPCXX_VERBOSE=1
export GASNET_PHYSMEM_MAX=50GB
export GASNET_PHYSMEM_NOPROBE=1
GASNET_BACKTRACE=1 upcxx-run -vv -shared-heap=25% -n 2 ./test_dmfchol /home/users/lcambier/spaND/mats/neglapl_3_64.mm 13 1 1
