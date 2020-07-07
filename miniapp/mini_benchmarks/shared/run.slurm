#!/bin/bash
#SBATCH -o wait_benchmarks.%j.out
#SBATCH -n 1
#SBATCH -c 32

ml intel/python/2019-3.6.8
python3 wait_benchmarks.py
