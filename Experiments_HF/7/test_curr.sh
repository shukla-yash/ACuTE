#!/bin/bash
#SBATCH -n 15                # Number of cores
#SBATCH -t 0-11:30          # Runtime in D-HH:MM
#SBATCH -p gpu               # Partition: gpu, preempt
#SBATCH --mem=20g
##SBATCH --mem-per-cpu=20000  # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
export MPLBACKEND='agg'
python test_curr.py
