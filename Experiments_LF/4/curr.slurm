#!/bin/bash
#SBATCH -n 20                # Number of cores
#SBATCH -t 7-00:00          # Runtime in D-HH:MM
## SBATCH -p batch            # Partition to submit to: gpu, batch
#SBATCH --mem-per-cpu=8000  # Memory (in MB) per cpu
#SBATCH -o log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
## SBATCH -w pcomp31       # pgpu01, alpha025, omega025
## SBATCH --export=ALL        # Pass any exported env vars to this script and its children

## module load anaconda/3
source /cluster/tufts/sinapovlab/gtatiy01/anaconda2/etc/profile.d/conda.sh
conda activate py3_7_tf1150
export MPLBACKEND='agg'

python curr.py
