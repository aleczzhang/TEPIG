#!/bin/bash
#SBATCH --job-name=cdrp_scan50
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# 50-fold leave-one-platemap-out all-gene screen on cache/cdrp_50pm.pkl
# (produced by run_extract_50pm.slurm). Ridge first (minutes), then lasso.
# SCAN_NJOBS parallelizes the per-gene LassoCV fits within each fold.

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting
export SCAN_NJOBS=$SLURM_CPUS_PER_TASK
python scan_platemap_cv.py --model ridge --caches cdrp_50pm.pkl
python scan_platemap_cv.py --model lasso --preselect pycytominer --caches cdrp_50pm.pkl
