#!/bin/bash
#SBATCH --job-name=tepig_simulation
#SBATCH --account=staraptor-lab
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python
python simulation.py
