#!/bin/bash
#SBATCH --job-name=tepig_synthetic_%a
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096
#SBATCH --array=0-4

# Sample sizes: index 0->50, 1->100, 2->200, 3->300, 4->500
N_VALUES=(50 100 200 300 500)
N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python
python simulation_synthetic.py --n $N
