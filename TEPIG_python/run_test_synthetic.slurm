#!/bin/bash
#SBATCH --job-name=tepig_test
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096
#SBATCH --array=0-4
#SBATCH --output=logs/test_%A_%a.out

N_VALUES=(300 500 300 300 900)
Q_VALUES=(50  50  50  200 50)
S_VALUES=(0.8 0.8 0.4 0.8 0.8)

N=${N_VALUES[$SLURM_ARRAY_TASK_ID]}
Q=${Q_VALUES[$SLURM_ARRAY_TASK_ID]}
SP=${S_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Test job $SLURM_ARRAY_TASK_ID: n=$N, q=$Q, sparsity=$SP"

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python
mkdir -p logs

python simulation_synthetic.py --n $N --q $Q --sparsity $SP
