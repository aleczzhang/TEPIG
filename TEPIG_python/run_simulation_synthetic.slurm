#!/bin/bash
#SBATCH --job-name=tepig_syn_%a
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096
#SBATCH --array=0-69

# All combinations: 7 n-values x 5 q-values x 2 sparsity-values = 70 jobs
N_VALUES=(300 500 700 900 1100 1500 2000)
Q_VALUES=(10 50 100 150 200)
S_VALUES=(0.4 0.8)

# Map array index to (n, q, sparsity)
# index = n_idx*10 + q_idx*2 + s_idx
N_IDX=$(( SLURM_ARRAY_TASK_ID / 10 ))
Q_IDX=$(( (SLURM_ARRAY_TASK_ID % 10) / 2 ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${N_VALUES[$N_IDX]}
Q=${Q_VALUES[$Q_IDX]}
SP=${S_VALUES[$S_IDX]}

echo "Job $SLURM_ARRAY_TASK_ID: n=$N, q=$Q, sparsity=$SP"

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python
python simulation_synthetic.py --n $N --q $Q --sparsity $SP
