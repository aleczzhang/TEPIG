#!/bin/bash
#SBATCH --job-name=tepig_syn
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096
#SBATCH --array=0-69
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/syn_%A_%a.out

# Full simulation study: 7 n-values x 5 q-values x 2 sparsity-values = 70 tasks,
# 200 replicates each. Submit free (preemptible; tasks are requeued and a finished
# setting is skipped, so resubmitting is always safe):
#   mkdir -p logs   (from the repository root)
#   sbatch --partition=scavenger --qos=scavenger slurm/run_simulation_synthetic.slurm
# Then:  python simulation/summarize_synthetic.py --folder $FOLDER   (and --stat mean)
#        python simulation/plot_figures.py --folder $FOLDER
N_VALUES=(300 500 700 900 1100 1500 2000)
Q_VALUES=(10 50 100 150 200)
S_VALUES=(0.4 0.8)
FOLDER=${FOLDER:-rerun_2026_11}
REPS=${REPS:-200}

N_IDX=$(( SLURM_ARRAY_TASK_ID / 10 ))
Q_IDX=$(( (SLURM_ARRAY_TASK_ID % 10) / 2 ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))
N=${N_VALUES[$N_IDX]}
Q=${Q_VALUES[$Q_IDX]}
SP=${S_VALUES[$S_IDX]}
SPS=$(awk -v s=$SP 'BEGIN{printf "%02d", s*10}')

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
source /home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
unset PYTHONPATH

cd /home/azzhang/scratch.staraptor-lab/TEPIG
OUT=outputs/data/$FOLDER/simulation_synthetic_n${N}_q${Q}_s${SPS}_results.pkl
if [[ -f $OUT ]]; then echo "$OUT exists; skipping"; exit 0; fi
echo "Job $SLURM_ARRAY_TASK_ID: n=$N, q=$Q, sparsity=$SP -> $FOLDER  ($(date))"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # joblib workers own the cores
python simulation/simulation_synthetic.py --n $N --q $Q --sparsity $SP \
    --reps $REPS --folder $FOLDER --n-jobs $SLURM_CPUS_PER_TASK
echo "done $(date)"
