#!/bin/bash
#SBATCH --job-name=cdrp_panel213
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-49
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/panel213_%A_%a.out

# Replicate-scale panel run: top-10 genes x 5 grouped 40/10 PLATEMAP splits on
# cache/cdrp_213.pkl (213 plates, ~68k wells; all replicates of a platemap stay on
# the same split side via obs_platemap; inner CV folds are platemap-grouped too).
# One array task = one (gene, seed). Preemption-safe: a finished task's pkl is
# skipped and each method is checkpointed in results/partial/, so resubmitting the
# array after preemption only redoes unfinished work.
# Needs the merged cache first (see merge_all_caches.py docstring).
# Submit free:
#   mkdir -p logs   (from the repository root)
#   sbatch --partition=scavenger --qos=scavenger ../slurm/run_panel213.slurm
# Combine when all 50 are done:
#   python cell_painting/panel/combine_panel50.py --tag cdrp_213

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
source /home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
unset PYTHONPATH

cd /home/azzhang/scratch.staraptor-lab/TEPIG
GENES=(HMGCS1 NSDHL HMGCR FDFT1 HERPUD1 XBP1 TMEM97 CREB3L2 INSIG1 CRELD2)
GENE=${GENES[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((42 + 100 * (SLURM_ARRAY_TASK_ID % 5)))
CACHE=${CACHE:-cdrp_213.pkl}
echo "task $SLURM_ARRAY_TASK_ID: $GENE seed $SEED cache $CACHE  ($(date))"
python cell_painting/panel/run_panel.py $GENE --cache $CACHE --select pycytominer \
    --split-by-plate --seeds $SEED
echo "done $(date)"
