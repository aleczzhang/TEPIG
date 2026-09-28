#!/bin/bash
#SBATCH --job-name=cdrp_panel213
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-49
#SBATCH --output=logs/panel213_%A_%a.out

# Replicate-scale panel run: top-10 genes x 5 grouped 40/10 PLATEMAP splits on
# cache/cdrp_213.pkl (213 plates, ~68k wells; all replicates of a platemap stay
# on the same split side via obs_platemap). ~10 h/task at this n.
# Needs the merged cache first (see merge_all_caches.py docstring).
# Submit free:
#   sbatch --partition=scavenger --qos=scavenger TEPIG_python/slurm/run_panel213.slurm

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
source /home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
unset PYTHONPATH

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting
GENES=(HMGCS1 NSDHL HMGCR FDFT1 HERPUD1 XBP1 TMEM97 CREB3L2 INSIG1 CRELD2)
GENE=${GENES[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((42 + 100 * (SLURM_ARRAY_TASK_ID % 5)))
python run_panel.py $GENE --cache cdrp_213.pkl --select pycytominer \
    --split-by-plate --seeds $SEED