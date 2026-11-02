#!/bin/bash
#SBATCH --job-name=cdrp_panel50
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-49
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/panel50_%A_%a.out

# One-plate-per-platemap panel (50 plates, cache/cdrp_50pm_merged.pkl): top-10
# genes x 5 grouped 40/10 platemap splits, one (gene, seed) per task. Same
# preemption safety as run_panel213.slurm. Submit free:
#   mkdir -p logs   (from the repository root)
#   sbatch --partition=scavenger --qos=scavenger ../slurm/run_panel50.slurm
# Combine:  python cell_painting/panel/combine_panel50.py --tag cdrp_50pm_merged

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
source /home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
unset PYTHONPATH

cd /home/azzhang/scratch.staraptor-lab/TEPIG
GENES=(HMGCS1 NSDHL HMGCR FDFT1 HERPUD1 XBP1 TMEM97 CREB3L2 INSIG1 CRELD2)
GENE=${GENES[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((42 + 100 * (SLURM_ARRAY_TASK_ID % 5)))
echo "task $SLURM_ARRAY_TASK_ID: $GENE seed $SEED  ($(date))"
python cell_painting/panel/run_panel.py $GENE --cache cdrp_50pm_merged.pkl --select pycytominer \
    --split-by-plate --seeds $SEED
echo "done $(date)"
