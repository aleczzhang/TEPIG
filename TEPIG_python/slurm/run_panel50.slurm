#!/bin/bash
#SBATCH --job-name=cdrp_panel50
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-49
#SBATCH --output=logs/panel50_%A_%a.out

# Full TEPIG/CLUSSO/naive pipeline on the professor-approved top-10 genes from
# the 50-platemap screen, with grouped 40/10 platemap splits (5 seeds).
# One array task = one (gene, seed): 10 genes x 5 seeds = 50 tasks.
# Needs cache/cdrp_50pm_merged.pkl (python merge_pmz_caches.py, once).
# Submit free:  mkdir -p logs
#   sbatch --partition=scavenger --qos=scavenger TEPIG_python/slurm/run_panel50.slurm

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
source /home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
unset PYTHONPATH

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting
GENES=(HMGCS1 NSDHL HMGCR FDFT1 HERPUD1 XBP1 TMEM97 CREB3L2 INSIG1 CRELD2)
GENE=${GENES[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=$((42 + 100 * (SLURM_ARRAY_TASK_ID % 5)))
python run_panel.py $GENE --cache cdrp_50pm_merged.pkl --select pycytominer \
    --split-by-plate --seeds $SEED