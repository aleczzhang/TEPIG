#!/bin/bash
#SBATCH --job-name=cp_tepig_%a
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8192
#SBATCH --array=0-65
#
# 66 jobs = 11 outcomes x 6 dose levels.
# Each job runs ONE outcome at ONE dose level and, per the agreed protocol:
#   * prunes near-duplicate features at |r| <= 0.95   (68 -> ~34)
#   * refits TEPIG / CLUSSO / naive over 10 repeated runs (different splits)
#   * reports mean +/- sd of test MSE / R^2, and L1-normalised selection frequencies
#
# BILLING NOTE -- KEEP cpus-per-task=1.
# The allocation is only ~80 core-hours (0.08 kSU), and SLURM validates a job against
# its REQUESTED resources. The generous 24h wall clock is safe ONLY at 1 core:
#     1 core x 24h  =  24 core-hours per job   -> under the 80 limit   OK
#     8 cores x 24h = 192 core-hours per job   -> over  the 80 limit   HELD
# (An earlier run was held with reason AssocGrpBillingMinutes for exactly this.)
# Billing is charged on ACTUAL elapsed time (~15 min/job), so the real spend across
# all 66 jobs is only ~16 core-hours. Check headroom before submitting: sbalance

OUTCOMES=(GLRX RAP1GAP IGFBP3 ATF6 CDKN1A DDIT4 CCNA2 SMC4 TSC22D3 MELK CELLCYCLE)

GENE_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
DOSE=$(( (SLURM_ARRAY_TASK_ID % 6) + 1 ))
GENE=${OUTCOMES[$GENE_IDX]}
echo "Job $SLURM_ARRAY_TASK_ID: gene=$GENE dose=$DOSE"

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting

# Needs cache/cp_lincs_tensor.pkl (built once by build_data.py, or copied up).
python repeated_runs.py --gene $GENE --dose $DOSE --runs 10
