#!/bin/bash
#SBATCH --job-name=cp_tepig_%a
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=00:45:00
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
# BILLING NOTE -- the wall clock is the binding constraint, so do not raise it.
# SLURM charges the array's REQUESTED resources against the allocation, summed over
# ALL jobs -- not per job. With ~70 core-hours of headroom and 66 jobs:
#     max time per job = 70 / 66 ~= 1.06 h  ->  must stay UNDER ~1 hour
#     1 core x 45 min x 66 = 49.5 core-hours  -> fits            OK
#     1 core x 24 h  x 66  = 1584 core-hours  -> only 1 job runs, rest HELD
#                                                (reason: AssocGrpBillingMinutes)
# A job only needs ~15 min, so 45 min is already a 3x margin. Keep cpus-per-task=1
# (the code is single-threaded; more cores multiply the cost for no speedup).
# Check headroom before submitting:  sbalance

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
