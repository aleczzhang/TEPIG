#!/bin/bash
#SBATCH --job-name=cdrp_diag213
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/diag213_%A_%a.out

# The two paper diagnostics that previously existed only as inline session
# computations, now scripted and run on the 213-plate cache:
#   task 0: permutation null of the gene screen -- SAME data and folds as the
#           screen itself (cdrp_50pm_merged.pkl, 50 leave-one-platemap-out folds,
#           ridge, 5 outcome shuffles) -> results/gene_scan_pmcv_ridge_cdrp_50pm_merged_perm.csv
#   task 1: structured-vs-averaged ridge diagnostic on the 213-plate cache,
#           pycytominer features, 5 platemap-grouped folds, oracle alpha
#           -> results/struct_ridge_cdrp_213_pycytominer_oracle.csv
# (A lasso permutation null is possible with
#    python cell_painting/screen/scan_platemap_cv.py --model lasso --preselect pycytominer --caches cdrp_50pm_merged.pkl --permute 5
#  but costs ~5 x the real lasso screen, i.e. several days of 16-core time; under
#  the null LassoCV picks lambda_max and returns R^2 ~ 0, so the ridge null is the
#  informative one.)
# Submit free:
#   mkdir -p logs   (from the repository root)
#   sbatch --partition=scavenger --qos=scavenger ../slurm/run_diagnostics213.slurm

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
source /home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
unset PYTHONPATH

cd /home/azzhang/scratch.staraptor-lab/TEPIG
export SCAN_NJOBS=$SLURM_CPUS_PER_TASK
echo "task $SLURM_ARRAY_TASK_ID ($(date))"
case $SLURM_ARRAY_TASK_ID in
  0) python cell_painting/screen/scan_platemap_cv.py --model ridge --caches cdrp_50pm_merged.pkl --permute 5 ;;
  1) python cell_painting/screen/struct_ridge_check.py --caches cdrp_213.pkl --preselect pycytominer --folds 5 --alpha oracle ;;
esac
echo "done $(date)"
