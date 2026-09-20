#!/bin/bash
#SBATCH --job-name=cdrp_scan50
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-49
#SBATCH --output=logs/scan50_%A_%a.out

# 50-fold leave-one-platemap-out LASSO screen over the 10 batch caches
# (cache/cdrp_pmz_b1..b10.pkl), one array task per held-out platemap.
# Submit on the free queue:
#   mkdir -p logs
#   sbatch --partition=scavenger --qos=scavenger TEPIG_python/slurm/run_scan_50pm.slurm
# The cheap RIDGE screen is a single short job:
#   sbatch --partition=scavenger --qos=scavenger --array=0 --time=00:30:00 \
#       --export=ALL,MODEL=ridge,WHOLE=1 TEPIG_python/slurm/run_scan_50pm.slurm
# When all 50 lasso tasks are done, combine into the CSV (fast, login node is fine):
#   python scan_platemap_cv.py --model lasso --preselect pycytominer --caches <b1..b10> --combine

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
# self-contained env (module numpy is too old for pycytominer); build once with:
#   python -m venv ~/scratch.staraptor-lab/venv_tepig
#   source ~/scratch.staraptor-lab/venv_tepig/bin/activate
#   pip install numpy pandas scikit-learn scipy joblib threadpoolctl pycytominer
source /home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
unset PYTHONPATH   # the python module injects its old site-packages into venvs

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting
export SCAN_NJOBS=$SLURM_CPUS_PER_TASK
CACHES=$(for i in $(seq 1 10); do echo -n "cdrp_pmz_b$i.pkl "; done)
MODEL=${MODEL:-lasso}
if [[ ${WHOLE:-0} == 1 ]]; then
  python scan_platemap_cv.py --model $MODEL --caches $CACHES
else
  python scan_platemap_cv.py --model $MODEL --preselect pycytominer \
      --caches $CACHES --fold $SLURM_ARRAY_TASK_ID
fi