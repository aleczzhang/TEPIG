#!/bin/bash
#SBATCH --job-name=cdrp_multi
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/cdrp_multi_%j.out
#
# Overnight multi-plate CDRP single-cell run: extract P plates -> one cache ->
# TEPIG / CLUSSO / naive panel over several outcome genes with repeated runs.
#
# SIZING -- do not guess: run `python compute_plan.py --host zaratan` after the
# first Zaratan timings land in results/compute_log.csv, and set PLATES to the
# count it reports for 12 h. Local (M4 Mac) numbers: ~5 min/plate extraction,
# ~80 s per gene-run (q=262, n=320) rising to ~160 s at n=1280 -- see compute_plan.
#
# BILLING: SLURM charges cpus-per-task x walltime against the allocation, so this
# job costs 2 x 12 = 24 core-hours. Check headroom first:  sbalance
# cpus-per-task=2 only helps the GMM (BLAS threads); the fits are single-threaded.
#
# DISK: --download --delete-sqlite keeps at most one 8 GB sqlite at a time plus
# ~1.8 GB parked cells per plate under cache/ (deleted after the cache is built).

PLATES=${PLATES:-"24277 26724 26071 26058 26128 26138 25593 24301"}   # 8 platemaps with full L1000 coverage (cdrp_extract.py --suggest 8)
GENES=${GENES:-"CCNA2 CELLCYCLE CDKN1A DDIT4 ATF6 GLRX"}
K=${K:-262}
RUNS=${RUNS:-10}
NP=$(echo $PLATES | wc -w)
CACHE=cdrp_${NP}plates.pkl

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting
mkdir -p logs results cache data

echo "== extract $NP plates: $PLATES"
python cdrp_extract.py --plates $PLATES --download --delete-sqlite \
       --out cache/$CACHE --no-diagnostics || exit 1

echo "== panel: k=$K runs=$RUNS genes=$GENES"
python run_panel.py --cache $CACHE --k $K --runs $RUNS $GENES

echo "== compute plan from measured timings"
python compute_plan.py --host zaratan --genes $(echo $GENES | wc -w) --runs $RUNS --k $K
