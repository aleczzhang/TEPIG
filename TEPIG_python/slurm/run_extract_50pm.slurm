#!/bin/bash
#SBATCH --job-name=cdrp_extract50
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --requeue

# SU note: charged max(cores, mem_GiB/4) x hours = 6 SU/h here, by ACTUAL walltime
# (~6-8 h expected -> ~40-50 SU). If the allocation is still the 80 SU starter, try
# the free preemptible queue instead (no SU charge; --reuse-parked makes requeues
# cheap since finished plates are skipped):
#   sbatch --partition=scavenger --qos=scavenger TEPIG_python/slurm/run_extract_50pm.slurm
# (check partition names with `sinfo -s` -- scavenger naming varies)

# Extract ONE randomly-sampled plate from each of the 50 fully-L1000-covered
# platemaps (data/random_plate_sample.csv, numpy seed 2026) into a single cache
# with one global GMM -> cache/cdrp_50pm.pkl. Downloads ~420 GB of sqlites from
# the Cell Painting Gallery (deleted as it goes); transient scratch peak is the
# parked per-plate npys, ~170 GB, plus one 8 GB sqlite.
#
# If compute nodes have no internet access, pre-download on the login/DTN node:
#   cd TEPIG_python/cell_painting && ./predownload_50pm.sh
# then submit this job (existing sqlites are reused, not re-downloaded).

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc
# use the scratch venv if present (see run_scan_50pm.slurm for how to build it)
VENV=/home/azzhang/scratch.staraptor-lab/venv_tepig/bin/activate
[ -f $VENV ] && source $VENV

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting
# PLATES and OUT can be injected per batch (zaratan_extract_batches.sh);
# defaults = the full 50-plate single-cache run (needs >200 GB scratch quota).
PLATES=${PLATES:-$(tail -n +4 data/random_plate_sample.csv | cut -d, -f2 | tr '\n' ' ')}
OUT=${OUT:-cache/cdrp_50pm.pkl}
echo "extracting plates: $PLATES -> $OUT"
python cdrp_extract.py --plates $PLATES --download --delete-sqlite \
    --reuse-parked --out $OUT
