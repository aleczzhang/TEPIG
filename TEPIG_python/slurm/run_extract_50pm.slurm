#!/bin/bash
#SBATCH --job-name=cdrp_extract50
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting
PLATES=$(tail -n +4 data/random_plate_sample.csv | cut -d, -f2 | tr '\n' ' ')
echo "extracting plates: $PLATES"
python cdrp_extract.py --plates $PLATES --download --delete-sqlite \
    --reuse-parked --out cache/cdrp_50pm.pkl
