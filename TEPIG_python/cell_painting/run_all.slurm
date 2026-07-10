#!/bin/bash
#SBATCH --job-name=cp_tepig_%a
#SBATCH --account=staraptor-lab-aac
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --array=0-65

# 66 jobs = 11 outcomes x 6 dose levels.
# Each job = one outcome at one dose level (~1,400 i.i.d. subjects).
OUTCOMES=(GLRX RAP1GAP IGFBP3 ATF6 CDKN1A DDIT4 CCNA2 SMC4 TSC22D3 MELK CELLCYCLE)

GENE_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
DOSE=$(( (SLURM_ARRAY_TASK_ID % 6) + 1 ))
GENE=${OUTCOMES[$GENE_IDX]}
echo "Job $SLURM_ARRAY_TASK_ID: gene=$GENE dose=$DOSE"

module purge
module load gcc/11.3.0
module load python/3.10.10/gcc

cd /home/azzhang/scratch.staraptor-lab/TEPIG/TEPIG_python/cell_painting

# NOTE: build_data.py must be run ONCE first (fast, ~15s) to create the cache:
#   python build_data.py
# It needs the LINCS-Pilot1 parquet files present under TEPIG/LINCS-Pilot1/.

python run_gene.py --gene $GENE --dose $DOSE
