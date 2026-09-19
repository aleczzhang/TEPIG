#!/bin/bash
# Pre-download EVERYTHING run_extract_50pm.slurm needs, on a node with internet
# access (Zaratan login/DTN) -- compute nodes cannot reach the internet, so the
# batch job must find all of this already in data/:
#   1. the small metadata files (barcode table, platemap layouts, L1000 outcomes)
#   2. the 50 randomly-sampled plate sqlites (~420 GB total, each ~8 GB)
# Safe to re-run: complete files are skipped, partial sqlites resume (curl -C -).
set -e
cd "$(dirname "$0")"
GAL="https://cellpainting-gallery.s3.amazonaws.com"
CPG="$GAL/cpg0012-wawer-bioactivecompoundprofiling/broad/workspace"

[ -f data/barcode.csv ] || curl -fsS -o data/barcode.csv \
    "$CPG/metadata/platemaps/CDRP/barcode_platemap.csv"
[ -f data/cdrp_l1k.csv.gz ] || curl -fS -o data/cdrp_l1k.csv.gz \
    "$GAL/cpg0003-rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/L1000/replicate_level_l1k.csv.gz"
for pm in $(tail -n +4 data/random_plate_sample.csv | cut -d, -f1); do
  [ -f "data/$pm.txt" ] || curl -fsS -o "data/$pm.txt" \
      "$CPG/metadata/platemaps/CDRP/platemap/$pm.txt"
done
echo "metadata present"

for p in $(tail -n +4 data/random_plate_sample.csv | cut -d, -f2); do
  [ -f "data/$p.sqlite" ] && { echo "have $p"; continue; }
  echo "downloading $p ..."
  curl -C - -o "data/$p.sqlite" "$CPG/backend/CDRP/$p/$p.sqlite"
done
echo "all 50 sqlites present"