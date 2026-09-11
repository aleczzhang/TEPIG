#!/bin/bash
# Pre-download the 50 randomly-sampled plate sqlites (~420 GB total, each ~8 GB)
# on a node with internet access (Zaratan login/DTN), for run_extract_50pm.slurm.
# Safe to re-run: existing complete files are skipped by curl -C -.
set -e
cd "$(dirname "$0")"
URL="https://cellpainting-gallery.s3.amazonaws.com/cpg0012-wawer-bioactivecompoundprofiling/broad/workspace/backend/CDRP"
for p in $(tail -n +4 data/random_plate_sample.csv | cut -d, -f2); do
  [ -f "data/$p.sqlite" ] && { echo "have $p"; continue; }
  echo "downloading $p ..."
  curl -C - -o "data/$p.sqlite" "$URL/$p/$p.sqlite"
done
echo "all 50 sqlites present"
