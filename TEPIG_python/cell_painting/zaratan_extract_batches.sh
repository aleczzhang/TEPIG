#!/bin/bash
# Zaratan driver for the 50-platemap extraction under the 93 GiB scratch quota.
# Run on a LOGIN node inside tmux (login nodes have internet; compute nodes do
# not). Loops over batches of 5 plates: download the batch's sqlites here, then
# sbatch --wait a scavenger job to process them (the job deletes each sqlite
# after parking it and deletes the parked npys once the batch cache is built),
# then move to the next batch. Transient scratch stays under ~50 GB.
# Safe to re-run: finished batch caches are skipped, downloads resume.
set -u
cd "$(dirname "$0")"
GAL="https://cellpainting-gallery.s3.amazonaws.com"
CPG="$GAL/cpg0012-wawer-bioactivecompoundprofiling/broad/workspace"
REPO=$(cd ../.. && pwd)

PLATES=($(tail -n +4 data/random_plate_sample.csv | cut -d, -f2))
NB=5                                     # plates per batch -> 10 batches
for b in $(seq 0 9); do
  batch=("${PLATES[@]:$((b * NB)):$NB}")
  out="cache/cdrp_pmz_b$((b + 1)).pkl"
  if [[ -f $out ]]; then echo "== batch $((b + 1))/10 done, skipping =="; continue; fi
  echo "== batch $((b + 1))/10: ${batch[*]} =="

  for p in "${batch[@]}"; do
    [[ -f data/$p.sqlite ]] && { echo "  have $p"; continue; }
    echo "  downloading $p ..."
    curl -C - -sS -o "data/$p.sqlite" "$CPG/backend/CDRP/$p/$p.sqlite" || exit 1
  done

  jid=$(sbatch --parsable --wait --partition=scavenger --qos=scavenger \
        --output="$HOME/extract50-%j.out" \
        --export=ALL,PLATES="${batch[*]}",OUT="$out" \
        "$REPO/TEPIG_python/slurm/run_extract_50pm.slurm")
  if [[ ! -f $out ]]; then
    echo "batch $((b + 1)) FAILED (job $jid); see ~/extract50-$jid.out"; exit 1
  fi
  echo "== batch $((b + 1)) saved $out (job $jid) =="
done
echo "== all 10 batch caches built =="