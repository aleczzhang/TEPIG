#!/bin/bash
# Extract the 163 replicate plates (data/replicate_plates.csv) in batches of 4,
# same pattern as zaratan_extract_batches.sh: run on a LOGIN node inside tmux;
# downloads happen here, processing happens in scavenger jobs. Batches of 4
# keep transient scratch ~<40 GB under the lab's 93 GiB quota.
# Safe to re-run: finished batch caches are skipped, downloads resume.
set -u
cd "$(dirname "$0")"
GAL="https://cellpainting-gallery.s3.amazonaws.com"
CPG="$GAL/cpg0012-wawer-bioactivecompoundprofiling/broad/workspace"
REPO=$(cd ../.. && pwd)

PLATES=($(tail -n +4 data/replicate_plates.csv | cut -d, -f2))
NB=4                                     # 163 plates -> 41 batches
NBATCH=$(( (${#PLATES[@]} + NB - 1) / NB ))
for b in $(seq 0 $((NBATCH - 1))); do
  batch=("${PLATES[@]:$((b * NB)):$NB}")
  out="cache/cdrp_rep_b$((b + 1)).pkl"
  if [[ -f $out ]]; then echo "== rep batch $((b + 1))/$NBATCH done, skipping =="; continue; fi
  echo "== rep batch $((b + 1))/$NBATCH: ${batch[*]} =="

  for p in "${batch[@]}"; do
    [[ -f data/$p.sqlite ]] && { echo "  have $p"; continue; }
    echo "  downloading $p ..."
    curl -C - -sS -o "data/$p.sqlite" "$CPG/backend/CDRP/$p/$p.sqlite" || exit 1
  done

  jid=$(sbatch --parsable --wait --partition=scavenger --qos=scavenger \
        --output="$HOME/extract_rep-%j.out" \
        --export=ALL,PLATES="${batch[*]}",OUT="$out" \
        "$REPO/TEPIG_python/slurm/run_extract_50pm.slurm")
  if [[ ! -f $out ]]; then
    echo "rep batch $((b + 1)) FAILED (job $jid); see ~/extract_rep-$jid.out"; exit 1
  fi
  echo "== rep batch $((b + 1)) saved $out (job $jid) =="
  df -h /scratch/zt1 | tail -1
done
echo "== all $NBATCH replicate batches built =="