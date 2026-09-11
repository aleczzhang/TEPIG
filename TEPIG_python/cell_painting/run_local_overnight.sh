#!/bin/bash
# Local (Mac) overnight run: extract P plates -> one cache -> the pre-screening
# comparison (all arms) over a gene panel. Every step logs time + peak RSS to
# results/compute_log.csv; per-arm results go to results/panel_<arm>_<cache>.pkl.
#
#   nohup caffeinate -i ./run_local_overnight.sh > logs/overnight.txt 2>&1 &
#
# caffeinate -i stops IDLE sleep only: keep the lid open and the Mac plugged in
# (closing the lid sleeps the machine and kills the download). Closing the
# Terminal window is fine (nohup).
#
# Disk: --delete-sqlite keeps at most one 8 GB sqlite at a time; parked cells are
# ~1.8 GB/plate and are removed once the cache is built. NOTE: this also deletes
# data/24277.sqlite after use -- drop --delete-sqlite if you want to keep it.
set -u
cd "$(dirname "$0")"
PY=${PY:-../../.venv_inspect/bin/python}
PLATES=${PLATES:-"24277 26724 26071 26058"}      # 4 plates, 4 platemaps, all compounds have L1000 (--suggest 4)
GENES=${GENES:-"CCNA2 CELLCYCLE CDKN1A DDIT4 ATF6 GLRX"}
RUNS=${RUNS:-10}
ARMS=${ARMS:-"varclust pycytominer dcsis sis"}    # see preselect.py
NP=$(echo $PLATES | wc -w | tr -d ' ')
CACHE=cdrp_${NP}plates.pkl
mkdir -p logs results cache data

echo "== $(date)  extract $NP plates: $PLATES"
if [ ! -f cache/$CACHE ]; then
  $PY cdrp_extract.py --plates $PLATES --download --delete-sqlite \
      --out cache/$CACHE --no-diagnostics || { echo "extraction failed"; exit 1; }
else
  echo "   cache/$CACHE exists, skipping extraction"
fi

for ARM in $ARMS; do
  echo; echo "== $(date)  arm=$ARM  runs=$RUNS  genes=$GENES"
  $PY run_panel.py --cache $CACHE --select $ARM --runs $RUNS $GENES
done

echo; echo "== $(date)  timings"
$PY compute_plan.py --host mac --k 262 --genes $(echo $GENES | wc -w | tr -d ' ') --runs $RUNS
