#!/bin/bash
# Seeds 10-19 at x0.5-x4 (brings N to 20). Three parallel streams, each writing its own CSV.
# Resumable: re-run after a crash/reboot and finished (seed, level) pairs are skipped.
set -uo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
export PYTHONWARNINGS=ignore
LEVELS="0.5 1 2 4"
OUTDIR=results/20261021_seeds_10-19
mkdir -p "$OUTDIR"

ga_stream() {
  local seeds="$1" tag="$2"
  DNA_MAX_WORKERS=40 $PY multi_seed_sweep.py --seeds $seeds --levels $LEVELS \
    --out "$OUTDIR/multi_seed_raw_$tag.csv" \
    --curves-out "$OUTDIR/multi_seed_curves_$tag.csv" \
    --done-marker "$OUTDIR/ga_${tag}_done.marker" \
    >> "$OUTDIR/ga_$tag.log" 2>&1
}

baseline_stream() {
  local seeds="10 11 12 13 14 15 16 17 18 19"
  DNA_MAX_WORKERS=16 $PY singletons_mirror.py --seeds $seeds --levels $LEVELS \
    --out "$OUTDIR/singletons_raw.csv" \
    --done-marker "$OUTDIR/singletons_done.marker" \
    >> "$OUTDIR/singletons.log" 2>&1
  DNA_MAX_WORKERS=16 $PY brute_force_mirror.py --seeds $seeds --levels $LEVELS --methods grid_search \
    --out "$OUTDIR/brute_force_raw.csv" \
    --done-marker "$OUTDIR/grid_done.marker" \
    >> "$OUTDIR/grid.log" 2>&1
}

echo "start $(date)" >> "$OUTDIR/run.log"
ga_stream "10 11 12 13 14" "10-14" &
ga_stream "15 16 17 18 19" "15-19" &
baseline_stream &
wait
echo "SEEDS 10-19 DONE $(date)" | tee -a "$OUTDIR/run.log" > "$OUTDIR/all_done.marker"
