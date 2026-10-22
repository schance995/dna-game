#!/bin/bash
# GA sweep for seeds 3-9 (mirrored Clifford+T), writes to a separate CSV
# so the seeds 0-2 file (old schema) is untouched. Resumable.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
export PYTHONWARNINGS=ignore
SEEDS="3 4 5 6 7 8 9"
LEVELS="0.5 1 2 4"
OUTDIR=results/20260706_ga_expand
mkdir -p "$OUTDIR"

$PY multi_seed_sweep.py --seeds $SEEDS --levels $LEVELS \
  --out "$OUTDIR/multi_seed_raw_3-9.csv" \
  --curves-out "$OUTDIR/multi_seed_curves_3-9.csv" \
  --done-marker "$OUTDIR/ga_expand_done.marker" \
  2>&1 | tee -a "$OUTDIR/ga_expand.log"
