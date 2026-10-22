#!/bin/bash
# ×6 / ×8: phase1 seeds 0–2, then health-gated expand to seeds 3–9 (N=10).
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
export PYTHONWARNINGS=ignore
export DNA_MAX_WORKERS="${DNA_MAX_WORKERS:-80}"

LEVELS="6 8"
OUTDIR=results/20260726_x6_x8
mkdir -p "$OUTDIR"

run_batch() {
  local seeds="$1"
  local tag="$2"
  echo "=== [$tag] singletons seeds=$seeds $(date) ===" | tee -a "$OUTDIR/run.log"
  $PY singletons_mirror.py --seeds $seeds --levels $LEVELS \
    --out "$OUTDIR/singletons_raw.csv" \
    --done-marker "$OUTDIR/singletons_${tag}_done.marker" \
    2>&1 | tee -a "$OUTDIR/singletons.log"

  echo "=== [$tag] grid seeds=$seeds $(date) ===" | tee -a "$OUTDIR/run.log"
  $PY brute_force_mirror.py --seeds $seeds --levels $LEVELS --methods grid_search \
    --out "$OUTDIR/brute_force_raw.csv" \
    --done-marker "$OUTDIR/grid_${tag}_done.marker" \
    2>&1 | tee -a "$OUTDIR/grid.log"

  echo "=== [$tag] GA seeds=$seeds $(date) ===" | tee -a "$OUTDIR/run.log"
  $PY multi_seed_sweep.py --seeds $seeds --levels $LEVELS \
    --out "$OUTDIR/multi_seed_raw.csv" \
    --curves-out "$OUTDIR/multi_seed_curves.csv" \
    --done-marker "$OUTDIR/ga_${tag}_done.marker" \
    2>&1 | tee -a "$OUTDIR/ga.log"
}

compute_looks_good() {
  local nproc_n
  nproc_n=$(nproc)
  local ga_n=$(($(wc -l < "$OUTDIR/multi_seed_raw.csv" 2>/dev/null || echo 1) - 1))
  local grid_n=$(($(wc -l < "$OUTDIR/brute_force_raw.csv" 2>/dev/null || echo 1) - 1))
  local workers="${DNA_MAX_WORKERS:-0}"

  echo "health check: nproc=$nproc_n workers=$workers ga_rows=$ga_n grid_rows=$grid_n" | tee -a "$OUTDIR/run.log"

  if [ "$nproc_n" -lt 64 ]; then
    echo "BAD: nproc=$nproc_n < 64 (not on big box)" | tee -a "$OUTDIR/run.log"
    return 1
  fi
  if [ "$workers" -lt 40 ]; then
    echo "BAD: DNA_MAX_WORKERS=$workers < 40" | tee -a "$OUTDIR/run.log"
    return 1
  fi
  if [ "$ga_n" -lt 6 ] || [ "$grid_n" -lt 6 ]; then
    echo "BAD: incomplete phase1 (need ga>=6 grid>=6)" | tee -a "$OUTDIR/run.log"
    return 1
  fi
  # If any GA ×8 took > 14h wall, box may not be helping enough — still expand but warn.
  if [ -f "$OUTDIR/multi_seed_raw.csv" ]; then
    local slow
    slow=$($PY - <<'PY'
import csv
rows=list(csv.DictReader(open("results/20260726_x6_x8/multi_seed_raw.csv")))
xs=[float(r["t_ga_s"]) for r in rows if float(r["difficulty"])==8.0]
print(max(xs) if xs else 0)
PY
)
    if awk "BEGIN{exit !($slow > 50400)}"; then
      echo "WARN: slowest ×8 GA was ${slow}s (>14h) — expanding anyway" | tee -a "$OUTDIR/run.log"
    fi
  fi
  echo "GOOD: compute looks fine — expanding to seeds 3–9" | tee -a "$OUTDIR/run.log"
  return 0
}

echo "=== x6/x8 workers=$DNA_MAX_WORKERS $(date) ===" | tee -a "$OUTDIR/run.log"

run_batch "0 1 2" "phase1"
echo "PHASE1 DONE $(date)" | tee "$OUTDIR/phase1_done.marker"

if compute_looks_good; then
  echo "ok" > "$OUTDIR/health_ok.marker"
  run_batch "3 4 5 6 7 8 9" "phase2"
  echo "X6_X8 N10 DONE $(date)" | tee "$OUTDIR/x6_x8_done.marker"
else
  echo "bad" > "$OUTDIR/health_bad.marker"
  echo "X6_X8 PHASE1 ONLY $(date) — expand skipped" | tee "$OUTDIR/x6_x8_done.marker"
fi
