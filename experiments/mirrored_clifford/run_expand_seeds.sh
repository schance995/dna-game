#!/bin/bash
# Expand baselines to seeds 3-9 (grid + singletons only, no random search).
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
export PYTHONWARNINGS=ignore
SEEDS="3 4 5 6 7 8 9"
LEVELS="0.5 1 2 4"
LOG_DIR=results/20260623_expand_seeds
mkdir -p "$LOG_DIR"

echo "=== expand seeds: singletons $(date) ===" | tee -a "$LOG_DIR/expand.log"
$PY singletons_mirror.py --seeds $SEEDS --levels $LEVELS \
  --out singletons_raw.csv \
  --done-marker "$LOG_DIR/singletons_done.marker" \
  2>&1 | tee -a "$LOG_DIR/singletons.log"

echo "=== expand seeds: grid search $(date) ===" | tee -a "$LOG_DIR/expand.log"
$PY brute_force_mirror.py --seeds $SEEDS --levels $LEVELS --methods grid_search \
  --out brute_force_raw.csv \
  --done-marker "$LOG_DIR/grid_done.marker" \
  2>&1 | tee -a "$LOG_DIR/grid.log"

echo "EXPAND SEEDS DONE $(date)" | tee "$LOG_DIR/expand_done.marker"
