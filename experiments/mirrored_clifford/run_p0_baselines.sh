#!/bin/bash
# P0 baselines: singletons (fast) then grid + random search (slow).
# Idempotent: both scripts skip completed (seed, level, method) rows.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
export PYTHONWARNINGS=ignore
SEEDS="0 1 2"
LEVELS="0.5 1 2 4"

echo "=== P0: singletons $(date) ===" | tee -a p0_baselines.log
$PY singletons_mirror.py --seeds $SEEDS --levels $LEVELS 2>&1 | tee -a p0_singletons.log

echo "=== P0: brute force $(date) ===" | tee -a p0_baselines.log
$PY brute_force_mirror.py --seeds $SEEDS --levels $LEVELS 2>&1 | tee -a p0_brute_force.log

echo "ALL P0 DONE $(date)" | tee p0_done.marker
