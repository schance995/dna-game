#!/usr/bin/env bash
# ×8 smoke test launcher (seed 0, GA then grid)
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
OUT=results/20260622_x8_smoke
mkdir -p "$OUT"
exec .venv/bin/python x8_smoke.py "$@" 2>&1 | tee "$OUT/x8_smoke.log"
