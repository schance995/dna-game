#!/bin/bash
# Resume the multi-seed sweep after the Jun 11 reboot killed it mid seed-0/x4.
# Completed before the crash (preserved as part1_*): seed 0, levels 0.5/1/2.
cd ~/dna-game/new

mv multi_seed_raw.csv part1_raw.csv 2>/dev/null
mv multi_seed_curves.csv part1_curves.csv 2>/dev/null
mv multi_seed.log part1.log 2>/dev/null

PYTHONWARNINGS=ignore .venv/bin/python multi_seed_sweep.py \
  --seeds 0 --levels 4 --pop 40 --epochs 20 > part2.log 2>&1
mv multi_seed_raw.csv part2_raw.csv 2>/dev/null
mv multi_seed_curves.csv part2_curves.csv 2>/dev/null

PYTHONWARNINGS=ignore .venv/bin/python multi_seed_sweep.py \
  --seeds 1 2 --levels 0.5 1 2 4 --pop 40 --epochs 20 > part3.log 2>&1

echo "ALL DONE $(date)" > resume_done.marker
