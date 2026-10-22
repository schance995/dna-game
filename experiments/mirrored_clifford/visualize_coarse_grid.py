"""Compare coarse vs fine grid search on mirrored Clifford+T."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent / "results"
COARSE = ROOT / "20260622_coarse_grid" / "coarse_grid_raw.csv"
FINE = ROOT / "20260616_p0_baselines" / "brute_force_raw.csv"
if not FINE.exists():
    FINE = Path(__file__).parent / "brute_force_raw.csv"
OUT = ROOT / "20260622_coarse_grid" / "fig_coarse_vs_fine_grid.png"

levels = [0.5, 1.0, 2.0, 4.0]
labels = [f"×{k:g}" for k in levels]

coarse = list(csv.DictReader(open(COARSE)))
fine = [r for r in csv.DictReader(open(FINE)) if r["method"] == "grid_search"]

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, rows, color, marker in [
    ("Fine grid (288 evals)", fine, "tab:green", "s"),
    ("Coarse grid (72 evals)", coarse, "tab:orange", "^"),
]:
    mu, sd = [], []
    for k in levels:
        vals = [float(r["ratio_raw"]) for r in rows if float(r["difficulty"]) == k]
        mu.append(np.mean(vals))
        sd.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    ax.errorbar(levels, mu, yerr=sd, fmt=f"{marker}-", capsize=4, color=color,
                label=name, ms=7)
    for k in levels:
        pts = [float(r["ratio_raw"]) for r in rows if float(r["difficulty"]) == k]
        ax.scatter([k] * len(pts), pts, s=30, color=color, alpha=0.45, zorder=4)

ax.axhline(1.0, color="tab:red", ls="--", lw=1, label="no mitigation")
ax.set_xscale("log", base=2)
ax.set_xticks(levels, labels)
ax.set_xlabel("Difficulty (mirrored Clifford+T)")
ax.set_ylabel("Error ratio on re-run (lower = better)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.set_title("Coarse vs fine grid — seeds 0–2, same ZNE/DDD presets")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
plt.close(fig)
print(f"Saved {OUT}")
