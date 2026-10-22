"""Plot multi-seed mirrored Clifford+T sweep results."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results" / "20260613_multi_seed"
OUT = RESULTS

raw = list(csv.DictReader(open(RESULTS / "multi_seed_raw.csv")))
summary = list(csv.DictReader(open(RESULTS / "multi_seed_summary.csv")))
curves = list(csv.DictReader(open(RESULTS / "multi_seed_curves.csv")))

levels = [0.5, 1.0, 2.0, 4.0]
labels = [f"×{k:g}" for k in levels]
seeds = sorted({int(r["seed"]) for r in raw})

# --- Figure 1: three metrics with error bars + per-seed dots ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
metrics = [
    ("best_fitness", "Best fitness (higher = better)", False),
    ("gens_to_converge", "Generations to converge", False),
    ("ratio_raw", "Error ratio on re-run (lower = better)", True),
]
for ax, (key, ylabel, ref1) in zip(axes, metrics):
    mu = [float(r[f"{key}_mean"]) for r in summary]
    sd = [float(r[f"{key}_std"]) for r in summary]
    ax.errorbar(levels, mu, yerr=sd, fmt="o-", capsize=5, color="tab:blue",
                label="mean ± std", zorder=3)
    for s in seeds:
        pts = [float(r[key]) for r in raw if int(r["seed"]) == s]
        ax.scatter(levels, pts, s=40, alpha=0.7, zorder=4)
    if ref1:
        ax.axhline(1.0, color="tab:red", ls="--", lw=1, label="no mitigation")
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, labels)
    ax.set_xlabel("Difficulty (× n² gates, mirrored)")
    ax.set_ylabel(ylabel)
    if ref1:
        ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Multi-seed mirrored Clifford+T — n=7, pop=40, 20 gen, seeds 0–2")
fig.tight_layout()
fig.savefig(OUT / "fig_metrics_errorbars.png", dpi=150)
plt.close(fig)

# --- Figure 2: convergence curves (max fitness per generation) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
for ax, k in zip(axes.flat, levels):
    for s, c in zip(seeds, colors):
        rows = sorted(
            [r for r in curves if int(r["seed"]) == s and float(r["difficulty"]) == k],
            key=lambda r: int(r["gen"]),
        )
        gens = [int(r["gen"]) for r in rows]
        mx = [float(r["max_fit"]) for r in rows]
        ax.plot(gens, mx, "-", color=c, alpha=0.85, label=f"seed {s}")
    ax.set_title(f"Difficulty ×{k:g}")
    ax.set_ylabel("Max fitness")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("Generation")
axes[0, 0].legend(fontsize=8)
fig.suptitle("GA max-fitness curves per seed (mirrored Clifford+T)")
fig.tight_layout()
fig.savefig(OUT / "fig_convergence_curves.png", dpi=150)
plt.close(fig)

# --- Figure 3: ideal / noisy / mitigated sanity check ---
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.25
x = np.arange(len(levels))
for i, (col, label, color) in enumerate([
    ("ideal", "ideal", "tab:green"),
    ("noisy", "noisy", "tab:orange"),
    ("mitigated_final_best", "mitigated (re-run)", "tab:purple"),
]):
    vals = []
    for k in levels:
        sub = [float(r[col]) for r in raw if float(r["difficulty"]) == k]
        vals.append(np.mean(sub))
    ax.bar(x + (i - 1) * w, vals, w, label=label, color=color)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_ylabel("⟨Z⊗7⟩ expectation")
ax.set_title("Raw values sanity check (mean across seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_raw_values.png", dpi=150)
plt.close(fig)

print(f"Saved plots to {OUT}/")
