"""Paper figures: circuit difficulty, search budget curve, composition vs single."""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <results_dir>")
    sys.exit(1)

OUT = sys.argv[1]
plt.style.use("seaborn-v0_8-whitegrid")

# --- Load & deduplicate ---
results = pd.read_csv(f"{OUT}/results.csv").replace([np.inf, -np.inf], np.nan)
benchmarks = pd.read_csv(f"{OUT}/benchmarks.csv").replace([np.inf, -np.inf], np.nan)

CIRCUITS = ["GHZ", "W-state", "Random Clifford T", "QFT"]
OPT_METHODS = ["GA", "grid_search", "random_search"]

# Deduplicate defaults (3 duplicate runs per condition)
defaults = (
    benchmarks[benchmarks["method"] == "defaults"]
    .groupby(["sequence", "circuit", "n_qubits", "seed"])["fitness"]
    .mean()
    .reset_index()
)
optimized = benchmarks[benchmarks["method"] != "defaults"]
bm = pd.concat([defaults, optimized]).reset_index(drop=True)

# Exclude broken conditions
condition_max = bm.groupby(["circuit", "n_qubits", "seed"])["fitness"].max()
broken = condition_max[condition_max.fillna(0) == 0].index
bm = bm[~bm.set_index(["circuit", "n_qubits", "seed"]).index.isin(broken)].reset_index(drop=True)

best = (
    bm[bm["method"] != "defaults"]
    .groupby(["method", "circuit", "n_qubits", "seed"])["fitness"]
    .max()
    .reset_index()
)
best_default = (
    defaults.groupby(["circuit", "n_qubits", "seed"])["fitness"].max().reset_index()
)
best_default["method"] = "best_default"

ga = results[results["method"] == "GA"]
n_qubits_vals = sorted(bm["n_qubits"].unique())
POP = 40

# ── Figure 1: Circuit Difficulty (box & whisker) ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Circuit Difficulty: Fitness Distribution by Method per Circuit", fontsize=13)

plot_data = best.dropna(subset=["fitness"])
positions = []
labels = []
box_data = []
colors = plt.cm.tab10(np.linspace(0, 1, len(OPT_METHODS)))

group_width = len(OPT_METHODS) + 1
for ci, circuit in enumerate(CIRCUITS):
    for mi, method in enumerate(OPT_METHODS):
        vals = plot_data[(plot_data["circuit"] == circuit) & (plot_data["method"] == method)]["fitness"].values
        box_data.append(vals)
        positions.append(ci * group_width + mi)

bp = ax.boxplot(box_data, positions=positions, widths=0.7, patch_artist=True, showfliers=True)
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(colors[i % len(OPT_METHODS)])

# Legend and labels
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(len(OPT_METHODS))]
ax.legend(handles, OPT_METHODS, fontsize=9)
ax.set_xticks([ci * group_width + 1 for ci in range(len(CIRCUITS))])
ax.set_xticklabels(CIRCUITS)
ax.set_ylabel("Fitness")
ax.set_ylim(0, 1.05)
plt.tight_layout()
fname = f"{OUT}/fig1_circuit_difficulty.png"
plt.savefig(fname, dpi=150); print(fname); plt.close()

# ── Figure 2: Search Budget Curve (one per n_qubits, with variance) ──────────
fig, axes = plt.subplots(1, len(n_qubits_vals), figsize=(7 * len(n_qubits_vals), 5), sharey=True)
fig.suptitle("Search Budget Curve: Best Fitness vs Evaluations", fontsize=13)
if len(n_qubits_vals) == 1:
    axes = [axes]

bm_opt = benchmarks[benchmarks["method"] != "defaults"].dropna(subset=["fitness"])

for ax, nq in zip(axes, n_qubits_vals):
    for circuit in CIRCUITS:
        # GA curve with variance across seeds
        sub = ga[(ga["circuit"] == circuit) & (ga["n_qubits"] == nq)].dropna(subset=["fitness"])
        if sub.empty:
            continue
        run_max = (
            sub.groupby(["seed", "epoch"])["fitness"]
            .max()
            .groupby(level="seed")
            .cummax()
            .reset_index()
        )
        run_max["evals"] = (run_max["epoch"] + 1) * POP
        avg = run_max.groupby("evals")["fitness"].mean()
        std = run_max.groupby("evals")["fitness"].std().fillna(0)
        ax.plot(avg.index, avg.values, marker="o", markersize=3, label=f"{circuit}")
        ax.fill_between(avg.index, (avg - std).clip(lower=0), (avg + std).clip(upper=1.0), alpha=0.15)

    # Grid/random baselines as horizontal bands
    for method, ls in [("grid_search", "--"), ("random_search", ":")]:
        sub = bm_opt[(bm_opt["method"] == method) & (bm_opt["n_qubits"] == nq)]
        for circuit in CIRCUITS:
            csub = sub[sub["circuit"] == circuit]["fitness"]
            if not csub.empty:
                ax.axhline(csub.mean(), linestyle=ls, linewidth=1, alpha=0.4)

    ax.set_title(f"n_qubits={nq}", fontsize=10)
    ax.set_xlabel("Evaluations")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 420)

axes[0].set_ylabel("Best Fitness Found")
axes[0].legend(fontsize=8)
plt.tight_layout()
fname = f"{OUT}/fig2_budget_curve.png"
plt.savefig(fname, dpi=150); print(fname); plt.close()

# ── Figure 3: Composition vs Single Technique ────────────────────────────────
all_methods = pd.concat([best, best_default]).dropna(subset=["fitness"]).reset_index(drop=True)
plot_methods = ["best_default"] + OPT_METHODS
width = 0.2
x = np.arange(len(CIRCUITS))

fig, axes = plt.subplots(1, len(n_qubits_vals), figsize=(6 * len(n_qubits_vals), 5), sharey=True)
fig.suptitle("Composition vs Single Technique: Mean Fitness Across Seeds", fontsize=13)
if len(n_qubits_vals) == 1:
    axes = [axes]

for ax, nq in zip(axes, n_qubits_vals):
    sub = all_methods[all_methods["n_qubits"] == nq]
    for i, method in enumerate(plot_methods):
        vals = [
            sub[(sub["method"] == method) & (sub["circuit"] == c)]["fitness"].mean()
            for c in CIRCUITS
        ]
        errs = [
            sub[(sub["method"] == method) & (sub["circuit"] == c)]["fitness"].std()
            for c in CIRCUITS
        ]
        ax.bar(x + i * width, vals, width, yerr=errs, capsize=3, label=method)
    ax.set_xticks(x + width * (len(plot_methods) - 1) / 2)
    ax.set_xticklabels(CIRCUITS, rotation=15, ha="right", fontsize=9)
    ax.set_title(f"n_qubits={nq}", fontsize=10)
    ax.set_ylabel("Mean Fitness")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

plt.tight_layout()
fname = f"{OUT}/fig3_composition_vs_single.png"
plt.savefig(fname, dpi=150); print(fname); plt.close()
