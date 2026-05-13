"""Generate diagnostic graphs for DNA game benchmark results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUT = "results/20260509T192016"
plt.style.use("seaborn-v0_8-whitegrid")

# --- Load & deduplicate ---
df = pd.read_csv(f"{OUT}/benchmarks.csv")
df = df.replace([np.inf, -np.inf], np.nan)

# Average the 3 duplicate default runs per (sequence, circuit, n_qubits, seed)
defaults = (
    df[df["method"] == "defaults"]
    .groupby(["sequence", "circuit", "n_qubits", "seed"])["fitness"]
    .mean()
    .reset_index()
)
optimized = df[df["method"] != "defaults"]
bm = pd.concat([defaults, optimized]).reset_index(drop=True)

# Exclude broken conditions: any (circuit, n_qubits, seed) where all fitness values are 0 or NaN
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

CIRCUITS = ["GHZ", "W-state", "Random Clifford T", "QFT"]
OPT_METHODS = ["GA", "grid_search", "random_search"]

# ── Graph 1: Ceiling Effect ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
fig.suptitle("Ceiling Effect: Fitness by Method per Circuit", fontsize=13)

plot_data = best.dropna(subset=["fitness"])

for ax, circuit in zip(axes, CIRCUITS):
    sub = plot_data[plot_data["circuit"] == circuit]
    for i, method in enumerate(OPT_METHODS):
        vals = sub[sub["method"] == method]["fitness"].values
        x = np.full(len(vals), i) + np.random.uniform(-0.15, 0.15, len(vals))
        ax.scatter(x, vals, alpha=0.7, s=30)
    ax.set_xticks(range(len(OPT_METHODS)))
    ax.set_xticklabels(OPT_METHODS, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(circuit, fontsize=10)
    ax.set_xlabel("")

axes[0].set_ylabel("Fitness")
plt.tight_layout()
fname = f"{OUT}/diag1_ceiling_effect.png"
plt.savefig(fname, dpi=150)
print(fname)
plt.close()

# ── Graph 2: Broken Conditions Heatmap ──────────────────────────────────────
# fitness=NaN means originally inf mitigation_ratio (pure ≈ noisy, uninformative)
broken = defaults.copy()
broken["broken"] = broken["fitness"].isna().astype(int)
broken["nq_seed"] = broken["n_qubits"].astype(str) + "_s" + broken["seed"].astype(str)

pivot = broken.groupby(["circuit", "nq_seed"])["broken"].max().unstack(fill_value=0)
# Ensure all circuits present as rows
pivot = pivot.reindex(CIRCUITS, fill_value=0)

fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.8 + 2), 4))
cmap = mcolors.ListedColormap(["white", "red"])
im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title("Conditions where pure ≈ noisy (fitness=0, uninformative)", fontsize=11)
plt.colorbar(im, ax=ax, ticks=[0, 1], label="broken")
plt.tight_layout()
fname = f"{OUT}/diag2_broken_conditions.png"
plt.savefig(fname, dpi=150)
print(fname)
plt.close()

# ── Graph 3: Seed Variance ───────────────────────────────────────────────────
TARGET_CIRCUITS = ["GHZ", "Random Clifford T"]
N_QUBITS = 5
seeds = [0, 1, 2]
x = np.arange(len(seeds))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle("Seed Variance: Method Rankings Across Seeds (n=5)", fontsize=13)

for ax, circuit in zip(axes, TARGET_CIRCUITS):
    sub = best[(best["circuit"] == circuit) & (best["n_qubits"] == N_QUBITS)].dropna(
        subset=["fitness"]
    )
    for i, method in enumerate(OPT_METHODS):
        vals = [
            sub[(sub["method"] == method) & (sub["seed"] == s)]["fitness"].mean()
            for s in seeds
        ]
        ax.bar(x + i * width, vals, width, label=method)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_title(circuit, fontsize=10)
    ax.set_ylabel("Fitness")
    ax.legend(fontsize=8)

plt.tight_layout()
fname = f"{OUT}/diag3_seed_variance.png"
plt.savefig(fname, dpi=150)
print(fname)
plt.close()

# ── Graph 4: Composition vs Single Technique ────────────────────────────────
all_methods_data = pd.concat([best, best_default]).reset_index(drop=True)
all_methods_data = all_methods_data.dropna(subset=["fitness"])

n_qubits_vals = sorted(all_methods_data["n_qubits"].unique())
plot_methods = ["best_default"] + OPT_METHODS
width = 0.2
x = np.arange(len(CIRCUITS))

fig, axes = plt.subplots(1, len(n_qubits_vals), figsize=(6 * len(n_qubits_vals), 5), sharey=True)
fig.suptitle("Composition vs Single Technique: Mean Fitness Across Seeds", fontsize=13)

if len(n_qubits_vals) == 1:
    axes = [axes]

for ax, nq in zip(axes, n_qubits_vals):
    sub = all_methods_data[all_methods_data["n_qubits"] == nq]
    for i, method in enumerate(plot_methods):
        vals = [
            sub[(sub["method"] == method) & (sub["circuit"] == c)]["fitness"].mean()
            for c in CIRCUITS
        ]
        ax.bar(x + i * width, vals, width, label=method)
    ax.set_xticks(x + width * (len(plot_methods) - 1) / 2)
    ax.set_xticklabels(CIRCUITS, rotation=15, ha="right", fontsize=9)
    ax.set_title(f"n_qubits={nq}", fontsize=10)
    ax.set_ylabel("Mean Fitness")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

plt.tight_layout()
fname = f"{OUT}/diag4_composition_vs_single.png"
plt.savefig(fname, dpi=150)
print(fname)
plt.close()
