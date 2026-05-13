"""Fitness-over-time graphs for GA (and comparison with grid/random search)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = "results/20260509T192016"
plt.style.use("seaborn-v0_8-whitegrid")

results = pd.read_csv(f"{OUT}/results.csv").replace([np.inf, -np.inf], np.nan)
benchmarks = pd.read_csv(f"{OUT}/benchmarks.csv").replace([np.inf, -np.inf], np.nan)

CIRCUITS = ["GHZ", "W-state", "Random Clifford T", "QFT"]
ga = results[results["method"] == "GA"]

# ── Graph A: GA convergence per circuit ──────────────────────────────────────
# max and mean fitness per generation, averaged across (n_qubits, seed)
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle("GA Convergence by Circuit (avg across n_qubits & seeds)", fontsize=13)

for ax, circuit in zip(axes, CIRCUITS):
    sub = ga[ga["circuit"] == circuit].dropna(subset=["fitness"])
    # per (n_qubits, seed, epoch): max and mean fitness in that generation
    gen_stats = sub.groupby(["n_qubits", "seed", "epoch"])["fitness"].agg(["max", "mean"])
    # average those across (n_qubits, seed)
    avg = gen_stats.groupby("epoch").mean()
    epochs = avg.index

    ax.plot(epochs, avg["max"], label="max", marker="o", markersize=4)
    ax.plot(epochs, avg["mean"], label="mean", marker="s", markersize=4, linestyle="--")
    ax.set_title(circuit, fontsize=10)
    ax.set_xlabel("Generation")
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Fitness")
axes[-1].legend(fontsize=9)
plt.tight_layout()
fname = f"{OUT}/time_A_ga_convergence_per_circuit.png"
plt.savefig(fname, dpi=150); print(fname); plt.close()

# ── Graph B: GA best-so-far (cumulative max) per circuit ─────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle("GA Best-So-Far Fitness per Circuit (avg across n_qubits & seeds)", fontsize=13)

for ax, circuit in zip(axes, CIRCUITS):
    sub = ga[ga["circuit"] == circuit].dropna(subset=["fitness"])
    # best fitness found up to each epoch, per (n_qubits, seed)
    run_max = (
        sub.groupby(["n_qubits", "seed", "epoch"])["fitness"]
        .max()
        .groupby(level=["n_qubits", "seed"])
        .cummax()
        .reset_index()
    )
    avg = run_max.groupby("epoch")["fitness"].mean()
    std = run_max.groupby("epoch")["fitness"].std()
    epochs = avg.index

    ax.plot(epochs, avg, marker="o", markersize=4, label="mean best-so-far")
    ax.fill_between(epochs, avg - std, (avg + std).clip(upper=1.0), alpha=0.2)
    ax.set_title(circuit, fontsize=10)
    ax.set_xlabel("Generation")
    ax.set_ylim(0, 1.05)

axes[0].set_ylabel("Best Fitness Found")
axes[0].legend(fontsize=9)
plt.tight_layout()
fname = f"{OUT}/time_B_ga_best_so_far.png"
plt.savefig(fname, dpi=150); print(fname); plt.close()

# ── Graph C: Budget curve — all 3 methods, evals on x-axis ──────────────────
# GA: best-so-far at each eval count (pop=40, so epoch k = 40*(k+1) evals)
# grid_search / random_search: horizontal line at their final best (400 evals)
# One subplot per circuit, averaged across (n_qubits, seed)

bm_opt = benchmarks[benchmarks["method"] != "defaults"].dropna(subset=["fitness"])

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle("Search Budget Curve: Best Fitness vs Evaluations (avg across n_qubits & seeds)", fontsize=13)

POP = 40

for ax, circuit in zip(axes, CIRCUITS):
    # GA curve
    sub = ga[ga["circuit"] == circuit].dropna(subset=["fitness"])
    run_max = (
        sub.groupby(["n_qubits", "seed", "epoch"])["fitness"]
        .max()
        .groupby(level=["n_qubits", "seed"])
        .cummax()
        .reset_index()
    )
    run_max["evals"] = (run_max["epoch"] + 1) * POP
    avg_ga = run_max.groupby("evals")["fitness"].mean()
    ax.plot(avg_ga.index, avg_ga.values, marker="o", markersize=4, label="GA")

    # grid_search and random_search horizontal lines
    for method, color, ls in [("grid_search", "C1", "--"), ("random_search", "C2", ":")]:
        mean_fit = (
            bm_opt[(bm_opt["method"] == method) & (bm_opt["circuit"] == circuit)]
            ["fitness"].mean()
        )
        if not np.isnan(mean_fit):
            ax.axhline(mean_fit, color=color, linestyle=ls, linewidth=1.5, label=method)

    ax.set_title(circuit, fontsize=10)
    ax.set_xlabel("Evaluations")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 420)

axes[0].set_ylabel("Best Fitness Found")
axes[0].legend(fontsize=8)
plt.tight_layout()
fname = f"{OUT}/time_C_budget_curve.png"
plt.savefig(fname, dpi=150); print(fname); plt.close()
