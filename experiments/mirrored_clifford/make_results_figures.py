from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_20260923 import holm, paired

D = Path("data")
OUT = Path("paper_figures")
OUT.mkdir(exist_ok=True)
LEVELS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
METHODS = ["REM", "ZNE", "DDD", "grid", "GA"]
COLORS = {"REM": "#c7c7c7", "ZNE": "#bcbd22", "DDD": "#17becf", "grid": "#1f77b4", "GA": "#d62728"}
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def lvl(x):
    return f"×{x:g}"


def fig_ratio(runs):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    w = 0.15
    for j, m in enumerate(METHODS):
        data = [runs[(runs.multiplier == L) & (runs.method == m)].ratio_raw.dropna() for L in LEVELS]
        pos = np.arange(len(LEVELS)) + (j - 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                        medianprops={"color": "k", "lw": 1})
        for b in bp["boxes"]:
            b.set_facecolor(COLORS[m])
        for p, v in zip(pos, data):
            ax.scatter(np.full(len(v), p) + rng.uniform(-w * 0.25, w * 0.25, len(v)), v, s=3, color="k", alpha=0.5, zorder=3)
        ax.plot([], [], "s", color=COLORS[m], label=m)
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(range(len(LEVELS)), [lvl(L) for L in LEVELS])
    ax.set_xlabel("depth multiplier k")
    ax.set_ylabel("error ratio (log; <1 = improvement)")
    ax.legend(ncol=5, loc="upper left", frameon=False)
    save(fig, "fig_results_ratio_by_multiplier")


def fig_paired(runs):
    wide = runs.pivot_table(index=["multiplier", "seed"], columns="method", values="ratio_raw").reset_index()
    stats = [paired(wide[wide.multiplier == L].GA, wide[wide.multiplier == L].grid) for L in LEVELS]
    adj = holm([s["wilcoxon_p"] for s in stats])
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, (L, s, p) in enumerate(zip(LEVELS, stats, adj)):
        sub = wide[wide.multiplier == L]
        m = (sub.GA / sub.grid).to_numpy()
        m = m[np.isfinite(m)]
        ax.scatter(np.full(len(m), i) + rng.uniform(-0.1, 0.1, len(m)), m, s=6, color="k", alpha=0.6, zorder=3)
        ax.errorbar(i + 0.28, s["median_multiplier"], yerr=[[s["median_multiplier"] - s["ci_lo"]], [s["ci_hi"] - s["median_multiplier"]]],
                    fmt="D", ms=3, color=COLORS["GA"], capsize=2, lw=1)
        ax.text(i, 150, f"{s['a_wins']}/{s['n']}\np={p:.2f}", ha="center", fontsize=6)
    ax.axhline(1, color="k", lw=0.7)
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 1e3)
    ax.set_xticks(range(len(LEVELS)), [lvl(L) for L in LEVELS])
    ax.set_xlabel("depth multiplier k")
    ax.set_ylabel("GA ratio ÷ grid ratio (per seed)")
    save(fig, "fig_results_paired_ga_grid")


def fig_reliability(runs, retest):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for i, L in enumerate(LEVELS):
        for j, m in enumerate(["ZNE", "grid", "GA"]):
            v = runs[(runs.multiplier == L) & (runs.method == m)].mitigated.dropna().clip(-4, 4)
            ax.scatter(np.full(len(v), i + (j - 1) * 0.22) + rng.uniform(-0.04, 0.04, len(v)), v, s=5,
                       color=COLORS[m], label=m if i == 0 else None)
    ax.axhline(1, color="green", ls="--", lw=0.8, label="ideal")
    ax.axhline(0, color="k", lw=0.7, label="noisy ≈ 0")
    ax.set_xticks(range(len(LEVELS)), [lvl(L) for L in LEVELS])
    ax.set_xlabel("depth multiplier k")
    ax.set_ylabel("mitigated value (clipped ±4)")
    ax.legend(ncol=5, loc="lower center", frameon=False, fontsize=6)
    ax.set_title("(a) mitigated estimates centre on the noisy value, not the ideal")
    ax = axes[1]
    keys = retest.groupby(["method", "multiplier", "seed"]).ratio_raw
    for x, ((m, _, _), v) in enumerate(keys):
        ax.plot([x, x], [v.min(), v.max()], color=COLORS[m], lw=1.5)
        ax.scatter(np.full(len(v), x), v, s=3, color="k", zorder=3)
    ax.axhline(1, color="k", lw=0.7)
    ax.set_yscale("log")
    ax.set_xticks([5.5, 17.5], ["GA winners", "grid winners"])
    ax.set_ylabel("error ratio, 10 re-runs")
    ax.set_title("(b) same configuration, 10 re-runs")
    save(fig, "fig_results_reliability")


def fig_properties(runs):
    win = runs[runs.method.isin(["GA", "grid"])]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))
    ax = axes[0]
    data = [win[win.method == m][p].dropna() for p in ["p0", "p1"] for m in ["GA", "grid"]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "k"})
    for b, m in zip(bp["boxes"], ["GA", "grid", "GA", "grid"]):
        b.set_facecolor(COLORS[m])
    ax.set_xticks([1, 2, 3, 4], ["p0\nGA", "p0\ngrid", "p1\nGA", "p1\ngrid"])
    ax.set_ylabel("REM parameter")
    ax.set_title("(a) REM stage")
    ax = axes[1]
    kind = np.where(win.technique == "DDD", "DDD " + win.ddd_rule.fillna(""), "ZNE " + win.factory.fillna(""))
    ct = win.assign(kind=kind).groupby(["kind", "method"]).size().unstack(fill_value=0)[["GA", "grid"]]
    ct = ct.loc[ct.sum(1).sort_values().index]
    ct.plot.barh(ax=ax, color=[COLORS["GA"], COLORS["grid"]], width=0.75)
    ax.legend(title=None, frameon=False, loc="lower right")
    ax.set_xlabel("winning configurations (of 60)")
    ax.set_ylabel("")
    ax.set_title("(b) second stage")
    ax = axes[2]
    for m in ["GA", "grid"]:
        z = win[(win.method == m) & (win.technique == "ZNE")]
        ax.scatter(z.max_scale, z.ratio_raw, s=6, color=COLORS[m], label=m)
    ax.axhline(1, color="k", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("largest ZNE scale factor")
    ax.set_ylabel("error ratio")
    ax.set_title("(c) scale factor vs outcome")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_results_properties")


def main():
    runs = pd.read_csv(D / "mirrored_runs.csv")
    retest = pd.read_csv(D / "mirrored_retest.csv")
    fig_ratio(runs)
    fig_paired(runs)
    fig_reliability(runs, retest)
    fig_properties(runs)
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
