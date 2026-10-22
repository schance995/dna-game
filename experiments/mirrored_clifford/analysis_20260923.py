import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = Path("results")
MAY = Path(__file__).resolve().parents[2] / "results" / "20260509T192016"
OUT = R / "20260923_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CIRCUITS = ["GHZ", "W-state", "Random Clifford T", "QFT"]
HARD = ["Random Clifford T", "QFT"]
COND = ["circuit", "n_qubits", "seed"]
LEVELS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
COLORS = {"GA": "#d62728", "grid_search": "#1f77b4", "random_search": "#2ca02c", "best_default": "#7f7f7f",
          "REM": "#c7c7c7", "ZNE": "#bcbd22", "DDD": "#17becf", "grid": "#1f77b4", "best_singleton": "#7f7f7f"}
rng = np.random.default_rng(0)
plt.style.use("seaborn-v0_8-whitegrid")


def lvl(x):
    return f"×{x:g}"


def parse_pipeline(s):
    out = dict(p0=np.nan, p1=np.nan, technique=None, ddd_rule=None, factory=None, fold=None,
               n_scale=np.nan, min_scale=np.nan, max_scale=np.nan)
    if m := re.search(r"rem\(([-\d.]+), ([-\d.]+)\)", s):
        out["p0"], out["p1"] = float(m[1]), float(m[2])
    if m := re.search(r"ddd\((\w+)\)", s):
        out.update(technique="DDD", ddd_rule=m[1])
    if m := re.search(r"zne\((\w+)Factory\((.*)\), (fold_\w+)\)", s, re.S):
        out.update(technique="ZNE", factory=m[1], fold=m[3])
        if sf := re.search(r"\[([^\]]*)\]", m[2]):
            v = np.array(sf[1].replace(",", " ").split(), float)
            out.update(n_scale=len(v), min_scale=v.min(), max_scale=v.max())
    return out


def add_features(df, col):
    return pd.concat([df.reset_index(drop=True), pd.DataFrame([parse_pipeline(s) for s in df[col]])], axis=1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def signflip_p(d):
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = (signs * np.abs(d)).mean(1)
    return np.mean(np.abs(null) >= abs(d.mean()) - 1e-12)


def boot_ci(d, stat=np.median, b=10000):
    idx = rng.integers(0, len(d), (b, len(d)))
    return np.percentile(stat(d[idx], axis=1), [2.5, 97.5])


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    d = np.log(a[ok]) - np.log(b[ok])
    lo, hi = boot_ci(d)
    return dict(n=len(d), median_multiplier=np.exp(np.median(d)), ci_lo=np.exp(lo), ci_hi=np.exp(hi),
                a_wins=int((d < 0).sum()), wilcoxon_p=stats.wilcoxon(d).pvalue,
                sign_p=stats.binomtest(int((d < 0).sum()), len(d)).pvalue, perm_p=signflip_p(d))


# ---------------------------------------------------------------- May (20260509T192016) data
def load_may():
    res = pd.read_csv(MAY / "results.csv")
    bm = pd.read_csv(MAY / "benchmarks.csv").replace([np.inf, -np.inf], np.nan)
    defaults = bm[bm.method == "defaults"].groupby(["sequence"] + COND).fitness.mean().reset_index()
    best_def = defaults.groupby(COND).fitness.max().reset_index().assign(method="best_default")
    bench = pd.concat([bm[bm.method != "defaults"][["method"] + COND + ["fitness"]], best_def,
                       defaults.rename(columns={"sequence": "method"})], ignore_index=True)
    cmax = bench.groupby(COND).fitness.max()
    broken = cmax[cmax.fillna(0) == 0].index
    keep = lambda df: df[~df.set_index(COND).index.isin(broken)].reset_index(drop=True)
    return keep(res), keep(bench), broken


def fig_budget(res, bench):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
    for ax, c in zip(axes, CIRCUITS):
        for meth in ["GA", "grid_search", "random_search"]:
            sub = res[(res.circuit == c) & (res.method == meth)]
            if meth == "GA":
                sub = sub.sort_values(COND + ["epoch"], kind="stable")
            curves = [g.fitness.fillna(0).cummax().to_numpy() for _, g in sub.groupby(["n_qubits", "seed"], sort=False)]
            n = min(map(len, curves))
            ax.plot(np.arange(1, n + 1), np.vstack([cv[:n] for cv in curves]).mean(0), color=COLORS[meth],
                    lw=2, label=f"{meth}: best seen during search")
            ax.axhline(bench[(bench.circuit == c) & (bench.method == meth)].fitness.mean(), color=COLORS[meth],
                       ls="--", lw=1.5, label=f"{meth}: winner re-evaluated once")
        ax.axhline(bench[(bench.circuit == c) & (bench.method == "best_default")].fitness.mean(),
                   color=COLORS["best_default"], ls=":", lw=1.5, label="best default (re-evaluated)")
        ax.set_title(c)
        ax.set_xlabel("pipelines evaluated")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("fitness (mean over conditions)")
    axes[-1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Budget curve, like-for-like: solid = in-search best-so-far (all methods), dashed = one fresh re-run of the winner")
    fig.tight_layout()
    fig.savefig(OUT / "figA1_budget_curve_fair.png", dpi=150)
    plt.close(fig)


def fig_box(bench):
    methods = ["REM", "ZNE", "DDD", "best_default", "GA", "grid_search", "random_search"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
    for ax, c in zip(axes, CIRCUITS):
        data = [bench[(bench.circuit == c) & (bench.method == m)].fitness.dropna().to_numpy() for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.55)
        for i, v in enumerate(data):
            ax.scatter(np.full(len(v), i + 1) + rng.uniform(-0.15, 0.15, len(v)), v, s=12, color="k", alpha=0.6, zorder=3)
        ax.set_xticks(range(1, len(methods) + 1), methods, rotation=35, ha="right", fontsize=8)
        ax.axvline(3.5, color="k", lw=0.6)
        ax.set_title(c)
    axes[0].set_ylabel("re-evaluated fitness")
    fig.suptitle("Circuit difficulty & composition vs single technique (re-evaluated winners; each dot = one condition)")
    fig.tight_layout()
    fig.savefig(OUT / "figA2_box_by_circuit.png", dpi=150)
    plt.close(fig)


def may_stats(bench):
    wide = bench.pivot_table(index=COND, columns="method", values="fitness").reset_index()
    rows = []
    for label, sub in [(c, wide[wide.circuit == c]) for c in CIRCUITS] + [("hard (RCT+QFT)", wide[wide.circuit.isin(HARD)]),
                                                                         ("all", wide)]:
        for other in ["grid_search", "random_search", "best_default"]:
            ok = sub[["GA", other]].dropna()
            d = ok.GA - ok[other]
            p = stats.wilcoxon(d).pvalue if (d != 0).sum() > 0 else np.nan
            rows.append(dict(circuit=label, comparison=f"GA vs {other}", n=len(d), GA_mean=ok.GA.mean(),
                             other_mean=ok[other].mean(), GA_wins=int((d > 0).sum()), wilcoxon_p=p))
    blocks = wide.groupby(["circuit", "n_qubits"])[["GA", "grid_search", "random_search", "best_default"]].mean()
    for other in ["grid_search", "random_search", "best_default"]:
        ok = blocks[["GA", other]].dropna()
        d = ok.GA - ok[other]
        rows.append(dict(circuit="all, seeds averaged (distinct circuits)", comparison=f"GA vs {other}", n=len(d),
                         GA_mean=ok.GA.mean(), other_mean=ok[other].mean(), GA_wins=int((d > 0).sum()),
                         wilcoxon_p=stats.wilcoxon(d).pvalue))
    return pd.DataFrame(rows)


def may_ga_diagnostics(res):
    ga = res[res.method == "GA"]
    uniq = ga.groupby(COND + ["epoch"]).chromosome_str.nunique().groupby("epoch").mean()
    mean_by_epoch = ga.groupby(["circuit", "epoch"]).fitness.mean().unstack()
    zero = res.assign(z=res.fitness.fillna(0) == 0).groupby("method").z.mean()
    return uniq, mean_by_epoch, zero


def may_ga_winners(res, bench):
    ga = res[res.method == "GA"]
    last = ga[ga.epoch == ga.groupby(COND).epoch.transform("max")]
    win = last.loc[last.groupby(COND).fitness.idxmax(), COND + ["fitness", "chromosome_str"]]
    win = win.rename(columns={"fitness": "search_fitness"}).merge(
        bench[bench.method == "GA"][COND + ["fitness"]].rename(columns={"fitness": "reeval_fitness"}), on=COND)
    return add_features(win, "chromosome_str")


def enrichment(feat, top_q=0.9):
    rows = []
    for c in CIRCUITS:
        sub = feat[feat.circuit == c]
        top = sub[sub.fitness >= sub.fitness.quantile(top_q)]
        for col in ["technique", "ddd_rule", "factory", "fold"]:
            for val, share_all in sub[col].value_counts(normalize=True).items():
                rows.append(dict(circuit=c, feature=col, value=val, share_all=share_all,
                                 share_top10=(top[col] == val).mean() / top[col].notna().mean() if top[col].notna().any() else np.nan))
    return pd.DataFrame(rows)


def design_matrix(feat, extra):
    X = pd.DataFrame({
        "is_ZNE": (feat.technique == "ZNE").astype(float),
        "rem_p0": feat.p0, "rem_p1": feat.p1,
        "ddd_xx": (feat.ddd_rule == "xx").astype(float), "ddd_yy": (feat.ddd_rule == "yy").astype(float),
        "n_scale": feat.n_scale.fillna(0), "min_scale": feat.min_scale.fillna(0), "max_scale": feat.max_scale.fillna(0),
    })
    for f in ["Linear", "Poly", "Exp", "AdaExp"]:
        X[f"factory_{f}"] = (feat.factory == f).astype(float)
    for f in ["fold_all", "fold_gates_at_random"]:
        X[f] = (feat.fold == f).astype(float)
    return pd.concat([X, extra.reset_index(drop=True)], axis=1).fillna(0)


def logistic(X, y, cv, groups=None):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    auc = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="roc_auc")
    model.fit(X, y)
    coef = pd.Series(model[-1].coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
    return auc.mean(), auc.std(), coef


def fig_may_props(feat):
    hard = feat[feat.circuit.isin(HARD)]
    top = hard[hard.fitness >= hard.groupby("circuit").fitness.transform(lambda s: s.quantile(0.9))]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    ax = axes[0]
    data = [hard.p0.dropna(), top.p0.dropna(), hard.p1.dropna(), top.p1.dropna()]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2, 3, 4], ["p0 all", "p0 top10%", "p1 all", "p1 top10%"])
    ax.set_title("REM parameters")
    for ax, col in zip(axes[1:], ["ddd_rule", "factory", "fold"]):
        a = hard[col].value_counts(normalize=True)
        t = top[col].value_counts(normalize=True).reindex(a.index, fill_value=0)
        x = np.arange(len(a))
        ax.bar(x - 0.2, a.values, 0.4, label="all evaluations", color="#bbbbbb")
        ax.bar(x + 0.2, t.values, 0.4, label="top 10% fitness", color="#d62728")
        ax.set_xticks(x, a.index, rotation=20, fontsize=8)
        ax.set_title(col)
    axes[1].legend(fontsize=8)
    fig.suptitle("May data, Random Clifford T + QFT: what the top-10% evaluations look like (all methods, in-search fitness)")
    fig.tight_layout()
    fig.savefig(OUT / "figA3_pipeline_properties_may.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Mirrored N=10 data
def load_mirrored():
    ga = pd.concat([pd.read_csv(R / "20260613_multi_seed/multi_seed_raw.csv"),
                    pd.read_csv(R / "20260706_ga_expand/multi_seed_raw_3-9.csv"),
                    pd.read_csv(R / "20260726_x6_x8/multi_seed_raw.csv")])
    ga = ga.rename(columns={"mitigated_final_best": "mitigated", "best_pipeline": "pipeline"}).assign(method="GA")
    gr = pd.concat([pd.read_csv(R / "20260623_expand_seeds/brute_force_raw.csv"),
                    pd.read_csv(R / "20260726_x6_x8/brute_force_raw.csv")])
    gr = gr[gr.method == "grid_search"].rename(columns={"mitigated_best": "mitigated", "best_pipeline": "pipeline"}).assign(method="grid")
    sg = pd.concat([pd.read_csv(R / "20260623_expand_seeds/singletons_raw.csv"),
                    pd.read_csv(R / "20260726_x6_x8/singletons_raw.csv")])
    cols = ["seed", "difficulty", "method", "noisy", "mitigated", "ratio_raw"]
    df = pd.concat([ga[cols + ["pipeline"]], gr[cols + ["pipeline"]], sg[cols]], ignore_index=True)
    wide = df.pivot_table(index=["difficulty", "seed"], columns="method", values="ratio_raw")
    wide["best_singleton"] = wide[["REM", "ZNE", "DDD"]].min(axis=1)
    return df, wide.reset_index()


def mirrored_stats(wide):
    rows = []
    for a, b in [("GA", "grid"), ("GA", "best_singleton"), ("GA", "ZNE"), ("grid", "best_singleton")]:
        block = []
        for L in LEVELS:
            sub = wide[wide.difficulty == L]
            block.append(dict(level=lvl(L), comparison=f"{a} vs {b}", **paired(sub[a], sub[b])))
        adj = holm([r["wilcoxon_p"] for r in block])
        for r, p in zip(block, adj):
            r["wilcoxon_p_holm"] = p
        rows += block
    return pd.DataFrame(rows)


def between_levels(wide):
    d = {L: np.log(wide[wide.difficulty == L].GA) - np.log(wide[wide.difficulty == L].grid) for L in LEVELS}
    d = {L: v[np.isfinite(v)].to_numpy() for L, v in d.items()}
    pooled = np.concatenate([np.column_stack([np.full(len(v), L), v]) for L, v in d.items()])
    return pd.DataFrame([
        dict(test="Mann-Whitney ×4 vs ×6", stat=stats.mannwhitneyu(d[4.0], d[6.0]).statistic, p=stats.mannwhitneyu(d[4.0], d[6.0]).pvalue),
        dict(test="Mann-Whitney ×6 vs ×8", stat=stats.mannwhitneyu(d[6.0], d[8.0]).statistic, p=stats.mannwhitneyu(d[6.0], d[8.0]).pvalue),
        dict(test="Mann-Whitney ×6 vs all other levels", stat=stats.mannwhitneyu(d[6.0], np.concatenate([d[L] for L in LEVELS if L != 6.0])).statistic,
             p=stats.mannwhitneyu(d[6.0], np.concatenate([d[L] for L in LEVELS if L != 6.0])).pvalue),
        dict(test="Kruskal-Wallis across 6 levels", stat=stats.kruskal(*d.values()).statistic, p=stats.kruskal(*d.values()).pvalue),
        dict(test="Spearman(level, log GA/grid)", stat=stats.spearmanr(pooled[:, 0], pooled[:, 1]).statistic,
             p=stats.spearmanr(pooled[:, 0], pooled[:, 1]).pvalue),
    ])


def signal_tests(df):
    rows = []
    for (m, L), g in df.groupby(["method", "difficulty"]):
        v = g.dropna(subset=["mitigated", "ratio_raw"])
        k = int((v.ratio_raw < 1).sum())
        rows.append(dict(method=m, level=lvl(L), n=len(v), median_noisy=v.noisy.median(), median_mitigated=v.mitigated.median(),
                         sd_mitigated=v.mitigated.std(), wilcoxon_mitigated_vs_0_p=stats.wilcoxon(v.mitigated).pvalue,
                         ratio_below_1=k, binom_p_vs_half=stats.binomtest(k, len(v)).pvalue))
    return pd.DataFrame(rows)


def test_retest():
    ms = pd.read_csv(R / "20260617_multisample/multisample_raw.csv")
    ms["logr"] = np.log(ms.ratio_raw)
    rows = []
    for m, g in ms.groupby("method"):
        groups = [x.logr.to_numpy() for _, x in g.groupby(["seed", "difficulty"])]
        k = np.mean([len(x) for x in groups])
        grand = g.logr.mean()
        msb = k * sum((x.mean() - grand) ** 2 for x in groups) / (len(groups) - 1)
        msw = sum(((x - x.mean()) ** 2).sum() for x in groups) / (len(g) - len(groups))
        spread = g.groupby(["seed", "difficulty"]).ratio_raw.agg(["min", "median", "max"])
        rows.append(dict(method=m, pipelines=len(groups), reruns_each=int(k), icc=(msb - msw) / (msb + (k - 1) * msw),
                         anova_p=stats.f_oneway(*groups).pvalue, median_within_min=spread["min"].median(),
                         median_within_max=spread["max"].median(), share_reruns_below_1=(g.ratio_raw < 1).mean()))
    return ms, pd.DataFrame(rows)


def fig_paired(wide, st):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, other in zip(axes, ["grid", "best_singleton"]):
        s = st[st.comparison == f"GA vs {other}"].reset_index(drop=True)
        for i, L in enumerate(LEVELS):
            sub = wide[wide.difficulty == L]
            m = (sub.GA / sub[other]).to_numpy()
            m = m[np.isfinite(m)]
            ax.scatter(np.full(len(m), i) + rng.uniform(-0.12, 0.12, len(m)), m, color="k", s=16, alpha=0.6, zorder=3)
            r = s.iloc[i]
            ax.errorbar(i + 0.3, r.median_multiplier, yerr=[[r.median_multiplier - r.ci_lo], [r.ci_hi - r.median_multiplier]],
                        fmt="D", color=COLORS["GA"], capsize=4)
            ax.text(i, 60, f"GA wins {r.a_wins}/{r.n}\np={r.wilcoxon_p:.2f}\nHolm {r.wilcoxon_p_holm:.2f}", ha="center", fontsize=7.5)
        ax.axhline(1, color="k", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3e2)
        ax.set_xticks(range(len(LEVELS)), [lvl(L) for L in LEVELS])
        ax.set_title(f"GA ratio ÷ {other} ratio per seed (below 1 = GA better)")
        ax.set_xlabel("difficulty")
    axes[0].set_ylabel("ratio multiplier (log)")
    fig.suptitle("Paired per-seed comparison, N=10 mirrored circuits. Diamond = median with bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(OUT / "figB1_paired_ga_vs_baselines.png", dpi=150)
    plt.close(fig)


def fig_signal(df, ms):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    methods = ["ZNE", "grid", "GA"]
    for i, L in enumerate(LEVELS):
        for j, m in enumerate(methods):
            v = df[(df.difficulty == L) & (df.method == m)].mitigated.dropna()
            ax.scatter(np.full(len(v), i + (j - 1) * 0.25) + rng.uniform(-0.05, 0.05, len(v)), v, s=14,
                       color=COLORS[m], label=m if i == 0 else None, alpha=0.8)
    ax.axhline(1, color="green", ls="--", label="ideal = +1")
    ax.axhline(0, color="k", lw=1, label="noisy ≈ 0")
    ax.set_ylim(-4, 4)
    ax.set_xticks(range(len(LEVELS)), [lvl(L) for L in LEVELS])
    ax.set_ylabel("mitigated expectation value (clipped to ±4)")
    ax.set_title("Is any signal recovered? Mitigated values scatter around 0, not around the ideal +1")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    for j, (m, g) in enumerate(ms.groupby("method")):
        keys = list(g.groupby(["difficulty", "seed"]).groups)
        for i, key in enumerate(keys):
            v = g[(g.difficulty == key[0]) & (g.seed == key[1])].ratio_raw
            x = i + j * (len(keys) + 1)
            ax.plot([x, x], [v.min(), v.max()], color=COLORS["GA" if m == "GA" else "grid"], lw=2)
            ax.scatter(np.full(len(v), x), v, s=8, color="k", zorder=3)
    ax.axhline(1, color="k", lw=1)
    ax.set_yscale("log")
    ax.set_xticks([5.5, 18.5], ["GA winners", "grid winners"])
    ax.set_ylabel("ratio_raw of 10 re-runs of the SAME pipeline")
    ax.set_title("Test-retest: one pipeline, 10 re-runs (seeds 0–2, ×0.5–×4)")
    fig.tight_layout()
    fig.savefig(OUT / "figB2_signal_and_retest.png", dpi=150)
    plt.close(fig)


def fig_mirrored_props(win):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    ax = axes[0]
    data = [win[win.method == m][p].dropna() for p in ["p0", "p1"] for m in ["GA", "grid"]]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks([1, 2, 3, 4], ["p0 GA", "p0 grid", "p1 GA", "p1 grid"])
    ax.set_title("REM parameters of winners")
    ax = axes[1]
    ct = win.assign(kind=np.where(win.technique == "DDD", "DDD " + win.ddd_rule.fillna(""), "ZNE")).groupby(["method", "kind"]).size().unstack(fill_value=0)
    ct.T.plot.bar(ax=ax, color=[COLORS["GA"], COLORS["grid"]], rot=0)
    ax.set_title("winner technique / DDD rule (count, 60 runs each)")
    ax = axes[2]
    ct = win[win.technique == "ZNE"].groupby(["method", "factory"]).size().unstack(fill_value=0)
    ct.T.plot.bar(ax=ax, color=[COLORS["GA"], COLORS["grid"]], rot=0)
    ax.set_title("ZNE factory of winners")
    ax = axes[3]
    for m in ["GA", "grid"]:
        z = win[(win.method == m) & (win.technique == "ZNE")]
        ax.scatter(z.max_scale, z.ratio_raw, color=COLORS[m], s=18, label=m, alpha=0.8)
    ax.axhline(1, color="k", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("largest ZNE scale factor")
    ax.set_ylabel("ratio_raw (re-run)")
    ax.legend(fontsize=8)
    ax.set_title("scale factor vs re-run ratio")
    fig.suptitle("Mirrored N=10 winners (all levels): properties of chosen pipelines")
    fig.tight_layout()
    fig.savefig(OUT / "figB3_pipeline_properties_mirrored.png", dpi=150)
    plt.close(fig)


def main():
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 110)

    res, bench, broken = load_may()
    print("May broken conditions excluded:", list(broken))
    fig_budget(res, bench)
    fig_box(bench)
    ms_may = may_stats(bench)
    ms_may.to_csv(OUT / "may_stats.csv", index=False)
    print("\n=== May: GA vs others on re-evaluated fitness ===\n", ms_may.round(3).to_string(index=False))

    uniq, mean_by_epoch, zero = may_ga_diagnostics(res)
    print("\n=== May GA: unique chromosomes per generation (of 40) ===\n", uniq.round(1).to_string())
    print("\n=== May GA: population mean fitness by epoch ===\n", mean_by_epoch.round(3).to_string())
    print("\n=== share of evaluations with fitness exactly 0 (failures) ===\n", zero.round(3).to_string())

    feat = add_features(res, "chromosome_str")
    print("\n=== ZNE evals with a scale factor < 1 (invalid, from integer mutation) ===\n",
          feat[feat.technique == "ZNE"].groupby("method").apply(lambda g: pd.Series(dict(
              share=(g.min_scale < 1).mean(), fitness_if_invalid=g[g.min_scale < 1].fitness.mean(),
              fitness_if_valid=g[g.min_scale >= 1].fitness.mean()))).round(3).to_string())
    enr = enrichment(feat)
    enr.to_csv(OUT / "may_top10_enrichment.csv", index=False)
    print("\n=== May top-10% enrichment (hard circuits) ===\n", enr[enr.circuit.isin(HARD)].round(3).to_string(index=False))
    fig_may_props(feat)

    wins = may_ga_winners(res, bench)
    wins.to_csv(OUT / "may_ga_winners.csv", index=False)
    hw = wins[wins.circuit.isin(HARD)].sort_values("reeval_fitness", ascending=False)
    print("\n=== May GA winners on hard circuits (search vs re-eval) ===\n",
          hw[COND + ["search_fitness", "reeval_fitness", "chromosome_str"]].round(3).to_string(index=False))

    f = feat.dropna(subset=["fitness"]).reset_index(drop=True)
    extra = pd.get_dummies(f.circuit, prefix="circ", drop_first=True, dtype=float).assign(n_qubits=f.n_qubits.to_numpy())
    X = design_matrix(f, extra)
    auc, sd, coef = logistic(X, (f.fitness > 0.5).astype(int), GroupKFold(4), groups=f.circuit + f.n_qubits.astype(str))
    print(f"\n=== Logistic regression, May all evaluations, y = fitness>0.5 (n={len(f)}, base rate {(f.fitness > 0.5).mean():.2f}) ===")
    print(f"CV AUC (held-out circuit sizes) = {auc:.3f} ± {sd:.3f}\n", coef.round(3).to_string())
    coef.rename("coef_standardized").to_csv(OUT / "logreg_may_coefficients.csv")

    df, wide = load_mirrored()
    st = mirrored_stats(wide)
    st.to_csv(OUT / "mirrored_paired_tests.csv", index=False)
    print("\n=== Mirrored N=10: paired tests on log ratio_raw (multiplier <1 = first method better) ===\n", st.round(3).to_string(index=False))
    bl = between_levels(wide)
    bl.to_csv(OUT / "mirrored_between_levels.csv", index=False)
    print("\n=== Between levels, d = log(GA/grid) ===\n", bl.round(3).to_string(index=False))
    sig = signal_tests(df)
    sig.to_csv(OUT / "mirrored_signal_tests.csv", index=False)
    print("\n=== Signal: is mitigated different from 0? how often ratio<1? ===\n", sig.round(3).to_string(index=False))
    ms, tr = test_retest()
    tr.to_csv(OUT / "mirrored_test_retest.csv", index=False)
    print("\n=== Test-retest (same pipeline re-run 10x) ===\n", tr.round(3).to_string(index=False))
    fig_paired(wide, st)
    fig_signal(df, ms)

    win = add_features(df[df.method.isin(["GA", "grid"])].dropna(subset=["pipeline"]), "pipeline")
    win.to_csv(OUT / "mirrored_winners_parsed.csv", index=False)
    fig_mirrored_props(win)
    print("\n=== Mirrored winners: technique / rule counts ===\n",
          win.groupby(["method", "technique", "ddd_rule"], dropna=False).size().to_string())
    print("\n=== REM p0/p1 of winners ===\n", win.groupby("method")[["p0", "p1"]].describe().round(3).to_string())
    z = win[win.technique == "ZNE"].dropna(subset=["ratio_raw"])
    print("\nSpearman max_scale vs |mitigated| (ZNE winners):", stats.spearmanr(z.max_scale, z.mitigated.abs()))
    print("Spearman max_scale vs ratio_raw (ZNE winners):", stats.spearmanr(z.max_scale, z.ratio_raw))

    wf = win.dropna(subset=["ratio_raw"]).reset_index(drop=True)
    extra = pd.DataFrame({"difficulty": wf.difficulty, "is_GA": (wf.method == "GA").astype(float)})
    y = (wf.ratio_raw < 1).astype(int)
    auc, sd, coef = logistic(design_matrix(wf, extra), y, StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"\n=== Logistic regression, mirrored winners, y = ratio_raw<1 (n={len(wf)}, base rate {y.mean():.2f}) ===")
    print(f"CV AUC = {auc:.3f} ± {sd:.3f}\n", coef.round(3).to_string())
    coef.rename("coef_standardized").to_csv(OUT / "logreg_mirrored_coefficients.csv")

    g6 = wide[wide.difficulty == 6.0][["seed", "GA", "grid", "best_singleton"]]
    print("\n=== ×6 per seed ===\n", g6.round(3).to_string(index=False))
    ex = df[(df.method == "GA") & df.difficulty.isin([6.0, 8.0])].sort_values("ratio_raw")
    print("\n=== GA ×6/×8 best and worst runs ===\n",
          pd.concat([ex.head(4), ex.dropna(subset=["ratio_raw"]).tail(4)])[["difficulty", "seed", "mitigated", "ratio_raw", "pipeline"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
