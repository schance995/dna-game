from pathlib import Path

import numpy as np
import pandas as pd

from analysis_20260923 import parse_pipeline

R = Path("results")
OUT = Path("data")
OUT.mkdir(exist_ok=True)
GA_EVALS = 40 * 21
SINGLETON_PIPELINES = {
    "REM": "default rem(0.05, 0.05)",
    "ZNE": "default zne(RichardsonFactory([1, 2, 3]), fold_gates_at_random)",
    "DDD": "default ddd(xx)",
}


def ga_runs():
    old = pd.read_csv(R / "20260613_multi_seed/multi_seed_raw.csv").assign(ga_reeval="last_gen")
    new = pd.concat([pd.read_csv(R / "20260706_ga_expand/multi_seed_raw_3-9.csv"),
                     pd.read_csv(R / "20260726_x6_x8/multi_seed_raw.csv")]).assign(ga_reeval="best_ever")
    ga = pd.concat([old, new], ignore_index=True)
    return pd.DataFrame({
        "seed": ga.seed, "multiplier": ga.difficulty, "gates_per_type": ga.gates_per_type, "method": "GA",
        "ideal": ga.ideal, "noisy": ga.noisy, "mitigated": ga.mitigated_final_best, "ratio_raw": ga.ratio_raw,
        "search_fitness": ga.best_fitness, "n_evals": GA_EVALS, "runtime_s": ga.t_ga_s,
        "pipeline": ga.best_pipeline, "ga_reeval": ga.ga_reeval, "ratio_raw_last_gen": ga.get("ratio_raw_last_gen"),
        "gens_to_converge": ga.gens_to_converge,
    })


def grid_runs():
    gr = pd.concat([pd.read_csv(R / "20260623_expand_seeds/brute_force_raw.csv"),
                    pd.read_csv(R / "20260726_x6_x8/brute_force_raw.csv")])
    gr = gr[gr.method == "grid_search"]
    return pd.DataFrame({
        "seed": gr.seed, "multiplier": gr.difficulty, "gates_per_type": gr.gates_per_type, "method": "grid",
        "ideal": gr.ideal, "noisy": gr.noisy, "mitigated": gr.mitigated_best, "ratio_raw": gr.ratio_raw,
        "search_fitness": gr.best_fitness, "n_evals": gr.n_evals, "runtime_s": gr.t_search_s, "pipeline": gr.best_pipeline,
    })


def singleton_runs():
    sg = pd.concat([pd.read_csv(R / "20260623_expand_seeds/singletons_raw.csv"),
                    pd.read_csv(R / "20260726_x6_x8/singletons_raw.csv")])
    return pd.DataFrame({
        "seed": sg.seed, "multiplier": sg.difficulty, "gates_per_type": sg.gates_per_type, "method": sg.method,
        "ideal": sg.ideal, "noisy": sg.noisy, "mitigated": sg.mitigated, "ratio_raw": sg.ratio_raw,
        "search_fitness": sg.fitness, "n_evals": 1, "runtime_s": sg.t_run_s, "pipeline": sg.method.map(SINGLETON_PIPELINES),
    })


def main():
    runs = pd.concat([ga_runs(), grid_runs(), singleton_runs()], ignore_index=True)
    runs["circuit_gates"] = 2 * 3 * runs.gates_per_type
    feats = pd.DataFrame([parse_pipeline(s) if isinstance(s, str) and not s.startswith("default") else parse_pipeline("")
                          for s in runs.pipeline])
    feats.loc[runs.method.isin(["REM", "ZNE", "DDD"]).to_numpy(), "technique"] = runs.method[runs.method.isin(["REM", "ZNE", "DDD"])].to_numpy()
    runs = pd.concat([runs, feats], axis=1)
    runs["log_ratio"] = np.log(runs.ratio_raw)
    cols = ["seed", "multiplier", "gates_per_type", "circuit_gates", "method", "ideal", "noisy", "mitigated", "ratio_raw",
            "log_ratio", "search_fitness", "n_evals", "runtime_s", "pipeline", "technique", "p0", "p1", "ddd_rule",
            "factory", "fold", "n_scale", "min_scale", "max_scale", "ga_reeval", "ratio_raw_last_gen", "gens_to_converge"]
    runs = runs[cols].sort_values(["multiplier", "seed", "method"])
    runs.to_csv(OUT / "mirrored_runs.csv", index=False)

    curves = pd.concat([pd.read_csv(R / "20260613_multi_seed/multi_seed_curves.csv"),
                        pd.read_csv(R / "20260706_ga_expand/multi_seed_curves_3-9.csv"),
                        pd.read_csv(R / "20260726_x6_x8/multi_seed_curves.csv")])
    curves.rename(columns={"difficulty": "multiplier"}).sort_values(["multiplier", "seed", "gen"]).to_csv(
        OUT / "mirrored_ga_curves.csv", index=False)

    rt = pd.read_csv(R / "20260617_multisample/multisample_raw.csv")
    rt = rt.rename(columns={"difficulty": "multiplier", "best_pipeline": "pipeline", "t_s": "runtime_s"})
    rt["method"] = rt.method.replace({"grid_search": "grid"})
    rt.drop(columns=["source"]).to_csv(OUT / "mirrored_retest.csv", index=False)

    print(runs.groupby(["multiplier", "method"]).size().unstack())
    print(len(curves), "curve rows;", len(rt), "retest rows")


if __name__ == "__main__":
    main()
