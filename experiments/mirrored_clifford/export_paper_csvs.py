"""
Convert mirrored Clifford sweep CSVs into results.csv + benchmarks.csv
for make_paper_figures.py (rev3-dev format).

    python export_paper_csvs.py --out results/20260623_mirrored_paper
"""
import argparse
import csv
import math
from pathlib import Path

N_QUBITS = 7
LEVELS = [0.5, 1.0, 2.0, 4.0]
ROOT = Path(__file__).parent / "results"


def diff_label(k):
    return f"×{k:g}"


def fitness_to_ratio(fitness):
    if fitness <= 0 or fitness >= 1:
        return float("inf")
    return math.exp(-math.log(fitness / (1 - fitness)))


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/20260623_mirrored_paper")
    ap.add_argument("--singletons", default=str(ROOT / "20260623_expand_seeds" / "singletons_raw.csv"))
    ap.add_argument("--grid", default=str(ROOT / "20260623_expand_seeds" / "brute_force_raw.csv"))
    ap.add_argument(
        "--ga-raw",
        nargs="+",
        default=[
            str(ROOT / "20260613_multi_seed" / "multi_seed_raw.csv"),
            str(ROOT / "20260706_ga_expand" / "multi_seed_raw_3-9.csv"),
        ],
    )
    ap.add_argument(
        "--ga-curves",
        nargs="+",
        default=[
            str(ROOT / "20260613_multi_seed" / "multi_seed_curves.csv"),
            str(ROOT / "20260706_ga_expand" / "multi_seed_curves_3-9.csv"),
        ],
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bench_rows = []
    result_rows = []

    for r in load_csv(Path(args.singletons)):
        k = float(r["difficulty"])
        fit = float(r["fitness"])
        ratio = float(r["ratio_raw"])
        bench_rows.append({
            "method": "defaults",
            "sequence": r["method"],
            "circuit": diff_label(k),
            "n_qubits": N_QUBITS,
            "seed": int(r["seed"]),
            "fitness": fit,
            "mitigation_ratio": ratio,
        })

    for r in load_csv(Path(args.grid)):
        if r["method"] != "grid_search":
            continue
        k = float(r["difficulty"])
        fit = float(r["best_fitness"])
        ratio = float(r["ratio_raw"])
        bench_rows.append({
            "method": "grid_search",
            "sequence": "DNA",
            "circuit": diff_label(k),
            "n_qubits": N_QUBITS,
            "seed": int(r["seed"]),
            "fitness": fit,
            "mitigation_ratio": ratio,
        })

    ga_raw = []
    for path in args.ga_raw:
        ga_raw.extend(load_csv(Path(path)))
    for r in ga_raw:
        k = float(r["difficulty"])
        fit = float(r["best_fitness"])
        ratio = float(r["ratio_raw"])
        bench_rows.append({
            "method": "GA",
            "sequence": "DNA",
            "circuit": diff_label(k),
            "n_qubits": N_QUBITS,
            "seed": int(r["seed"]),
            "fitness": fit,
            "mitigation_ratio": ratio,
        })

    for path in args.ga_curves:
        for r in load_csv(Path(path)):
            k = float(r["difficulty"])
            fit = float(r["max_fit"])
            result_rows.append({
                "method": "GA",
                "sequence": "DNA",
                "circuit": diff_label(k),
                "n_qubits": N_QUBITS,
                "seed": int(r["seed"]),
                "epoch": int(r["gen"]),
                "fitness": fit,
                "mitigation_ratio": fitness_to_ratio(fit),
                "chromosome_str": "",
            })

    bench_fields = ["method", "sequence", "circuit", "n_qubits", "seed", "fitness", "mitigation_ratio"]
    result_fields = [
        "method", "sequence", "circuit", "n_qubits", "seed", "epoch",
        "fitness", "mitigation_ratio", "chromosome_str",
    ]

    with open(out / "benchmarks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bench_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(bench_rows)

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result_fields)
        w.writeheader()
        w.writerows(result_rows)

    ga_seeds = sorted({int(r["seed"]) for r in ga_raw})
    grid_seeds = sorted({
        int(r["seed"]) for r in load_csv(Path(args.grid)) if r["method"] == "grid_search"
    })
    sg_seeds = sorted({int(r["seed"]) for r in load_csv(Path(args.singletons))})

    print(f"Wrote {out}/benchmarks.csv  ({len(bench_rows)} rows)")
    print(f"Wrote {out}/results.csv     ({len(result_rows)} rows)")
    print(f"  singletons seeds: {sg_seeds} (n={len(sg_seeds)})")
    print(f"  grid seeds:       {grid_seeds} (n={len(grid_seeds)})")
    print(f"  GA seeds:         {ga_seeds} (n={len(ga_seeds)})")


if __name__ == "__main__":
    main()
