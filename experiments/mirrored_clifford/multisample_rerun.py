"""
Multi-sample re-run of saved best pipelines (GA + grid).

Reads best_pipeline strings from prior sweep CSVs, re-executes mitigation
n_reps times per pipeline, and reports mean +- std of ratio_raw.

Grid and GA only (random search omitted — single-shot overfitting).

    python multisample_rerun.py --n-reps 10
    python multisample_rerun.py --ga-csv results/20260613_multi_seed/multi_seed_raw.csv \\
        --grid-csv results/20260616_p0_baselines/brute_force_raw.csv
"""
import argparse
import csv
import os
import statistics
import time

from mirror_utils import build_circuit, load_dna_game, parse_pipeline, ratio_raw


def load_jobs(ga_csv, grid_csv):
    jobs = []
    for source, path, method in [
        ("ga", ga_csv, "GA"),
        ("grid", grid_csv, "grid_search"),
    ]:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("method") not in (None, "grid_search", "GA"):
                    continue
                jobs.append({
                    "source": source,
                    "method": method if source == "grid" else "GA",
                    "seed": int(r["seed"]),
                    "difficulty": float(r["difficulty"]),
                    "gates_per_type": int(r["gates_per_type"]),
                    "best_pipeline": r["best_pipeline"],
                    "ratio_raw_single": float(r["ratio_raw"]),
                })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--n-reps", type=int, default=10)
    ap.add_argument("--ga-csv", default="results/20260613_multi_seed/multi_seed_raw.csv")
    ap.add_argument("--grid-csv", default="results/20260616_p0_baselines/brute_force_raw.csv")
    ap.add_argument("--out-raw", default="results/20260617_multisample/multisample_raw.csv")
    ap.add_argument("--out-summary", default="results/20260617_multisample/multisample_summary.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_raw) or ".", exist_ok=True)

    m = load_dna_game()
    n = args.n
    obs = m.Observable(m.PauliString("Z" * n))
    base = n ** 2

    jobs = load_jobs(args.ga_csv, args.grid_csv)
    done = set()
    raw_rows = []
    if os.path.exists(args.out_raw):
        with open(args.out_raw, newline="") as f:
            for r in csv.DictReader(f):
                raw_rows.append(r)
                done.add((r["source"], int(r["seed"]), float(r["difficulty"]), int(r["rep"])))

    print(f"{len(jobs)} pipelines x {args.n_reps} reps (resume: {len(done)} done)\n", flush=True)

    for job in jobs:
        seed, k = job["seed"], job["difficulty"]
        gates = int(round(k * base))
        circuit = build_circuit(m, n, gates, seed)
        ideal = float(m.run_pure(circuit, obs).real)
        noisy = float(m.run_noise(circuit, obs).real)
        chrom = parse_pipeline(job["best_pipeline"], n, obs, m)

        for rep in range(args.n_reps):
            key = (job["source"], seed, k, rep)
            if key in done:
                continue
            t0 = time.perf_counter()
            try:
                mitigated = float(m.mitigate(circuit, chrom, m.execute).real)
            except Exception as e:
                mitigated = float("nan")
                print(f"  rep failed {job['source']} seed={seed} x{k} rep={rep}: {e}", flush=True)
            rr = ratio_raw(ideal, noisy, mitigated)
            row = {
                "source": job["source"],
                "method": job["method"],
                "seed": seed,
                "difficulty": k,
                "gates_per_type": gates,
                "rep": rep,
                "ideal": round(ideal, 5),
                "noisy": round(noisy, 5),
                "mitigated": round(mitigated, 5) if mitigated == mitigated else "",
                "ratio_raw": round(rr, 5) if rr == rr else "",
                "t_s": round(time.perf_counter() - t0, 2),
                "best_pipeline": job["best_pipeline"],
            }
            raw_rows.append(row)
            done.add(key)

            with open(args.out_raw, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerows(raw_rows)

            print(f"{job['source']:>5} seed={seed} x{k:<4} rep={rep+1}/{args.n_reps} "
                  f"ratio={rr:.4f} t={row['t_s']:.1f}s", flush=True)

    summary_rows = []
    for job in jobs:
        sub = [
            r for r in raw_rows
            if r["source"] == job["source"]
            and int(r["seed"]) == job["seed"]
            and float(r["difficulty"]) == job["difficulty"]
            and r["ratio_raw"] not in ("", "nan")
        ]
        if not sub:
            continue
        vals = [float(r["ratio_raw"]) for r in sub]
        summary_rows.append({
            "source": job["source"],
            "method": job["method"],
            "seed": job["seed"],
            "difficulty": job["difficulty"],
            "n_reps": len(vals),
            "ratio_raw_mean": round(statistics.mean(vals), 5),
            "ratio_raw_std": round(statistics.stdev(vals), 5) if len(vals) > 1 else 0.0,
            "ratio_raw_single": job["ratio_raw_single"],
            "best_pipeline": job["best_pipeline"],
        })

    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSaved {args.out_raw}", flush=True)
    print(f"Saved {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
