"""
Multi-seed mirrored Clifford+T difficulty sweep.

Validates the single-seed (42) finding from results/20260610_mirror_sweep/:
GA convergence generation stays roughly flat with depth, while best
achievable mitigation quality drops. Runs several seeds per depth level so
the claim gets mean +- std instead of one circuit instance.

Outputs:
  multi_seed_raw.csv     one row per (seed, level): fitness, convergence gen,
                         raw ideal/noisy/mitigated values, error ratio
                         recomputed from raw values, best pipeline string
  multi_seed_curves.csv  per-generation max/mean fitness for every run
  multi_seed_summary.csv mean +- std across seeds per level
  multi_seed_sweep.png   error-bar plots of the three headline metrics

Note: mutate() edits chromosomes in place, so only the peak-fitness
chromosome (deep-copied each generation) is used for ratio_raw / best_pipeline.
final_pipeline / ratio_raw_last_gen log the last generation's winner for comparison.

    python multi_seed_sweep.py --seeds 0 1 2 --levels 0.5 1 2 4
"""
import sys, os, importlib.util, time, argparse, csv, statistics

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def build_circuit(n_qubits, gates, seed):
    """Mirrored random Clifford+T: ideal <Z^n> is exactly +1."""
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=n_qubits,
        num_oneq_cliffords=gates,
        num_twoq_cliffords=gates,
        num_t_gates=gates,
        seed=seed,
    )
    return circuit + m.cirq.inverse(circuit)


def generations_to_converge(max_fits, eps=1e-3):
    target = max(max_fits)
    for i, v in enumerate(max_fits):
        if target - v <= eps:
            return i
    return len(max_fits) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--levels', type=float, nargs='+', default=[0.5, 1, 2, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--pop', type=int, default=40)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--eps', type=float, default=1e-3)
    ap.add_argument('--out', default='multi_seed_raw.csv',
                    help='accumulating results file; completed (seed,level) runs are skipped')
    ap.add_argument('--curves-out', default='multi_seed_curves.csv')
    ap.add_argument('--done-marker', default='sweep_done.marker',
                    help='file written when the full grid is complete')
    args = ap.parse_args()

    n = args.n
    base = n ** 2
    obs = m.Observable(m.PauliString("Z" * n))

    # Resume: load any prior results and skip (seed, level) pairs already done.
    float_cols = {'difficulty', 't_ga_s', 'best_fitness', 'final_best_fitness',
                  'ideal', 'noisy', 'mitigated_final_best', 'ratio_raw'}
    int_cols = {'seed', 'gates_per_type', 'circuit_gates', 'gens_to_converge'}

    def cast(r):
        for col in float_cols & r.keys():
            r[col] = float(r[col])
        for col in int_cols & r.keys():
            r[col] = int(r[col])
        return r

    raw_rows, curve_rows, done = [], [], set()
    if os.path.exists(args.out):
        with open(args.out, newline='') as f:
            for r in csv.DictReader(f):
                raw_rows.append(cast(r))
                done.add((int(r['seed']), float(r['difficulty'])))
    if os.path.exists(args.curves_out):
        with open(args.curves_out, newline='') as f:
            curve_rows = list(csv.DictReader(f))
    print(f"n={n}, levels={args.levels}, seeds={args.seeds}, "
          f"pop={args.pop}, epochs={args.epochs}, mirrored", flush=True)
    print(f"resuming: {len(done)} run(s) already complete, will skip those\n", flush=True)

    for seed in args.seeds:
        for k in args.levels:
            if (seed, float(k)) in done:
                print(f"seed={seed} diff x{k}: already done, skipping", flush=True)
                continue
            gates = int(round(k * base))
            circuit = build_circuit(n, gates, seed)

            ideal = float(m.run_pure(circuit, obs).real)
            noisy = float(m.run_noise(circuit, obs).real)

            t0 = time.perf_counter()
            pops, fits, best_chrom_peak = m.genetic_algorithm_2(
                pop_size=args.pop, n_qubits=n, obs=obs,
                circuit=circuit, epochs=args.epochs,
            )
            t_ga = time.perf_counter() - t0

            max_fits = [f[0] for f in fits]  # populations are fitness-sorted
            mean_fits = [sum(f) / len(f) for f in fits]
            for g, (mx, mn) in enumerate(zip(max_fits, mean_fits)):
                curve_rows.append({
                    'seed': seed, 'difficulty': k, 'gen': g,
                    'max_fit': round(mx, 5), 'mean_fit': round(mn, 5),
                })

            best_chrom_final = pops[-1][0]

            def rerun(chrom):
                try:
                    return float(m.mitigate(circuit, chrom, m.execute).real)
                except Exception as e:
                    print(f"  (mitigate failed: {e})", flush=True)
                    return float('nan')

            mitigated_peak = rerun(best_chrom_peak)
            mitigated_final = rerun(best_chrom_final)

            denom = abs(noisy - ideal)
            ratio_peak = abs(mitigated_peak - ideal) / denom if denom > 0 else float('inf')
            ratio_final = abs(mitigated_final - ideal) / denom if denom > 0 else float('inf')

            row = {
                'seed': seed,
                'difficulty': k,
                'gates_per_type': gates,
                'circuit_gates': 2 * 3 * gates,
                't_ga_s': round(t_ga, 2),
                'gens_to_converge': generations_to_converge(max_fits, args.eps),
                'best_fitness': round(max(max_fits), 5),
                'final_best_fitness': round(max_fits[-1], 5),
                'ideal': round(ideal, 5),
                'noisy': round(noisy, 5),
                'mitigated_final_best': round(mitigated_peak, 5),
                'mitigated_last_gen': round(mitigated_final, 5),
                'ratio_raw': round(ratio_peak, 5),
                'ratio_raw_last_gen': round(ratio_final, 5),
                'best_pipeline': str(best_chrom_peak),
                'final_pipeline': str(best_chrom_final),
            }
            raw_rows.append(row)
            print(f"seed={seed} diff x{k:<4} gates/type={gates:>4} "
                  f"t_ga={t_ga:7.1f}s conv@gen={row['gens_to_converge']:>2} "
                  f"best={row['best_fitness']:.4f} ratio_raw={ratio_peak:.4f} "
                  f"(last_gen={ratio_final:.4f}) "
                  f"pipeline={row['best_pipeline']}", flush=True)

            # checkpoint after every run so partial results survive
            for name, rows in [(args.out, raw_rows),
                               (args.curves_out, curve_rows)]:
                with open(name, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    # aggregate across seeds
    summary_rows = []
    for k in args.levels:
        sub = [r for r in raw_rows if r['difficulty'] == k]
        def agg(key):
            vals = [r[key] for r in sub if r[key] == r[key]]  # drop NaN
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return round(mu, 5), round(sd, 5)
        bf_mu, bf_sd = agg('best_fitness')
        cg_mu, cg_sd = agg('gens_to_converge')
        rr_mu, rr_sd = agg('ratio_raw')
        summary_rows.append({
            'difficulty': k, 'n_seeds': len(sub),
            'best_fitness_mean': bf_mu, 'best_fitness_std': bf_sd,
            'gens_to_converge_mean': cg_mu, 'gens_to_converge_std': cg_sd,
            'ratio_raw_mean': rr_mu, 'ratio_raw_std': rr_sd,
        })
    with open('multi_seed_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print("\nSaved multi_seed_raw.csv, multi_seed_curves.csv, multi_seed_summary.csv",
          flush=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        xs = [r['difficulty'] for r in summary_rows]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (mu_key, sd_key, label) in zip(axes, [
            ('best_fitness_mean', 'best_fitness_std', 'Best fitness'),
            ('gens_to_converge_mean', 'gens_to_converge_std', 'Generations to converge'),
            ('ratio_raw_mean', 'ratio_raw_std', 'Error ratio (lower = better)'),
        ]):
            ax.errorbar(xs, [r[mu_key] for r in summary_rows],
                        yerr=[r[sd_key] for r in summary_rows],
                        fmt='o-', capsize=4)
            ax.set_xlabel('Difficulty (x n^2 gates, mirrored)')
            ax.set_ylabel(label)
            ax.set_xscale('log', base=2)
        fig.suptitle(f'Multi-seed mirrored sweep — n={n}, pop={args.pop}, '
                     f'epochs={args.epochs}, seeds={args.seeds}')
        fig.tight_layout()
        plt.savefig('multi_seed_sweep.png', dpi=150)
        print("Saved multi_seed_sweep.png", flush=True)
    except Exception as e:
        print(f"(plot skipped: {e})", flush=True)

    with open(args.done_marker, 'w') as f:
        f.write(f"done: {len(raw_rows)} runs at {time.ctime()}\n")
    print(f"Wrote {args.done_marker}", flush=True)


if __name__ == '__main__':
    main()
