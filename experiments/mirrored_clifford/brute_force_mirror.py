"""
Brute-force baselines (grid search + random search) on the mirrored
Clifford+T difficulty levels, matching the multi-seed GA sweep
(multi_seed_sweep.py) circuit-for-circuit so results are directly comparable.

random_search uses n_samples=420 per sequence type (840 total), matching the
GA budget of pop=40 x 21 generations of evaluations.

Outputs (checkpointed after every seed/level):
  brute_force_raw.csv  one row per (seed, level, method): best fitness, best
                       pipeline, fresh ideal/noisy/mitigated re-execution,
                       error ratio from raw values

    python brute_force_mirror.py --seeds 0 1 2 --levels 0.5 1 2 4
"""
import sys, os, importlib.util, time, argparse, csv

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def build_circuit(n_qubits, gates, seed):
    """Mirrored random Clifford+T, identical to multi_seed_sweep.py."""
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=n_qubits,
        num_oneq_cliffords=gates,
        num_twoq_cliffords=gates,
        num_t_gates=gates,
        seed=seed,
    )
    return circuit + m.cirq.inverse(circuit)


def best_over_sequences(search_fn, n, obs, circuit, **kwargs):
    """Run a search over both sequence types, return overall best."""
    best_chrom, best_fit = None, -1.0
    n_evals = 0
    for seq in ['REM+DDD', 'REM+ZNE']:
        chrom, rows = search_fn(n, obs, circuit, seq, **kwargs)
        n_evals += len(rows)
        top = max(rows, key=lambda r: r['fitness'])
        if top['fitness'] > best_fit:
            best_fit = top['fitness']
            best_chrom = chrom
    return best_chrom, best_fit, n_evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--levels', type=float, nargs='+', default=[0.5, 1, 2, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--random-samples', type=int, default=420,
                    help='random_search samples per sequence type (420 x 2 = GA budget)')
    ap.add_argument('--methods', nargs='+', default=['grid_search', 'random_search'],
                    choices=['grid_search', 'random_search'])
    ap.add_argument('--out', default='brute_force_raw.csv')
    ap.add_argument('--done-marker', default='brute_force_done.marker')
    args = ap.parse_args()

    n = args.n
    base = n ** 2
    obs = m.Observable(m.PauliString("Z" * n))

    all_methods = {
        'grid_search': (m.grid_search, {}),
        'random_search': (m.random_search, {'n_samples': args.random_samples}),
    }
    methods = [(name, all_methods[name][0], all_methods[name][1]) for name in args.methods]

    done = set()
    rows = []
    if os.path.exists(args.out):
        with open(args.out, newline='') as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((int(r['seed']), float(r['difficulty']), r['method']))

    print(f"n={n}, levels={args.levels}, seeds={args.seeds}, "
          f"random_samples={args.random_samples}/sequence, mirrored\n"
          f"resuming: {len(done)} run(s) already done\n", flush=True)

    for seed in args.seeds:
        for k in args.levels:
            gates = int(round(k * base))
            circuit = build_circuit(n, gates, seed)
            ideal = float(m.run_pure(circuit, obs).real)
            noisy = float(m.run_noise(circuit, obs).real)

            for name, fn, kwargs in methods:
                if (seed, float(k), name) in done:
                    print(f"seed={seed} diff x{k} {name}: skip", flush=True)
                    continue
                t0 = time.perf_counter()
                best_chrom, best_fit, n_evals = best_over_sequences(
                    fn, n, obs, circuit, **kwargs)
                t_search = time.perf_counter() - t0

                try:
                    mitigated = float(m.mitigate(circuit, best_chrom, m.execute).real)
                except Exception as e:
                    mitigated = float('nan')
                    print(f"  (mitigate failed: {e})", flush=True)

                denom = abs(noisy - ideal)
                ratio_raw = abs(mitigated - ideal) / denom if denom > 0 else float('inf')

                rows.append({
                    'seed': seed,
                    'difficulty': k,
                    'gates_per_type': gates,
                    'method': name,
                    'n_evals': n_evals,
                    't_search_s': round(t_search, 2),
                    'best_fitness': round(best_fit, 5),
                    'ideal': round(ideal, 5),
                    'noisy': round(noisy, 5),
                    'mitigated_best': round(mitigated, 5),
                    'ratio_raw': round(ratio_raw, 5),
                    'best_pipeline': str(best_chrom),
                })
                print(f"seed={seed} diff x{k:<4} {name:<14} evals={n_evals:>4} "
                      f"t={t_search:7.1f}s best={best_fit:.4f} "
                      f"ratio_raw={ratio_raw:.4f} pipeline={best_chrom}", flush=True)

                with open(args.out, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    with open(args.done_marker, 'w') as f:
        f.write(f"done: {len(rows)} runs at {time.ctime()}\n")
    print(f"\nSaved {args.out}", flush=True)


if __name__ == '__main__':
    main()
