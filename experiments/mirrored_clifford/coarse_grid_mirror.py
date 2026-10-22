"""
Coarse grid search on mirrored Clifford+T — ablation vs fine grid (288 configs).

Fine grid (brute_force_mirror.py): REM p0/p1 in {0.01, 0.05, 0.1, 0.2} → 288 evals
Coarse grid (this script):           REM p0/p1 in {0.05, 0.20}           → 72 evals

Same seeds/levels as GA + fine grid for direct comparison.

    python coarse_grid_mirror.py --seeds 0 1 2 --levels 0.5 1 2 4
"""
import sys, os, importlib.util, time, argparse, csv
from functools import partial

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

COARSE_REM = [0.05, 0.20]
FINE_REM = [0.01, 0.05, 0.1, 0.2]


def build_circuit(n_qubits, gates, seed):
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=n_qubits,
        num_oneq_cliffords=gates,
        num_twoq_cliffords=gates,
        num_t_gates=gates,
        seed=seed,
    )
    return circuit + m.cirq.inverse(circuit)


def best_over_sequences(n, obs, circuit, rem_p_vals):
    search = partial(m.grid_search, rem_p_vals=rem_p_vals)
    best_chrom, best_fit = None, -1.0
    n_evals = 0
    for seq in ['REM+DDD', 'REM+ZNE']:
        chrom, rows = search(n, obs, circuit, seq)
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
    ap.add_argument('--out', default='results/20260622_coarse_grid/coarse_grid_raw.csv')
    ap.add_argument('--done-marker', default='results/20260622_coarse_grid/coarse_grid_done.marker')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    n = args.n
    base = n ** 2
    obs = m.Observable(m.PauliString("Z" * n))

    done = set()
    rows = []
    if os.path.exists(args.out):
        with open(args.out, newline='') as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((int(r['seed']), float(r['difficulty'])))

    n_ddd = len(COARSE_REM) ** 2 * len(m.DDDGene.rules)
    n_zne = len(COARSE_REM) ** 2 * len(m.ZNEGene.factories) * len(m.ZNEGene.scale_noises)
    print(f"coarse grid: REM={COARSE_REM}, {n_ddd}+{n_zne}={n_ddd+n_zne} evals/run", flush=True)
    print(f"fine grid:   REM={FINE_REM}, 288 evals/run (see brute_force_raw.csv)\n", flush=True)
    print(f"resuming: {len(done)} run(s) done\n", flush=True)

    for seed in args.seeds:
        for k in args.levels:
            if (seed, float(k)) in done:
                print(f"seed={seed} diff x{k}: skip", flush=True)
                continue
            gates = int(round(k * base))
            circuit = build_circuit(n, gates, seed)
            ideal = float(m.run_pure(circuit, obs).real)
            noisy = float(m.run_noise(circuit, obs).real)

            t0 = time.perf_counter()
            best_chrom, best_fit, n_evals = best_over_sequences(
                n, obs, circuit, COARSE_REM)
            t_search = time.perf_counter() - t0

            try:
                mitigated = float(m.mitigate(circuit, best_chrom, m.execute).real)
            except Exception as e:
                mitigated = float('nan')
                print(f"  (mitigate failed: {e})", flush=True)

            denom = abs(noisy - ideal)
            ratio_raw = abs(mitigated - ideal) / denom if denom > 0 else float('inf')

            row = {
                'seed': seed,
                'difficulty': k,
                'gates_per_type': gates,
                'method': 'coarse_grid',
                'rem_grid': str(COARSE_REM),
                'n_evals': n_evals,
                't_search_s': round(t_search, 2),
                'best_fitness': round(best_fit, 5),
                'ideal': round(ideal, 5),
                'noisy': round(noisy, 5),
                'mitigated_best': round(mitigated, 5),
                'ratio_raw': round(ratio_raw, 5),
                'best_pipeline': str(best_chrom),
            }
            rows.append(row)
            print(f"seed={seed} diff x{k:<4} evals={n_evals:>3} t={t_search:7.1f}s "
                  f"best={best_fit:.4f} ratio={ratio_raw:.4f}", flush=True)

            with open(args.out, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerows(rows)

    with open(args.done_marker, 'w') as f:
        f.write(f"done: {len(rows)} runs at {time.ctime()}\n")
    print(f"\nSaved {args.out}", flush=True)


if __name__ == '__main__':
    main()
