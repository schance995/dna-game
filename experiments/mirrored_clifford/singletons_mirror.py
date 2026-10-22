"""
Default singleton mitigators (REM, ZNE, DDD) on mirrored Clifford+T circuits,
matching multi_seed_sweep.py seeds/levels for direct comparison with GA.

Uses the same default mitiq calls as benchmark() in dna-game-2.py (not GA chromosomes).

Outputs (checkpointed after every run):
  singletons_raw.csv

    python singletons_mirror.py --seeds 0 1 2 --levels 0.5 1 2 4
"""
import sys, os, importlib.util, time, argparse, csv

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def build_circuit(n_qubits, gates, seed):
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=n_qubits,
        num_oneq_cliffords=gates,
        num_twoq_cliffords=gates,
        num_t_gates=gates,
        seed=seed,
    )
    return circuit + m.cirq.inverse(circuit)


def run_singleton(name, circuit, n, obs):
    """Run one default singleton; return mitigated expectation."""
    stripped = m._strip_measurements(circuit)
    if name == 'REM':
        icm = m.rem.generate_inverse_confusion_matrix(n, 0.05, 0.05)
        executor = m.rem.mitigate_executor(m.execute, inverse_confusion_matrix=icm)
        return obs.expectation(stripped, executor)
    if name == 'ZNE':
        return m.zne.execute_with_zne(stripped, m.execute, obs)
    if name == 'DDD':
        return m.ddd.execute_with_ddd(stripped, m.execute, obs, rule=m.ddd.rules.xx)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--levels', type=float, nargs='+', default=[0.5, 1, 2, 4])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--out', default='singletons_raw.csv')
    ap.add_argument('--done-marker', default='singletons_done.marker')
    args = ap.parse_args()

    n = args.n
    base = n ** 2
    obs = m.Observable(m.PauliString("Z" * n))
    methods = ['REM', 'ZNE', 'DDD']

    done = set()
    rows = []
    if os.path.exists(args.out):
        with open(args.out, newline='') as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((int(r['seed']), float(r['difficulty']), r['method']))

    print(f"n={n}, levels={args.levels}, seeds={args.seeds}, mirrored singletons\n"
          f"resuming: {len(done)} run(s) already done\n", flush=True)

    for seed in args.seeds:
        for k in args.levels:
            gates = int(round(k * base))
            circuit = build_circuit(n, gates, seed)
            ideal = float(m.run_pure(circuit, obs).real)
            noisy = float(m.run_noise(circuit, obs).real)

            for name in methods:
                key = (seed, float(k), name)
                if key in done:
                    print(f"seed={seed} diff x{k} {name}: skip", flush=True)
                    continue

                t0 = time.perf_counter()
                try:
                    mitigated = float(run_singleton(name, circuit, n, obs).real)
                except Exception as e:
                    mitigated = float('nan')
                    print(f"  ({name} failed: {e})", flush=True)
                t_run = time.perf_counter() - t0

                ratio = m.compute_mitigation_ratio(ideal, noisy, mitigated)
                fitness = m.ratio_to_fitness(ratio)
                denom = abs(noisy - ideal)
                ratio_raw = abs(mitigated - ideal) / denom if denom > 0 else float('inf')

                row = {
                    'seed': seed,
                    'difficulty': k,
                    'gates_per_type': gates,
                    'method': name,
                    't_run_s': round(t_run, 2),
                    'fitness': round(fitness, 5),
                    'ideal': round(ideal, 5),
                    'noisy': round(noisy, 5),
                    'mitigated': round(mitigated, 5),
                    'ratio': round(float(ratio), 5),
                    'ratio_raw': round(ratio_raw, 5),
                }
                rows.append(row)
                print(f"seed={seed} diff x{k:<4} {name:<4} t={t_run:6.1f}s "
                      f"fitness={fitness:.4f} ratio_raw={ratio_raw:.4f}", flush=True)

                with open(args.out, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    with open(args.done_marker, 'w') as f:
        f.write(f"done: {len(rows)} runs at {time.ctime()}\n")
    print(f"\nSaved {args.out}", flush=True)


if __name__ == '__main__':
    main()
