"""
Difficulty sweep for the Random Clifford+T benchmark.

Baseline (convergence_check.py / benchmark_timing.py) uses gates = n**2 for each
of {1q Cliffords, 2q Cliffords, T gates}. That circuit converges quickly. This
script scales the gate count above the n**2 baseline by a difficulty factor and
measures how much longer the GA takes to converge as difficulty rises.

For each difficulty level it records:
  - per-eval fitness wall-clock (single evaluate_fitness call)
  - full GA wall-clock
  - generations-to-converge (first gen within EPS of the final best max-fitness)
  - final best fitness

Outputs difficulty_sweep.csv and difficulty_sweep.png.

Run on a machine where mitiq/cirq are installed (your Mac, or the remote box):
    python difficulty_sweep.py
    python difficulty_sweep.py --n 7 --levels 1 2 4 8 --pop 40 --epochs 20 --seed 42
"""
import sys, importlib.util, time, argparse, csv

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def build_circuit(n_qubits, gates, seed, mirror=False):
    """Random Clifford+T circuit with `gates` of each gate type.

    With mirror=True, appends the circuit's inverse so the ideal state returns
    to |0...0> and the ideal <Z^n> is exactly +1. This keeps the fitness
    denominator |noisy - ideal| well above the shot-noise floor (the plain
    random circuit scrambles <Z^n> to ~0, degenerating the fitness metric --
    see results/20260610_difficulty_sweep/verify_saturation_output.txt).
    """
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=n_qubits,
        num_oneq_cliffords=gates,
        num_twoq_cliffords=gates,
        num_t_gates=gates,
        seed=seed,
    )
    if mirror:
        circuit = circuit + m.cirq.inverse(circuit)
    return circuit


def generations_to_converge(max_fits, eps=1e-3):
    """First generation index whose max-fitness is within eps of the final best."""
    target = max(max_fits)
    for i, v in enumerate(max_fits):
        if target - v <= eps:
            return i
    return len(max_fits) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=7, help='number of qubits')
    ap.add_argument('--levels', type=float, nargs='+', default=[1, 2, 4, 8],
                    help='difficulty multipliers applied to the n**2 gate baseline')
    ap.add_argument('--pop', type=int, default=40, help='GA population size')
    ap.add_argument('--epochs', type=int, default=20, help='GA generations')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eps', type=float, default=1e-3,
                    help='convergence tolerance on max fitness')
    ap.add_argument('--mirror', action='store_true',
                    help='append the inverse circuit so the ideal <Z^n> is exactly +1')
    args = ap.parse_args()

    n = args.n
    base = n ** 2
    obs = m.Observable(m.PauliString("Z" * n))

    rows = []
    print(f"n={n}, baseline gates (n^2) = {base}, levels = {args.levels}, "
          f"mirror = {args.mirror}\n")

    for k in args.levels:
        gates = int(round(k * base))
        circuit = build_circuit(n, gates, args.seed, mirror=args.mirror)

        # single-eval timing (REM+ZNE chromosome, same as benchmark_timing.py)
        chrom = [
            m.REMGene(p0=0.05, p1=0.05, n_qubits=n),
            m.ZNEGene(factory=m.ZNEGene.factories[0],
                      scale_noise=m.ZNEGene.scale_noises[0],
                      num_to_avg=1, obs=obs),
        ]
        t0 = time.perf_counter()
        m.evaluate_fitness(chrom, circuit, obs)
        t_eval = time.perf_counter() - t0

        # full GA convergence run
        t0 = time.perf_counter()
        _pops, fits, _best = m.genetic_algorithm_2(
            pop_size=args.pop, n_qubits=n, obs=obs,
            circuit=circuit, epochs=args.epochs,
        )
        t_ga = time.perf_counter() - t0

        max_fits = [max(f) for f in fits]
        gen_conv = generations_to_converge(max_fits, eps=args.eps)
        best = max(max_fits)

        row = {
            'difficulty': k,
            'gates_per_type': gates,
            'total_gates': 3 * gates,
            't_eval_s': round(t_eval, 3),
            't_ga_s': round(t_ga, 2),
            'gens_to_converge': gen_conv,
            'best_fitness': round(best, 5),
        }
        rows.append(row)
        print(f"diff x{k:>4}  gates/type={gates:>4}  t_eval={t_eval:6.2f}s  "
              f"t_ga={t_ga:8.2f}s  conv@gen={gen_conv:>2}  best={best:.4f}")

    stem = 'difficulty_sweep_mirror' if args.mirror else 'difficulty_sweep'

    # CSV
    with open(f'{stem}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {stem}.csv")

    # Plot: GA wall-clock and generations-to-converge vs difficulty
    try:
        import matplotlib.pyplot as plt
        xs = [r['difficulty'] for r in rows]
        fig, ax1 = plt.subplots()
        ax1.plot(xs, [r['t_ga_s'] for r in rows], 'o-', color='tab:blue',
                 label='GA wall-clock (s)')
        ax1.set_xlabel('Difficulty (x n^2 gate baseline)')
        ax1.set_ylabel('GA wall-clock (s)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(xs, [r['gens_to_converge'] for r in rows], 's--', color='tab:red',
                 label='generations to converge')
        ax2.set_ylabel('Generations to converge', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        title_kind = 'mirrored ' if args.mirror else ''
        plt.title(f'Difficulty sweep ({title_kind}Clifford+T) — n={n}, '
                  f'pop={args.pop}, epochs={args.epochs}')
        fig.tight_layout()
        plt.savefig(f'{stem}.png', dpi=150)
        print(f"Saved {stem}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
