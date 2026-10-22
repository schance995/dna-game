"""
Verify the metric-saturation diagnosis from the 20260610 difficulty sweep.

Hypothesis: as gate count grows, the ideal expectation <Z^n> of a random
Clifford+T circuit collapses toward 0, below the 1000-shot noise floor
(~1/sqrt(1000) ~= 0.032). When |ideal - noisy| is that small, the fitness
ratio |mitigated - ideal| / |noisy - ideal| degenerates and candidates hit
fitness 1.0 by shot luck rather than by mitigating anything.

Prints ideal and noisy expectation values per difficulty level, averaged
over several executor repetitions to show the spread.
"""
import sys, importlib.util, statistics

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

N_QUBITS = 7
SEED = 42
REPS = 5
SHOT_FLOOR = 1 / 1000 ** 0.5

obs = m.Observable(m.PauliString("Z" * N_QUBITS))

print(f"n={N_QUBITS}, seed={SEED}, {REPS} reps per estimate, "
      f"shot-noise floor ~ {SHOT_FLOOR:.3f}\n")
print(f"{'diff':>5} {'gates':>6} {'ideal (mean+-sd)':>22} {'noisy (mean+-sd)':>22} {'|ideal-noisy|':>14} {'saturated?':>10}")

for k in [1, 2, 4, 8]:
    gates = k * N_QUBITS ** 2
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=N_QUBITS,
        num_oneq_cliffords=gates,
        num_twoq_cliffords=gates,
        num_t_gates=gates,
        seed=SEED,
    )
    ideals = [float(m.run_pure(circuit, obs).real) for _ in range(REPS)]
    noisys = [float(m.run_noise(circuit, obs).real) for _ in range(REPS)]
    i_mean, i_sd = statistics.mean(ideals), statistics.stdev(ideals)
    n_mean, n_sd = statistics.mean(noisys), statistics.stdev(noisys)
    gap = abs(i_mean - n_mean)
    saturated = "YES" if gap < SHOT_FLOOR else "no"
    print(f"x{k:<4} {gates:>6} {i_mean:>+12.4f} +- {i_sd:<6.4f} "
          f"{n_mean:>+12.4f} +- {n_sd:<6.4f} {gap:>14.4f} {saturated:>10}")
