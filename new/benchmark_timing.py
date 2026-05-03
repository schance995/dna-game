import sys, importlib.util, time, statistics

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

n_qubits = 7
obs = m.Observable(m.PauliString("Z" * n_qubits))
circuit = m.generate_random_clifford_t_circuit(
    num_qubits=n_qubits,
    num_oneq_cliffords=n_qubits,
    num_twoq_cliffords=n_qubits,
    num_t_gates=n_qubits,
)

chromosomes = {
    "REM+DDD": [
        m.REMGene(p0=0.05, p1=0.05, n_qubits=n_qubits),
        m.DDDGene(rule=m.ddd.rules.xx, obs=obs),
    ],
    "REM+ZNE": [
        m.REMGene(p0=0.05, p1=0.05, n_qubits=n_qubits),
        m.ZNEGene(factory=m.ZNEGene.factories[0], scale_noise=m.ZNEGene.scale_noises[0], num_to_avg=1, obs=obs),
    ],
}

N = 3
medians = {}
for name, chrom in chromosomes.items():
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        m.evaluate_fitness(chrom, circuit, obs)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    medians[name] = med
    print(f"{name}: {[round(t, 2) for t in times]} → median {med:.2f}s")

def total_runtime(t_eval, pop, gen, n_circuits=4, n_qubits_counts=3, n_seeds=3, workers=16):
    runs = n_circuits * n_qubits_counts * n_seeds
    return (pop * gen / workers) * t_eval * runs / 60

t = medians["REM+ZNE"]
print(f"\nt_eval = {t:.1f}s")
for pop in [40, 20, 10]:
    mins = total_runtime(t, pop, gen=10)
    print(f"  pop={pop}, gen=10 → ~{mins:.0f} min total")
