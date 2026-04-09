"""
Smoke test: runs benchmark() on a single GHZ circuit with a hardcoded chromosome.
Skips the GA entirely. Use this to verify the noise model, execute(), and benchmark() work.
"""
import sys, importlib.util
sys.path.insert(0, '.')

spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

n_qubits = 5
obs = m.Observable(m.PauliString("Z" * n_qubits))
circuit = m.generate_ghz_circuit(n_qubits)
chrom = [
    m.REMGene(p0=0.05, p1=0.05, n_qubits=n_qubits),
    m.DDDGene(rule=m.ddd.rules.xx, obs=obs),
]

print("Running benchmark smoke test...")
fitnesses, ratios = m.benchmark(chrom, circuit, n_qubits, obs)
print("Fitnesses:", {k: f"{v:.4f}" for k, v in fitnesses.items()})
print("Ratios:   ", {k: f"{v:.4f}" for k, v in ratios.items()})
print("PASS")
