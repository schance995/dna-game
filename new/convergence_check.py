import sys, importlib.util, matplotlib.pyplot as plt

sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location("dna_game_2", "dna-game-2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if __name__ == '__main__':
    n_qubits = 7
    seed = 42
    obs = m.Observable(m.PauliString("Z" * n_qubits))
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=n_qubits,
        num_oneq_cliffords=n_qubits,
        num_twoq_cliffords=n_qubits,
        num_t_gates=n_qubits,
        seed=seed,
    )

    pops, fits = m.genetic_algorithm_2(
        pop_size=40,
        n_qubits=n_qubits,
        obs=obs,
        circuit=circuit,
        epochs=10,
    )

    max_fits  = [max(f) for f in fits]
    mean_fits = [sum(f) / len(f) for f in fits]

    plt.plot(max_fits, label='max fitness')
    plt.plot(mean_fits, label='mean fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('GA Convergence — n=7, Random Clifford-T, pop=40, gen=10')
    plt.legend()
    plt.savefig('convergence.png', dpi=150)
    print("Saved convergence.png")
    print("Max fitness per generation:", [f"{v:.4f}" for v in max_fits])
    print("Mean fitness per generation:", [f"{v:.4f}" for v in mean_fits])
