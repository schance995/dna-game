"""Exact density-matrix check: does inserting DDD change <Z...Z> under the execute() noise model?"""
import cirq
import numpy as np
from mitiq import ddd
from mitiq.benchmarks import generate_random_clifford_t_circuit


def noise(c):
    n = c.with_noise(cirq.depolarize(0.01))
    for ch in [cirq.bit_flip(.05), cirq.phase_flip(.05), cirq.amplitude_damp(.05)]:
        n.append(ch.on_each(n.all_qubits()))
    return n


def exp_z(c, noisy=True):
    qs = sorted(c.all_qubits())
    rho = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(noise(c) if noisy else c, qubit_order=qs).final_density_matrix
    return cirq.DensePauliString("Z" * len(qs)).on(*qs).expectation_from_density_matrix(rho, {q: i for i, q in enumerate(qs)}).real


q = cirq.LineQubit.range(2)
toy = cirq.Circuit([cirq.Moment([cirq.H(q[0]), cirq.H(q[1])]), cirq.Moment([cirq.T(q[1])]), cirq.Moment([cirq.T(q[1])]),
                    cirq.Moment([cirq.S(q[1])]), cirq.Moment([cirq.H(q[0]), cirq.H(q[1])])])
print("Toy circuit with noise (q0 idle in moments 1-3):")
print(noise(toy))
mir = generate_random_clifford_t_circuit(num_qubits=5, num_oneq_cliffords=12, num_twoq_cliffords=12, num_t_gates=12, seed=0)
mir = mir + cirq.inverse(mir)
for name, c in [("toy", toy), ("mirrored 5q", mir)]:
    base = exp_z(c)
    for rule in [ddd.rules.xx, ddd.rules.yy, ddd.rules.xyxy]:
        d = ddd.insert_ddd_sequences(c, rule)
        added = sum(1 for _ in d.all_operations()) - sum(1 for _ in c.all_operations())
        print(f"{name:12s} {rule.__name__:5s} moments {len(c)}->{len(d)}  +{added} gates  "
              f"<Z..Z> no DDD {base:.10f}  with DDD {exp_z(d):.10f}  ideal {exp_z(c, False):.3f}")
