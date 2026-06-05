# REM: Readout Error Mitigation
#  Circuit -> Result & Confidence Interval
#  resolves measurement error using calibration data + classical statistics

from .base import BaseGene
from mitiq import rem
from numpy import random

class REMGene(BaseGene):
    def __str__(self) -> str:
        return f'rem({self.p0:.2f}, {self.p1:.2f})'

    def __init__(self, p0=0.05, p1=0.05, n_qubits=5):
        super().__init__()
        self.p0 = p0
        self.p1 = p1
        self.n_qubits = n_qubits

    def executor(self, executable):
        icm = rem.generate_inverse_confusion_matrix(self.n_qubits, self.p0, self.p1)
        return rem.mitigate_executor(executable, inverse_confusion_matrix=icm)

def mutate(gene):
	match random.randint(3):
		case 0:
			pass
		case 1:
			gene.p0 += 0.01 * random.randn()
		case 2:
			gene.p1 += 0.01 * random.randn()
	gene.p0 = max(0, min(gene.p0, 1))
	gene.p1 = max(0, min(gene.p1, 1))