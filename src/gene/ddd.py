# DDD: Dynamic Digital Decoupling
#  Circuit -> Circuit
#  adds involutive pulse sequences for idle qubits

from .base import BaseGene
from mitiq import ddd
from numpy import random

class DDDGene(BaseGene):
	rules = [
		ddd.rules.xx,
		ddd.rules.xyxy,
		ddd.rules.yy,
	]

	def __str__(self):
		return f'ddd({self.rule.__name__})'

	def __init__(self, rule, obs):
		super().__init__()
		self.rule = rule
		self.obs = obs

	def executor(self, executable):
		return ddd.mitigate_executor(
			executable,
			rule=self.rule,
			observable=self.obs
		)

def mutate(gene):
	gene.rule = random.choice(DDDGene.rules)