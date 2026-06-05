# ZNE: Zero Noise Extrapolation
#  Circuit -> [Circuit] -> Readout + Confidence Interval
#  varies the circuit to extrapolate a noise-corrected readout

from .base import BaseGene
from mitiq import zne

import numpy as np
from numpy import random

class ZNEGene(BaseGene):
    scale_noises = [
        zne.scaling.fold_global,
        zne.scaling.fold_all,
        # zne.scaling.fold_gates_from_left,
        # zne.scaling.fold_gates_from_right,
        zne.scaling.fold_gates_at_random,
        # zne.scaling.insert_id_layers,
        # zne.scaling.layer_folding,
    ]

    factories = [
        zne.inference.RichardsonFactory(scale_factors=[1, 3, 5]),
        zne.inference.LinearFactory(scale_factors=[1, 3]),
        zne.inference.PolyFactory(scale_factors=[1, 1.5, 2, 2.5, 3], order=2),
        zne.inference.ExpFactory(scale_factors=[1, 2, 3], asymptote=0.5),  # may fail to converge
        zne.inference.AdaExpFactory(steps=5, asymptote=0.5),  # may fail to converge
    ]

    @staticmethod
    def generate_factory():
        scale_factors = np.sort(
            random.uniform(
                low  = 1.0,
                high = 10.0,
                size = random.randint(2, 7),
            )
        )
        match random.randint(5):
            case 0: # Richardson
                return zne.inference.RichardsonFactory(
                    scale_factors = scale_factors
                )
            case 1: # Linear
                return zne.inference.LinearFactory(
                    scale_factors = scale_factors
                )
            case 2: # Poly
                return zne.inference.PolyFactory(
                    scale_factors = scale_factors,
                    order = random.randint(len(scale_factors))
                )
            case 3: # Exp
                return zne.inference.ExpFactory(
                    scale_factors = scale_factors,
                    asymptote = random.uniform(
                        low  = -10.0,
                        high = +10.0,
                    )
                )
            case 4: # AdaExp
                match random.randint(2):
                    case 0:
                        # without asymptote
                        return zne.inference.AdaExpFactory(
                            steps = random.randint(4, 15),
                        )
                    case 1:
                        # with asymptote
                        return zne.inference.AdaExpFactory(
                            steps = random.randint(3, 15),
                            asymptote = random.uniform(
                                low  = -10.0,
                                high = +10.0,
                            )
                        )

    def __str__(self):
        factory_name = self.factory.__class__.__name__
        parameters = ''
        match self.factory:
            case zne.inference.RichardsonFactory(_scale_factors=scale_factors):
                parameters = f'{scale_factors}'
            case zne.inference.LinearFactory(_scale_factors=scale_factors):
                parameters = f'{scale_factors}'
            case zne.inference.PolyFactory(_scale_factors=scale_factors, _options=options):
                parameters = f'{scale_factors}, {options["order"]}'
            case zne.inference.ExpFactory(_scale_factors=scale_factors, _options=options):
                parameters = f'{scale_factors}, {options["asymptote"]}'
            case zne.inference.AdaExpFactory(_steps=steps, asymptote=asymptote):
                parameters = f'{steps}, {asymptote}'
        return f'zne({factory_name}({parameters}), {self.scale_noise.__name__})'

    def __init__(self, factory, scale_noise, num_to_avg, obs):
        super().__init__()
        self.factory = factory
        self.scale_noise = scale_noise
        self.num_to_avg = num_to_avg
        self.obs = obs

    def executor(self, executable):
        return zne.mitigate_executor(
            executable,
            observable=self.obs,
            factory=self.factory,
            scale_noise=self.scale_noise,
            num_to_average=self.num_to_avg,
        )

def mutate(gene):
	match random.randint(3):
		case 0:
			gene.scale_noise = random.choice(ZNEGene.scale_noises)
		case 1:
			gene.factory = ZNEGene.generate_factory()
		case 2:
			match gene.factory:
				case zne.inference.RichardsonFactory(_scale_factors=scale_factors):
					gene.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
				case zne.inference.LinearFactory(_scale_factors=scale_factors):
					gene.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
				case zne.inference.PolyFactory(_scale_factors=scale_factors, _order=order):
					gene.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
				case zne.inference.ExpFactory(_scale_factors=scale_factors, _asymptote=asymptote):
					gene.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
				case zne.inference.AdaExpFactory(_steps=steps, _asymptote=asymptote):
					gene.factory.steps = gene.factory.steps + random.randint(3) - 1