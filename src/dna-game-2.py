#!/usr/bin/env python
# coding: utf-8

# # TODO
# 
# Code:
# 
# - Genetic algorithm should evolve with a single loop
# - Genetic algorithm should record the entire population and fitness after each generation.
# - Statistics can be recorded after the fact, not during the calculation.
# 
# New pseudocode: (with proper parameterization)
# 
# ```
# def genetic_algorithm():
#     pop = init_pop()
#     pops = [pop]
#     for e in epochs:
#         pop = evolve(pop)
#         pops.append(pop)
#     return pops
# 
# def evolve(pop):
#     do stuff
#     return pop
# 
# def run_experiment(n_qubits, seed):
#     pops = genetic_algorithm()
#     save pops to a file
# 
# def run_experiments():
#     for q in n_qubits:
#         for seed in seeds:
#             run_experiment(q, seed)
# ```
# 
# Dataset format should look like this:
# 
# ```
# n_qubits | seed | circuit | epoch | population | fitnesses | mitigation ratios
# int | int | str | int | list[str] | list[float] | list[float]
# ```
# 
# That is, each line records a single population at a single epoch, its hyperparameters, and its scores.
# 
# Plots should include:
# - Barplot of mitigation ratios over methods / qubits
# - How often each mitigation was used
# 
# Future work:
# - ensemble methods for mitigation
# - additional hyperparameter tuning
# - additional pipelines
# - additional evolutionary schemes
# - run on real quantum hardware
# - more noise models
# - other realistic circuits
# - integration into Mitiq
# - consider compute time in the mitigation performance

# In[2]:


# utils
import sys, time, gc, math, os
from pathlib import Path
from copy import deepcopy
from abc import *
from functools import partial

# parallelism
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm, trange

# data science
import pandas as pd
import numpy as np
from numpy import random as random
from scipy.special import expit, logit
from matplotlib import pyplot as plt

# quantum
import cirq
from mitiq import Observable, PauliString, MeasurementResult
from mitiq import raw, rem, zne, ddd

# test
from mitiq.benchmarks import generate_rb_circuits, generate_ghz_circuit, generate_w_circuit
from mitiq.benchmarks.randomized_clifford_t_circuit import generate_random_clifford_t_circuit
from mitiq.benchmarks.mirror_qv_circuits import generate_mirror_qv_circuit

# parallel
from joblib import delayed, Parallel

# warnings
import warnings
from numpy.exceptions import ComplexWarning


# In[3]:


# globals
SIMULATOR = cirq.DensityMatrixSimulator()
plt.rcParams['figure.constrained_layout.use'] = True


# In[4]:


class BaseGene(ABC):
    def __str__(self):
        return f'base'

    def __repr__(self):
        return str(self)

    def __init__(self):
        pass

    @abstractmethod
    def executor(self, executable):
        pass


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


def _strip_measurements(circuit: cirq.Circuit) -> cirq.Circuit:
    ''' Remove terminal measurement moment if present, so mitiq can add its own. '''
    if circuit and any(cirq.is_measurement(op) for op in circuit[-1].operations):
        return circuit[:-1]
    return circuit

def run_pure(circuit, obs):
    ''' pure-state execution '''
    return raw.execute(_strip_measurements(circuit), partial(execute, noise_level=0, p0=0), obs)

def run_noise(circuit, obs):
    ''' simulated noisy qpu '''
    return raw.execute(_strip_measurements(circuit), execute, obs)


# In[9]:


def execute(circuit: cirq.Circuit, noise_level: float = 0.01, p0: float = 0.05) -> MeasurementResult:
    '''
    run circuit with depolarization/noise, and measure results
    noise_level: strength of depolarization noise
    p0: used to suppress all noise channels when set to 0 (pure baseline)
    '''
    bit_flip_p   = 0.05 if p0 > 0 else 0
    phase_flip_p = 0.05 if p0 > 0 else 0
    amp_damp_p   = 0.05 if p0 > 0 else 0

    measurements = circuit[-1]
    circuit = circuit[:-1]
    circuit = circuit.with_noise(cirq.depolarize(noise_level))
    circuit.append(cirq.bit_flip(bit_flip_p).on_each(circuit.all_qubits()))
    circuit.append(cirq.phase_flip(phase_flip_p).on_each(circuit.all_qubits()))
    circuit.append(cirq.amplitude_damp(amp_damp_p).on_each(circuit.all_qubits()))
    circuit.append(measurements)

    result = SIMULATOR.run(circuit, repetitions=1000)
    bitstrings = np.column_stack(list(result.measurements.values()))
    return MeasurementResult(bitstrings)


# In[10]:


def mitigate(circuit: cirq.Circuit, chromosome, executor):
    '''
    maps a chromosome to a mitigation chain and execute it.
    execute: circuit -> QuantumResult
    '''

    for gene in chromosome:
        executor = gene.executor(executor)
    return executor(_strip_measurements(circuit))


# In[11]:
def compute_mitigation_ratio(pure, noise, mitigation):
    if noise == pure:
        return math.inf
    return abs((mitigation - pure) / (noise - pure))

def ratio_to_fitness(ratio):
    return expit(-np.log(ratio))

def compute_fitness(pure, noise, mitigation):

    # bound fitness between [0, 1]
    ratio = compute_mitigation_ratio(pure, noise, mitigation)
    return ratio_to_fitness(ratio)


# In[12]:


def evaluate_fitness(candidate, circuit, obs):
    """
    Evaluate mitigation performance of a candidate configuration on a circuit
    - fitness: relative gain in mitigation
    - higher fitness: difference between noisy and ideal > mitigated and ideal
    - ideal noise: is as far away as possible
    - maximize negative sigmoid log ratio of differences
    """
    try:
        return compute_fitness(
            run_pure(circuit, obs),
            run_noise(circuit, obs),
            mitigate(circuit, candidate, execute)
        )
    except Exception as e:
        # really bad if it doesn't work; can't log worst configurations
        # known causes:
        # - zne.inference.ExtrapolationError
        # raise e
        # print('Error')
        # print(e)
        return 0 # -1e8


# In[13]:


def mutate(chromosome):
    i = random.randint(len(chromosome))
    c = chromosome[i]
    # if no change happened, consider it as a no-op
    match c:
        case REMGene(p0=p0, p1=p1):
            match random.randint(3):
                case 0:
                    pass
                case 1:
                    c.p0 += 0.01 * random.randn()
                case 2:
                    c.p1 += 0.01 * random.randn()
            c.p0 = np.clip(c.p0, 0, 1)
            c.p1 = np.clip(c.p1, 0, 1)
        case ZNEGene(scale_noise=scale_noise):
            match random.randint(3):
                case 0:
                    c.scale_noise = random.choice(ZNEGene.scale_noises)
                case 1:
                    c.factory = ZNEGene.generate_factory()
                case 2:
                    match c.factory:
                        case zne.inference.RichardsonFactory(_scale_factors=scale_factors):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.LinearFactory(_scale_factors=scale_factors):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.PolyFactory(_scale_factors=scale_factors, _order=order):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.ExpFactory(_scale_factors=scale_factors, _asymptote=asymptote):
                            c.factory.scale_factors = random.randint(max(scale_factors) * 2, size=len(scale_factors))
                        case zne.inference.AdaExpFactory(_steps=steps, _asymptote=asymptote):
                            c.factory.steps = c.factory.steps + random.randint(3) - 1
        case DDDGene(rule=rule):
            c.rule = random.choice(DDDGene.rules)
    return chromosome

def crossover(population, times=None):
    if times is None:
        times = len(population) // 2

    # individual crossover
    def cross(x, y, i=0):
        ix = random.randint(len(x))
        iy = random.randint(len(y))
        # right now, we can only swap matching gene types
        if type(x[ix]) == type(y[iy]):
            t     = x[ix]
            x[ix] = y[iy]
            y[iy] = t
        else:
            # failed, try again up to 10 times
            if i < 10:
                cross(x, y, i+1)

    # do an arbitrary number of crossover iterations
    for _ in range(times):
        ix, iy = random.randint(len(population), size=2)  # need to fix random sed
        cross(population[ix], population[iy], 0)

    return population


# In[14]:


# hardcoded population starting points
def initialize_population(population_size, n_qubits, obs):
    '''
    Initializes the population chromosome configurations:
    - REM + DDD
    - REM + ZNE.

    NOTE: Limited functionality due to upstream API design.
    '''

    def chain_1():
        return [
            REMGene(p0 = 0.05, p1 = 0.05, n_qubits = n_qubits),
            DDDGene(
                rule = random.choice(DDDGene.rules),
                obs = obs,
            ),
        ]

    def chain_2():
        return [
            REMGene(p0 = 0.05, p1 = 0.05, n_qubits = n_qubits),
            ZNEGene(
                factory = random.choice(ZNEGene.factories),
                scale_noise = random.choice(ZNEGene.scale_noises),
                num_to_avg = 1,
                obs = obs,
            ),
        ]

    def make_chain():
        known_working = [chain_1, chain_2]
        return known_working[random.randint(len(known_working))]()

    return [
        make_chain()
        for _ in range(population_size)
    ]


# In[22]:


def get_fitness(pop, circuit, obs, pool):
    def get_res(f, indiv):
        try:
            return f.result()
        except Exception:
            return evaluate_fitness(indiv, circuit, obs)

    try:
        futures = [pool.submit(evaluate_fitness, indiv, circuit, obs) for indiv in pop]
        fitnesses = [get_res(f, indiv) for f, indiv in tqdm(zip(futures, pop), total=len(pop))]
    except Exception:
        fitnesses = [evaluate_fitness(indiv, circuit, obs) for indiv in tqdm(pop)]

    pop_and_fit = sorted(zip(pop, fitnesses), key=lambda v: -v[1])
    return [p[0] for p in pop_and_fit], [p[1] for p in pop_and_fit]


# In[23]:


def repop(pop):
    # repopulation
    half = len(pop) // 2
    rest = len(pop) - half # account for odd-length population sizes
    pop = [
        [
            deepcopy(j)
            for j in i
        ]
        for i in pop[:half] + pop[:rest]
    ]
    return pop


# In[24]:


def genetic_algorithm_2(pop_size, n_qubits, obs, circuit, epochs):
    '''
    Optimize a population of 'pop size' on 'circuit' for 'epochs' generations.
    '''
    ctx = get_context('spawn')
    with ProcessPoolExecutor(mp_context=ctx, max_workers=16) as pool:
        pop = initialize_population(pop_size, n_qubits, obs)
        pop, this_fits = get_fitness(pop, circuit, obs, pool)
        pops = [pop]
        fits = [this_fits]
        for _ in trange(epochs, desc='genetic algorithm', unit='generation'):
            pop = [mutate(i) for i in pop]
            pop = crossover(pop, times=1)
            pop = repop(pop)
            pop, this_fits = get_fitness(pop, circuit, obs, pool)
            pops.append(pop)
            fits.append(this_fits)
    return pops, fits


# In[25]:


def generate_circuits(n_qubits, seed):
    qubits = cirq.LineQubit.range(n_qubits)
    qft_circuit = cirq.Circuit(
        cirq.QuantumFourierTransformGate(n_qubits).on(*qubits),
        cirq.measure(*qubits),
    )
    return [
        generate_ghz_circuit(n_qubits),
        generate_w_circuit(n_qubits),
        generate_random_clifford_t_circuit(
            num_qubits=n_qubits,
            num_oneq_cliffords=n_qubits,
            num_twoq_cliffords=n_qubits,
            num_t_gates=n_qubits,
            seed=seed,
        ),
        qft_circuit,
    ]

def benchmark(fittest_chromosome, circuit, n_qubits, obs):
    print("\n========= BENCHMARKING RESULTS ========")

    ideal_measurement = run_pure(circuit, obs)
    noisy_measurement = run_noise(circuit, obs)
    #print("Ideal value:", "{:.5f}".format(ideal_measurement.real))
    #print("Noisy value:", "{:.5f}".format(noisy_measurement.real))

    icm = rem.generate_inverse_confusion_matrix(n_qubits, 0.05, 0.05) # arbitrary config
    rem_executor = rem.mitigate_executor(execute, inverse_confusion_matrix=icm)

    stripped = _strip_measurements(circuit)
    rem_result = obs.expectation(stripped, rem_executor)
    #print("Mitigated value obtained with REM:", "{:.5f}".format(rem_result.real))

    zne_result = zne.execute_with_zne(stripped, execute, obs) # default params
    #print("Mitigated value obtained with ZNE:", "{:.5f}".format(zne_result.real))

    ddd_result = ddd.execute_with_ddd(stripped, execute, obs, rule = ddd.rules.xx) # default params
    #print("Mitigated value obtained with DDD:", "{:.5f}".format(ddd_result.real))

    try:
        optim_result = mitigate(circuit, fittest_chromosome, execute)
    except Exception:
        optim_result = noisy_measurement
    #print("Optim mitigated value:", "{:.5f}".format(optim_result.real))

    # compute mitigation ratios of each
    rem_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, rem_result)
    zne_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, zne_result)
    ddd_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, ddd_result)
    dna_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, optim_result)
    ratios = {
        'REM': rem_ratio,
        'ZNE': zne_ratio,
        'DDD': ddd_ratio,
        'DNA': dna_ratio,
    }
    # print('REM ratio: {:.5f}'.format(rem_ratio))
    # print('ZNE ratio: {:.5f}'.format(zne_ratio))
    # print('DDD ratio: {:.5f}'.format(ddd_ratio))
    # print('DNA ratio: {:.5f}'.format(dna_ratio))

    rem_fitness = ratio_to_fitness(rem_ratio)
    zne_fitness = ratio_to_fitness(zne_ratio)
    ddd_fitness = ratio_to_fitness(ddd_ratio)
    dna_fitness = ratio_to_fitness(dna_ratio)
    fitnesses = {
        'REM': rem_fitness,
        'ZNE': zne_fitness,
        'DDD': ddd_fitness,
        'DNA': dna_fitness,
    }
    # print('REM fitness: {:.5f}'.format(rem_fitness))
    # print('ZNE fitness: {:.5f}'.format(zne_fitness))
    # print('DDD fitness: {:.5f}'.format(ddd_fitness))
    # print('DNA fitness: {:.5f}'.format(dna_fitness))
    return fitnesses, ratios

# In[26]:



def grid_search(n_qubits, obs, circuit, sequence):
    """
    Enumerate all discrete parameter combinations for the given sequence type.
    sequence: 'REM+DDD' | 'REM+ZNE'
    Budget: same as GA = 400 evaluations.
    Returns (best_chromosome, results_rows).
    """
    from itertools import product as iproduct

    rem_p_vals = [0.01, 0.05, 0.1, 0.2]

    if sequence == 'REM+DDD':
        configs = [
            [REMGene(p0=p0, p1=p1, n_qubits=n_qubits), DDDGene(rule=rule, obs=obs)]
            for p0, p1, rule in iproduct(rem_p_vals, rem_p_vals, DDDGene.rules)
        ]
    else:  # REM+ZNE
        configs = [
            [REMGene(p0=p0, p1=p1, n_qubits=n_qubits),
             ZNEGene(factory=factory, scale_noise=scale_noise, num_to_avg=1, obs=obs)]
            for p0, p1, factory, scale_noise in iproduct(
                rem_p_vals, rem_p_vals, ZNEGene.factories, ZNEGene.scale_noises
            )
        ]

    ctx = get_context('spawn')
    with ProcessPoolExecutor(mp_context=ctx, max_workers=16) as pool:
        futures = [pool.submit(evaluate_fitness, chrom, circuit, obs) for chrom in configs]
        fitnesses = []
        for f, chrom in tqdm(zip(futures, configs), total=len(configs)):
            try:
                fitnesses.append(f.result())
            except Exception:
                fitnesses.append(evaluate_fitness(chrom, circuit, obs))

    results_rows = [
        {
            'method': 'grid_search',
            'sequence': str(chrom),
            'circuit': None,   # filled in by caller
            'n_qubits': n_qubits,
            'seed': None,
            'epoch': 0,
            'fitness': fitness,
            'mitigation_ratio': float(np.exp(-logit(fitness))) if 0 < fitness < 1 else math.inf,
            'chromosome_str': str(chrom),
        }
        for chrom, fitness in zip(configs, fitnesses)
    ]

    best_idx = int(np.argmax(fitnesses))
    return configs[best_idx], results_rows


def random_search(n_qubits, obs, circuit, sequence, n_samples=400):
    """
    Sample n_samples configs uniformly at random from the continuous parameter space.
    sequence: 'REM+DDD' | 'REM+ZNE'
    Returns (best_chromosome, results_rows).
    """
    if sequence == 'REM+DDD':
        configs = [
            [
                REMGene(p0=float(random.uniform(0, 1)), p1=float(random.uniform(0, 1)), n_qubits=n_qubits),
                DDDGene(rule=random.choice(DDDGene.rules), obs=obs),
            ]
            for _ in range(n_samples)
        ]
    else:  # REM+ZNE
        configs = [
            [
                REMGene(p0=float(random.uniform(0, 1)), p1=float(random.uniform(0, 1)), n_qubits=n_qubits),
                ZNEGene(
                    factory=ZNEGene.generate_factory(),
                    scale_noise=random.choice(ZNEGene.scale_noises),
                    num_to_avg=1,
                    obs=obs,
                ),
            ]
            for _ in range(n_samples)
        ]

    ctx = get_context('spawn')
    with ProcessPoolExecutor(mp_context=ctx, max_workers=16) as pool:
        futures = [pool.submit(evaluate_fitness, chrom, circuit, obs) for chrom in configs]
        fitnesses = []
        for f, chrom in tqdm(zip(futures, configs), total=len(configs)):
            try:
                fitnesses.append(f.result())
            except Exception:
                fitnesses.append(evaluate_fitness(chrom, circuit, obs))

    results_rows = [
        {
            'method': 'random_search',
            'sequence': str(chrom),
            'circuit': None,
            'n_qubits': n_qubits,
            'seed': None,
            'epoch': 0,
            'fitness': fitness,
            'mitigation_ratio': float(np.exp(-logit(fitness))) if 0 < fitness < 1 else math.inf,
            'chromosome_str': str(chrom),
        }
        for chrom, fitness in zip(configs, fitnesses)
    ]

    best_idx = int(np.argmax(fitnesses))
    return configs[best_idx], results_rows


def experiment(seed, n_qubits, circuit, circuit_name, method='GA'):
    np.random.seed(seed)

    results_rows = []

    if method == 'GA':
        pops, fits = genetic_algorithm_2(pop_size, n_qubits, obs, circuit, generation_count)
        results_rows = [
            {
                'method': 'GA',
                'sequence': str(chrom),
                'circuit': circuit_name,
                'n_qubits': n_qubits,
                'seed': seed,
                'epoch': epoch,
                'fitness': fitness,
                'mitigation_ratio': float(np.exp(-logit(fitness))) if 0 < fitness < 1 else math.inf,
                'chromosome_str': str(chrom),
            }
            for epoch, (pop, epoch_fits) in enumerate(zip(pops, fits))
            for chrom, fitness in zip(pop, epoch_fits)
        ]
        fittest_chromosome = pops[-1][np.argmax(fits[-1])]

    elif method == 'grid_search':
        all_rows = []
        best_chrom, best_fitness = None, -math.inf
        for sequence in ['REM+DDD', 'REM+ZNE']:
            chrom, rows = grid_search(n_qubits, obs, circuit, sequence)
            for row in rows:
                row['circuit'] = circuit_name
                row['seed'] = seed
            all_rows.extend(rows)
            seq_best_fitness = max(r['fitness'] for r in rows)
            if seq_best_fitness > best_fitness:
                best_fitness = seq_best_fitness
                best_chrom = chrom
        results_rows = all_rows
        fittest_chromosome = best_chrom

    elif method == 'random_search':
        all_rows = []
        best_chrom, best_fitness = None, -math.inf
        for sequence in ['REM+DDD', 'REM+ZNE']:
            chrom, rows = random_search(n_qubits, obs, circuit, sequence)
            for row in rows:
                row['circuit'] = circuit_name
                row['seed'] = seed
            all_rows.extend(rows)
            seq_best_fitness = max(r['fitness'] for r in rows)
            if seq_best_fitness > best_fitness:
                best_fitness = seq_best_fitness
                best_chrom = chrom
        results_rows = all_rows
        fittest_chromosome = best_chrom

    # build benchmark rows (defaults + optimized)
    fitnesses, ratios = benchmark(fittest_chromosome, circuit, n_qubits, obs)
    # defaults rows use method='defaults'; DNA row uses the optimizer method name
    benchmark_rows = []
    for bm_method in ['REM', 'ZNE', 'DDD']:
        benchmark_rows.append({
            'method': 'defaults',
            'sequence': bm_method,
            'circuit': circuit_name,
            'n_qubits': n_qubits,
            'seed': seed,
            'fitness': fitnesses[bm_method],
            'mitigation_ratio': ratios[bm_method],
        })
    benchmark_rows.append({
        'method': method,
        'sequence': 'DNA',
        'circuit': circuit_name,
        'n_qubits': n_qubits,
        'seed': seed,
        'fitness': fitnesses['DNA'],
        'mitigation_ratio': ratios['DNA'],
    })

    return results_rows, benchmark_rows

# In[27]:


if __name__ == '__main__':
    now = time.strftime('%Y%m%dT%H%M%S')
    pop_size = 40
    generation_count = 10
    n_qubits = 7
    n_seeds = 3

    all_results = []
    all_benchmarks = []

    methods = ['GA', 'grid_search', 'random_search']

    for n_q in range(5, n_qubits + 1):
        print(n_q)
        circuits = generate_circuits(n_q, 0)
        circuit_names = [
            'GHZ',
            'W-state',
            'Random Clifford T',
            'QFT',
        ]
        obs = Observable(PauliString("Z" * n_q))
        for seed in range(n_seeds):
            print(seed)
            for circuit, circuit_name in zip(circuits, circuit_names):
                print(circuit_name)
                for method in methods:
                    print(method)
                    results_rows, benchmark_rows = experiment(seed, n_q, circuit, circuit_name, method=method)
                    all_results.extend(results_rows)
                    all_benchmarks.extend(benchmark_rows)

    out_dir = Path(f'results/{now}')
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_dir / 'results.csv', index=False)
    pd.DataFrame(all_benchmarks).to_csv(out_dir / 'benchmarks.csv', index=False)
    exit()

"""
# In[ ]:


def genetic_algorithm(pop_size, generation_count, circuit, n_qubits, obs):
    '''
    Optimize a population of 'pop size' on 'circuit' for 'generation_count' generations.
    '''
    pop = initialize_population(pop_size, n_qubits, obs)
    pop_generations_fits = []
    pop_generations_chrs = []
    max_fitness_over_time = []
    mean_fitness_over_time = []
    med_fitness_over_time = []
    max_indivs_over_time = []
    med_indivs_over_time = []
    best_max_fitness_so_far = float('-inf')
    best_mean_fitness_so_far = float('-inf')
    best_med_fitness_so_far = float('-inf')
    ctx = get_context('spawn')
    with ProcessPoolExecutor(mp_context=ctx, max_workers=20) as executor:
        for generation in (pbar := trange(generation_count,
            desc = 'genetic algorithm',
            unit = 'generation',
        )):
            # mutation
            pop = [mutate(i) for i in pop]

            # crossover
            pop = crossover(pop, times=1)

            # fitness testing. Multiprocessing speeds this up
            futures = [executor.submit(evaluate_fitness, indiv, circuit, obs) for indiv in pop]
            fbar = trange(len(futures), leave=False)
            def get_res(f, indiv, circuit):
                fbar.update()
                # execute the exceptions synchronously if unpickleable:
                # AttributeError: Can't pickle local object 'PolyFactory.extrapolate.<locals>.zne_curve'
                try:
                    result = f.result()
                except Exception as e:  # assume that errors are related to multiprocessing
                    # print(e, file=sys.stderr)
                    result = evaluate_fitness(indiv, circuit, obs)
                return result

            fitnesses = [get_res(future, indiv, circuit) for future, indiv in zip(futures, pop)]

            # sort by fitnesses
            pop_fit = sorted(
                zip(pop, fitnesses),
                key = lambda v: -v[1],
            )
            pop_generations_fits.append(p[1] for p in pop_fit)
            pop_generations_chrs.append(str(p[0]) for p in pop_fit)

            # logging
            max_indiv, max_fit = pop_fit[0]
            # array is sorted, use constant-time lookups
            med_indiv, med_fit = pop_fit[len(fitnesses)//2] 
            mean_fit = sum(p[1] for p in pop_fit) / len(pop_fit)

            best_max_fitness_so_far = max(best_max_fitness_so_far, max_fit)
            best_med_fitness_so_far = max(best_med_fitness_so_far, med_fit)
            best_mean_fitness_so_far = max(best_mean_fitness_so_far, mean_fit)

            max_indivs_over_time.append(max_indiv)
            max_fitness_over_time.append(max_fit)
            mean_fitness_over_time.append(mean_fit)
            med_fitness_over_time.append(med_fit)
            med_indivs_over_time.append(med_indiv)

            # re-extract population
            pop = [
                i for (i, f) in pop_fit
            ]

            # logging
            pbar.set_postfix_str(f"Best Mean Fit: {best_mean_fitness_so_far:.3f}, Best Med Fit: {best_med_fitness_so_far:.3f}, Best Max Fit: {best_max_fitness_so_far:.3f}")

            # print_pop(pop)

            # return the final population and metrics once done
            if generation + 1 >= generation_count:
                results = {
                    "max_fitness": max_fitness_over_time,
                    "mean_fitness": mean_fitness_over_time,
                    "med_fitness": med_fitness_over_time,
                    "max_indivs": max_indivs_over_time,
                    "med_indivs": med_indivs_over_time,
                }
                return pop, results, pop_generations_fits, pop_generations_chrs

            # repopulation
            half = len(pop) // 2
            rest = len(pop) - half # account for odd-length population sizes
            pop = [
                [
                    deepcopy(j)
                    for j in i
                ]
                for i in pop[:half] + pop[:rest]
            ]


# In[ ]:


def print_pop(pop):
    for i,j in enumerate(pop):
        print(f'{i+1}. {j}')


# In[ ]:


def make_plot(max_fits, mean_fits, med_fits, title, serial_code):
    fig, ax = plt.subplots()
    ticks = list(range(1, 1 + len(max_fits)))
    ax.plot(ticks, max_fits, label='Max')
    ax.plot(ticks, mean_fits, label='Mean')
    ax.plot(ticks, med_fits, label='Median')
    ax.set(xlabel='Generation', ylabel='Fitness', title=title)
    ax.set_xticks(ticks)
    ax.legend()
    ax.grid()
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f'{title} (run {serial_code}).png')
    plt.close()
    gc.collect()


# In[ ]:


def benchmark_results(fittest_chromosome, circuit, n_qubits, obs):
    print("\n========= BENCHMARK RESULTS ========")


    ideal_measurement = ideal(circuit, obs)
    noisy_measurement = noisy(circuit, obs)
    print("Ideal value:", "{:.5f}".format(ideal_measurement.real))
    print("Noisy value:", "{:.5f}".format(noisy_measurement.real))

    icm = rem.generate_inverse_confusion_matrix(n_qubits, 0.05, 0.05) # arbitrary config
    rem_executor = rem.mitigate_executor(execute, inverse_confusion_matrix=icm)

    rem_result = obs.expectation(circuit, rem_executor)
    print("Mitigated value obtained with REM:", "{:.5f}".format(rem_result.real))

    zne_result = zne.execute_with_zne(circuit, execute, obs) # default params
    print("Mitigated value obtained with ZNE:", "{:.5f}".format(zne_result.real))

    ddd_result = ddd.execute_with_ddd(circuit, execute, obs, rule = ddd.rules.xx) # default params
    print("Mitigated value obtained with DDD:", "{:.5f}".format(ddd_result.real))

    optim_result = mitigated(fittest_chromosome, circuit, execute)
    print("Optim mitigated value:", "{:.5f}".format(optim_result.real))

    # compute mitigation ratios of each
    rem_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, rem_result)
    zne_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, zne_result)
    ddd_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, ddd_result)
    dna_ratio = compute_mitigation_ratio(ideal_measurement, noisy_measurement, optim_result)
    ratios = {
        'REM': rem_ratio,
        'ZNE': zne_ratio,
        'DDD': ddd_ratio,
        'DNA': dna_ratio,
    }
    print('REM ratio: {:.5f}'.format(rem_ratio))
    print('ZNE ratio: {:.5f}'.format(zne_ratio))
    print('DDD ratio: {:.5f}'.format(ddd_ratio))
    print('DNA ratio: {:.5f}'.format(dna_ratio))

    rem_fitness = compute_fitness(ratio=rem_ratio)
    zne_fitness = compute_fitness(ratio=zne_ratio)
    ddd_fitness = compute_fitness(ratio=ddd_ratio)
    dna_fitness = compute_fitness(ratio=dna_ratio)
    fitnesses = {
        'REM': rem_fitness,
        'ZNE': zne_fitness,
        'DDD': ddd_fitness,
        'DNA': dna_fitness,
    }
    print('REM fitness: {:.5f}'.format(rem_fitness))
    print('ZNE fitness: {:.5f}'.format(zne_fitness))
    print('DDD fitness: {:.5f}'.format(ddd_fitness))
    print('DNA fitness: {:.5f}'.format(dna_fitness))
    return fitnesses, ratios


# In[ ]:


def run_experiment(circuits, circuit_names, n_qubits, obs, serial_code):

    benchmark_fitnesses = [] # {circuit_name: [] for circuit_name in circuit_names}
    benchmark_ratios = [] # {circuit_name: [] for circuit_name in circuit_names}
    with open(f"output_{serial_code}.txt", "a") as f:
        with redirect_stdout(f): # this is a temporary workaround for logging because we couldn't get the logging module working. This current redirects ALL stdout to the file so it could capture random unintended outputs.
            print(f"\n\n\n############### EXPERIMENT {serial_code} ({time.asctime()}) ############")

            for circuit, circuit_name in zip(circuits, circuit_names):
                title = f'{circuit_name} with {n_qubits} qubits and seed {seed}'
                print(f'\n\nCIRCUIT: {title}')
                print(circuit)  # SVG cirq drawings are unreliable
                final_pop, results, pop_generations_fits, pop_generations_chrs = genetic_algorithm(pop_size, generation_count, circuit, n_qubits, obs)
                max_indivs = results['max_indivs']
                med_indivs = results['med_indivs']
                max_fits = results['max_fitness']
                mean_fits = results['mean_fitness']
                med_fits = results['med_fitness']
                print('Final pop')
                print_pop(final_pop)
                print('Max pop')
                print_pop(max_indivs)
                print('Med pop')
                print_pop(med_indivs)
                # TODO: log all fitnesses
                # make_plot(max_fits, mean_fits, med_fits, title, serial_code)

                # use the best max individual
                best_max_indiv = max_indivs[np.argmax(max_fits)]
                print('Best max individual')
                print(best_max_indiv)
                fitnesses, ratios = benchmark_results(best_max_indiv, circuit, n_qubits, obs)
                # convert dictionaries to dataframes
                benchmark_fitnesses.append(pd.DataFrame(fitnesses, index=[circuit_name]))
                benchmark_ratios.append(pd.DataFrame(ratios, index=[circuit_name]))
                experiment_fitnesses = pd.DataFrame(pop_generations_fits)
                experiment_chromosomes = pd.DataFrame(pop_generations_chrs, index=list(range(generation_count)))
                experiment_fitnesses = pd.DataFrame(pop_generations_chrs, index=list(range(generation_count)))
    return pd.concat(benchmark_fitnesses), pd.concat(benchmark_ratios)


# In[ ]:


def plot_benchmarks(benchmark_fitnesses, benchmark_ratios, n_qubits, serial_code):

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    fitness_df = pd.concat(benchmark_fitnesses)
    mean_fitness_df = fitness_df.groupby(fitness_df.index).mean()
    # std_fitness_df = fitness_df.groupby(fitness_df.index).std()
    ax = mean_fitness_df.plot.bar(rot=0) # , yerr=std_fitness_df)
    title = f'Mitigation fitness with {n_qubits} qubits (run {serial_code})'
    ax.set(xlabel='Mitigation', ylabel='Fitness', title=title)
    plt.savefig(plots_dir / f'{title}.png')
    plt.close()
    gc.collect() # https://github.com/matplotlib/matplotlib/issues/27713

    ratio_df = pd.concat(benchmark_ratios)
    mean_ratio_df = ratio_df.groupby(ratio_df.index).mean()
    # std_ratio_df = ratio_df.groupby(ratio_df.index).std()
    ax = mean_ratio_df.plot.bar(rot=0) # , yerr= std_ratio_df) # only include 1 standard error
    title = f'Mitigation ratio with {n_qubits} qubits (run {serial_code})'
    ax.set(xlabel='Mitigation', ylabel='Ratio', title=title)
    plt.savefig(plots_dir / f'{title}.png')
    plt.close()
    gc.collect()


# In[ ]:


def main():
    final_benchmark_fitnesses = []
    final_benchmark_ratios = []
    for n_qubits in [4, 5] :# range(6, max_qubits+1):
        benchmark_fitnesses = []
        benchmark_ratios = []
        for seed in range(n_seeds):
            np.random.seed(seed) # global random seed is probably not respected
            obs = Observable(PauliString("Z" * n_qubits))
            circuits = generate_circuits(n_qubits)
            circuit_names = [
                'GHZ',
                'W-state',
                'Random Clifford T',
            ]
            these_benchmark_fitnesses, these_benchmark_ratios = run_experiment(circuits, circuit_names, n_qubits, obs, serial_code)
            these_benchmark_fitnesses['seed'] = seed
            these_benchmark_ratios['seed'] = seed
            benchmark_fitnesses.append(these_benchmark_fitnesses)
            benchmark_ratios.append(these_benchmark_ratios)

        n_qubit_benchmark_fitnesses = pd.concat(benchmark_fitnesses)
        n_qubit_benchmark_ratios = pd.concat(benchmark_ratios)
        n_qubit_benchmark_fitnesses['qubits'] = n_qubits
        n_qubit_benchmark_ratios['qubits'] = n_qubits
        final_benchmark_fitnesses.append(n_qubit_benchmark_fitnesses)
        final_benchmark_ratios.append(n_qubit_benchmark_ratios)
        # plot_benchmarks(benchmark_fitnesses, benchmark_ratios, n_qubits, serial_code)

    pd.concat(final_benchmark_fitnesses).to_csv(f'tables/{now}-benchmark-fitnesses.csv')
    pd.concat(final_benchmark_ratio).to_csv(f'tables/{now}-benchmark-ratio.csv')


# In[ ]:


main()


# In[ ]:





# # 
"""
