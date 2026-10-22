"""Shared helpers for mirrored Clifford+T benchmark scripts."""
import re
import importlib.util
from pathlib import Path

import numpy as np


def load_dna_game():
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location("dna_game_2", root / "dna-game-2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_circuit(m, n_qubits, gates, seed):
    circuit = m.generate_random_clifford_t_circuit(
        num_qubits=n_qubits,
        num_oneq_cliffords=gates,
        num_twoq_cliffords=gates,
        num_t_gates=gates,
        seed=seed,
    )
    return circuit + m.cirq.inverse(circuit)


def _parse_float_list(inner: str) -> list[float]:
    inner = re.sub(r"(\d)([eE])", r"\1 \2", inner)
    inner = re.sub(r"(\d)\s+(\d)", r"\1, \2", inner)
    inner = inner.strip().strip(",")
    return [float(x.strip()) for x in inner.split(",") if x.strip()]


def _parse_rem(text: str, n_qubits, m):
    match = re.match(r"rem\(\s*([0-9.eE+\-]+)\s*,\s*([0-9.eE+\-]+)\s*\)", text.strip())
    if not match:
        raise ValueError(f"bad rem gene: {text!r}")
    p0, p1 = map(float, match.groups())
    return m.REMGene(p0=p0, p1=p1, n_qubits=n_qubits)


def _parse_ddd(text: str, obs, m):
    match = re.match(r"ddd\(\s*(\w+)\s*\)", text.strip())
    if not match:
        raise ValueError(f"bad ddd gene: {text!r}")
    rule = getattr(m.ddd.rules, match.group(1))
    return m.DDDGene(rule=rule, obs=obs)


def _parse_factory(text: str, m):
    text = text.strip()
    if text.startswith("RichardsonFactory"):
        inner = re.search(r"\[(.*)\]", text).group(1)
        return m.zne.inference.RichardsonFactory(scale_factors=_parse_float_list(inner))
    if text.startswith("LinearFactory"):
        inner = re.search(r"\[(.*)\]", text).group(1)
        return m.zne.inference.LinearFactory(scale_factors=_parse_float_list(inner))
    if text.startswith("PolyFactory"):
        arr = re.search(r"\[(.*)\]", text).group(1)
        order = int(re.search(r"\]\s*,\s*(\d+)", text).group(1))
        return m.zne.inference.PolyFactory(
            scale_factors=_parse_float_list(arr), order=order
        )
    if text.startswith("ExpFactory"):
        arr = re.search(r"\[(.*)\]", text).group(1)
        asymptote = float(re.search(r"asymptote\s*=\s*([0-9.eE+\-]+)", text).group(1))
        return m.zne.inference.ExpFactory(
            scale_factors=_parse_float_list(arr), asymptote=asymptote
        )
    if text.startswith("AdaExpFactory"):
        steps = int(re.search(r"steps\s*=\s*(\d+)", text).group(1))
        asym = re.search(r"asymptote\s*=\s*([0-9.eE+\-]+)", text)
        if asym:
            return m.zne.inference.AdaExpFactory(
                steps=steps, asymptote=float(asym.group(1))
            )
        return m.zne.inference.AdaExpFactory(steps=steps)
    raise ValueError(f"unknown ZNE factory: {text!r}")


def _parse_zne(text: str, obs, m):
    match = re.match(r"zne\((.+),\s*(fold_\w+)\)", text.strip())
    if not match:
        raise ValueError(f"bad zne gene: {text!r}")
    factory = _parse_factory(match.group(1).strip(), m)
    fold_name = match.group(2)
    scale_noise = getattr(m.zne.scaling, fold_name)
    return m.ZNEGene(factory=factory, scale_noise=scale_noise, num_to_avg=1, obs=obs)


def _split_genes(body: str) -> list[str]:
    genes, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            genes.append(body[start:i].strip())
            start = i + 1
    genes.append(body[start:].strip())
    return genes


def parse_pipeline(pipeline: str, n_qubits, obs, m=None):
    """Rebuild a chromosome list from a logged best_pipeline string."""
    if m is None:
        m = load_dna_game()
    text = pipeline.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"expected bracketed pipeline, got {text!r}")
    genes = _split_genes(text[1:-1].strip())
    chrom = []
    for gene in genes:
        gene = gene.strip()
        if gene.startswith("rem("):
            chrom.append(_parse_rem(gene, n_qubits, m))
        elif gene.startswith("ddd("):
            chrom.append(_parse_ddd(gene, obs, m))
        elif gene.startswith("zne("):
            chrom.append(_parse_zne(gene, obs, m))
        else:
            raise ValueError(f"unknown gene prefix in {gene!r}")
    return chrom


def ratio_raw(ideal, noisy, mitigated):
    denom = abs(noisy - ideal)
    if denom <= 0:
        return float("inf")
    return abs(mitigated - ideal) / denom
