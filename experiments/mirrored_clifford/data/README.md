# Mirrored Clifford+T dataset

Tidy, analysis-ready copy of every mirrored run (N = 10 seeds × 6 depth multipliers × 5 methods). Built from the dated folders in `../results/` by `../build_dataset.py`, so re-run that script if raw files change.

```python
import pandas as pd
runs = pd.read_csv("https://raw.githubusercontent.com/schance995/dna-game/marco/mirrored-analysis/experiments/mirrored_clifford/data/mirrored_runs.csv")
```

## Experimental setup (applies to every row)

- Circuit: random Clifford+T circuit C on 7 qubits with `gates_per_type` single-qubit Cliffords, two-qubit Cliffords and T gates, followed by `cirq.inverse(C)`. Ideal ⟨Z⊗7⟩ = +1.
- `seed` fixes the circuit, so the same seed means the same circuit for every method. Compare methods **paired by (seed, multiplier)**.
- Noise is fixed across all runs: depolarizing 0.01 on every qubit in every moment, then bit-flip, phase-flip and amplitude damping (0.05 each) before measurement; 1,000 shots.
- The multiplier scales circuit size only: gates_per_type = round(multiplier × 49).

## `mirrored_runs.csv`: one row per (seed, multiplier, method), 300 rows

| column | meaning |
|---|---|
| `seed` | circuit seed (0–9) |
| `multiplier` | depth multiplier k ∈ {0.5, 1, 2, 4, 6, 8} |
| `gates_per_type`, `circuit_gates` | round(49k), and the total gates in the mirrored circuit (6 × gates_per_type) |
| `method` | `GA`, `grid` (grid search), or a single default technique: `REM`, `ZNE`, `DDD` |
| `ideal`, `noisy`, `mitigated` | expectation values: exact; unmitigated noisy run; one fresh run of the chosen pipeline |
| `ratio_raw` | \|mitigated − ideal\| / \|noisy − ideal\|. **Main outcome.** Lower is better; 1 = no improvement |
| `log_ratio` | ln(ratio_raw), for paired tests |
| `search_fitness` | best fitness seen during search, sigmoid(−ln ratio); inflated by the winner's curse. For singletons it is the single run's fitness |
| `n_evals` | pipelines evaluated: GA 840 (40 × 21 generations), grid 288, singletons 1 |
| `runtime_s` | wall-clock seconds for the search or run |
| `pipeline` | chosen configuration as text; singletons are `default ...` |
| `technique` | `ZNE` or `DDD` for REM+X pipelines (the second stage); the technique itself for singletons |
| `p0`, `p1` | REM readout-error parameters |
| `ddd_rule` | `xx`, `yy`, `xyxy` |
| `factory`, `fold` | ZNE extrapolation factory and folding method |
| `n_scale`, `min_scale`, `max_scale` | ZNE scale factors (count, min, max); empty for AdaExp |
| `ga_reeval` | GA only: `last_gen` = re-ran the last generation's best (seeds 0–2 at ×0.5–×4, from before a logging fix); `best_ever` = re-ran the best configuration found |
| `ratio_raw_last_gen` | GA only, `best_ever` runs: ratio of the last generation's best, for comparison. On 48 runs the two don't differ significantly (Wilcoxon p = 0.15) |
| `gens_to_converge` | GA only: first generation within 1e-3 of the final best fitness |

Singleton defaults: REM with an inverse confusion matrix for p0 = p1 = 0.05; ZNE = mitiq default (Richardson, scale factors [1, 2, 3], `fold_gates_at_random`); DDD = `xx` rule.

Known gaps: GA ×8 seed 5 has `ratio_raw` = NaN.

## `mirrored_ga_curves.csv`: GA per-generation fitness, 1,260 rows

`seed, multiplier, gen (0–20), max_fit, mean_fit`. Fitness here is in-search, so a single noisy evaluation per configuration.

## `mirrored_retest.csv`: same configuration re-run 10×, 240 rows

The GA and grid winners for seeds 0–2 at ×0.5–×4, each re-run 10 times (`rep`): `method, seed, multiplier, gates_per_type, rep, ideal, noisy, mitigated, ratio_raw, runtime_s, pipeline`. Use this to measure how repeatable a single `ratio_raw` is (intraclass correlation ≈ 0; see `../README.md`).

## Non-mirrored benchmarks (May run)

GHZ, W-state, Random Clifford+T and QFT at 5–7 qubits live in `results/20260509T192016/` at the repo root. `results.csv` has every evaluated configuration (GA/grid/random search); `benchmarks.csv` has one re-evaluation of each winner plus the default techniques. Note that the circuit seed was fixed at 0 there, so `seed` 0–2 are repeat runs on the same circuit.
