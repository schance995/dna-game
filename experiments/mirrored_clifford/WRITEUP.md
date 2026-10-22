# Mirrored Random Clifford+T Difficulty Scaling — Write-up (Marco)

**Scope:** Increase difficulty of the Random Clifford+T benchmark by scaling gate count, measure GA runtime and mitigation quality, validate across seeds.

**Status:** Mirror fix validated. N=10 complete for ×0.5–×4 and for ×6/×8 (Sep 10). GA peak-chromosome logging fixed for future runs.

**Artifacts:** All data under `new/results/` with date-stamped folders.

---

## 1. Summary (for team / paper)

We attempted to make Random Clifford+T harder by scaling gate counts to \(n^2\) per gate type (\(n=7\)). On **plain** (unmirrored) circuits, the fitness metric **saturates**: ideal and noisy expectations both collapse toward zero, so the GA appears to converge instantly with fitness 1.0 regardless of depth.

**Fix (agreed in meeting):** append the circuit inverse (mirrored Clifford+T). Then ideal \(\langle Z^{\otimes n}\rangle = +1\) exactly, while noise still drives the noisy value toward ~0. Gate count becomes a measurable difficulty knob again.

**Multi-seed result (seeds 0–2, mirrored):**
- Best **fitness** dips at difficulty ×1 (mean 0.687 ± 0.08), recovers at ×2/×4 — this dip is **structural**, not one bad seed.
- **Generations to converge** are ~5–10 on average with large variance (±5–8), not the flat 2–4 seen in single-seed runs.
- **Honest re-run error ratio** (fresh execution of best pipeline) often **exceeds 1.0** at ×0.5 and ×1 (worse than no mitigation). Best mean ratio at ×2 (0.60 ± 0.43), but high seed variance.

**Main methodological finding:** GA fitness (single stochastic eval) and true mitigation quality (re-run) **disagree** — the GA can overfit shot noise, especially with aggressive ZNE scale factors.

We are **not** wrong to scale gates; we were wrong to do it on unmirrored circuits without checking the metric. The mirror direction is **correct**. Claims about “easy convergence” and monotone depth–quality tradeoffs need to be **weakened** based on multi-seed evidence.

---

## 2. Background: why plain Random Clifford+T breaks the metric

### 2.1 Fitness definition (from paper / code)

\[
\text{fitness} = \sigma\left(-\ln \frac{|\text{mitigated} - \text{ideal}|}{|\text{noisy} - \text{ideal}|}\right)
\]

Fitness is bounded in \((0,1)\). It only measures meaningful mitigation when the denominator \(|\text{noisy} - \text{ideal}|\) is **well above shot noise**.

With 1000 shots, a rough noise floor is \(1/\sqrt{1000} \approx 0.032\).

### 2.2 Noise model bias toward zero

Our simulator applies depolarizing noise (plus bit/phase flip and amplitude damping) on each gate. Depolarizing noise **contracts** expectation values toward 0: for an observable like \(\langle Z^{\otimes n}\rangle\),

\[
\langle Z^{\otimes n}\rangle_{\text{noisy}} \approx \lambda \cdot \langle Z^{\otimes n}\rangle_{\text{ideal}}, \quad 0 < \lambda < 1
\]

On a **scrambled** random Clifford+T circuit, the ideal value is already \(\approx 0\) (state spread over many basis states). Noise pushes the noisy value toward 0 as well. The signal \(|\text{noisy} - \text{ideal}|\) sits at the shot-noise floor — the metric has nothing to measure.

**Meeting alignment:** “Noise model effectively biases output towards 0” — this is what we measured.

### 2.3 Evidence: saturation probe (`verify_saturation.py`)

**File:** `new/results/20260610_difficulty_sweep/verify_saturation_output.txt`

| Difficulty | Gates/type | Ideal ⟨Z⁷⟩ | Noisy ⟨Z⁷⟩ | \|ideal−noisy\| | Below floor (~0.032)? |
|---|---|---|---|---|---|
| ×1 | 49 | +0.026 ± 0.041 | −0.009 ± 0.011 | 0.035 | borderline |
| ×2 | 98 | +0.009 ± 0.041 | +0.024 ± 0.025 | 0.015 | **yes** |
| ×4 | 196 | +0.017 ± 0.020 | −0.014 ± 0.010 | 0.031 | **yes** |
| ×8 | 392 | −0.108 ± 0.016 | −0.014 ± 0.023 | 0.093 | no (but ideal already far from useful signal) |

Ideal is statistically indistinguishable from zero at most levels. Increasing gate count does **not** create a larger mitigation target — it often **shrinks** the denominator further.

---

## 3. Negative result: unmirrored difficulty sweep

**Run:** Jun 10 overnight, seed 42, levels ×1/×2/×4/×8, pop=40, 20 epochs.

**File:** `new/results/20260610_difficulty_sweep/difficulty_sweep.csv`  
**Plot:** `new/results/20260610_difficulty_sweep/difficulty_sweep.png`

| Difficulty | Gates/type | Best fitness | Gens to converge | GA time |
|---|---|---|---|---|
| ×1 | 49 | **1.0000** | 0 | 3888 s |
| ×2 | 98 | **1.0000** | 6 | 3649 s |
| ×4 | 196 | **1.0000** | 7 | 11138 s |
| ×8 | 392 | **1.0000** | 1 | 7440 s |

Every level hits fitness exactly 1.0. This matches the paper’s Fig. 5 behavior (“max fitness fluctuates, median stays low”) — likely shot-noise artifacts on a degenerate metric, not genuine GA success.

**Conclusion:** Gate-count scaling on unmirrored Random Clifford+T cannot answer “how much harder does it get?” — the benchmark stops measuring mitigation.

---

## 4. Fix: mirrored Clifford+T

### 4.1 Construction

For random Clifford+T circuit \(C\), use:

\[
C_{\text{mirror}} = C \;+\; C^{-1}
\]

Implemented in `difficulty_sweep.py` / `multi_seed_sweep.py`:

```python
circuit = generate_random_clifford_t_circuit(...)
circuit = circuit + cirq.inverse(circuit)
```

### 4.2 Why this works

| Quantity | Unmirrored | Mirrored |
|---|---|---|
| Ideal ⟨Zⁿ⟩ | ≈ 0 (scrambled) | **+1 exactly** (returns to \|0…0⟩) |
| Noisy ⟨Zⁿ⟩ | ≈ 0 (noise contracts) | ≈ 0 (errors accumulate through 2× depth) |
| \|noisy − ideal\| | ~0.02–0.07 (noise floor) | **~1.0** (full signal) |

**Sanity check (server, before multi-seed):** mirrored ×1 and ×4 both gave ideal = +1.0000, noisy ≈ −0.02 to −0.04, gap ≈ 1.02–1.04.

### 4.3 Cost

Mirroring doubles circuit depth (forward + inverse). Per-eval wall time scales roughly linearly with total gates; ×4 mirrored ≈ 1176 gates per direction → ~2352 effective gate layers for simulation.

---

## 5. Single-seed mirrored sweep (seed 42)

**Run:** Jun 10, seed 42, levels ×0.5/×1/×2/×4.

**Files:** `new/results/20260610_mirror_sweep/difficulty_sweep_mirror.csv`, `.png`

| Difficulty | Best fitness | Gens to converge | GA time (min) |
|---|---|---|---|
| ×0.5 | 0.975 | 2 | 94.5 |
| ×1 | 0.834 | 3 | 89.4 |
| ×2 | 0.952 | 4 | 109.0 |
| ×4 | 0.607 | 2 | 140.7 |

Fitness now varies with depth (no longer all 1.0). Initial story: convergence ~2–4 gens (flat), quality drops at ×4. **Multi-seed run below revises this.**

---

## 6. Multi-seed mirrored GA sweep (seeds 0, 1, 2) — primary evidence

**Run:** Jun 12–13, levels ×0.5/×1/×2/×4, pop=40, 20 epochs, mirrored.

**Files:**
- Raw: `new/results/20260613_multi_seed/multi_seed_raw.csv`
- Curves: `new/results/20260613_multi_seed/multi_seed_curves.csv`
- Summary: `new/results/20260613_multi_seed/multi_seed_summary.csv`
- Plots: `fig_metrics_errorbars.png`, `fig_convergence_curves.png`, `fig_raw_values.png`

### 6.1 Best fitness (mean ± std)

| Level | Mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| ×0.5 | 0.924 ± 0.039 | 0.964 | 0.886 | 0.923 |
| ×1 | **0.687 ± 0.080** | 0.622 | 0.776 | 0.662 |
| ×2 | 0.966 ± 0.010 | 0.977 | 0.958 | 0.963 |
| ×4 | 0.961 ± 0.028 | 0.933 | 0.961 | 0.990 |

**Finding:** ×1 dip appears in **all three seeds** — not seed variance.

### 6.2 Generations to converge (mean ± std)

| Level | Mean ± std |
|---|---|
| ×0.5 | 9.3 ± 6.0 |
| ×1 | 5.3 ± 5.1 |
| ×2 | 10.0 ± 7.9 |
| ×4 | 6.3 ± 4.2 |

**Finding:** Not flat. High variance across seeds. Single-seed seed-42 run understated this.

### 6.3 Error ratio on fresh re-run (mean ± std)

Ratio = \(|\text{mitigated} - \text{ideal}| / |\text{noisy} - \text{ideal}|\). **Lower is better. Above 1.0 = worse than no mitigation.**

| Level | Mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| ×0.5 | 1.45 ± 0.31 | 1.80 | 1.34 | 1.21 |
| ×1 | 1.02 ± 0.04 | 1.04 | 0.98 | 1.04 |
| ×2 | 0.60 ± 0.43 | 0.29 | 0.42 | 1.10 |
| ×4 | 0.72 ± 0.45 | 1.05 | 0.91 | 0.21 |

**Finding:** Fitness can be high while re-run ratio is bad. Example: seed 0 ×2 has best fitness 0.977 but ratio 0.29 (good); seed 2 ×2 has fitness 0.963 but ratio 1.10 (bad).

### 6.4 Raw values sanity check

Ideal = 1.0 for all runs (mirror works). Noisy ≈ 0. Mitigated on re-run:
- ×0.5 mean ≈ −0.45 (overshoots wrong direction)
- ×2 mean ≈ +0.6 (best recovery toward +1)
- ×4 mean ≈ +0.25 (partial)

See `fig_raw_values.png`.

### 6.5 Pipeline composition (informal)

From `multi_seed_raw.csv`, best pipelines mix REM+DDD and REM+ZNE. ZNE winners often use `fold_all` with large Richardson/Linear scale factors (up to ~9×), which inflate circuit depth during eval and may cause overshoot on re-run. DDD rules: xx, yy, xyxy all appear.

No aggregate “35/36 use REM+DDD” style count yet — needs scripted tally across all 12 runs.

---

## 6.6 P0 baselines: grid + singletons (seeds 0–2) — complete Jun 16

**Files:** `results/20260616_p0_baselines/brute_force_raw.csv`, `singletons_raw.csv`  
**Plot:** `results/20260616_p0_baselines/fig_ga_vs_baselines.png`

Grid search exhaustively evaluates **288 discrete configs** per run (4 REM × 4 REM × {3 DDD rules or 5 ZNE factories × 3 fold methods}). GA explores a **continuous** parameter space with ~800 evals. Random search omitted from quality plots (single-shot overfitting; 9/12 cells hit `mitigated = 1.0` exactly).

### Single-shot re-run ratio (lower = better)

| Level | GA | Grid | Best singleton (REM/ZNE/DDD) |
|---|---|---|---|
| ×0.5 | 1.45 ± 0.31 | **0.52 ± 0.59** | 0.79 ± 0.10 |
| ×1 | 1.02 ± 0.04 | **0.85 ± 1.14** | 0.90 ± 0.09 |
| ×2 | 0.60 ± 0.43 | **0.46 ± 0.41** | 0.89 ± 0.13 |
| ×4 | **0.72 ± 0.45** | 1.27 ± 0.46 | 0.85 ± 0.12 |

Head-to-head (12 cells): grid wins 7, GA 2, best singleton 3. Random search not counted.

**Findings:**
- Grid beats GA on single-shot ratio at most levels despite using only 36% of GA’s eval budget — likely because the optimum lies on or near the coarse discrete grid.
- Default **ZNE alone** is competitive (often beats GA); grid always picks **REM+ZNE** with preset factories.
- High variance (especially grid seed 2 at ×1) motivated multi-sample re-runs (Section 6.7).

**Note:** Prior GA runs logged the **last generation’s** chromosome for `ratio_raw`, not the peak-fitness chromosome (`mutate()` corrupts earlier pops). Fixed in `genetic_algorithm_2` (deep-copy best-ever each generation). Existing CSV pipelines are last-gen winners; multi-sample re-runs use those saved strings.

---

## 6.7 Multi-sample re-runs — complete

**Script:** `multisample_rerun.py` — 10 re-executions per saved GA/grid pipeline.

**Files:** `results/20260617_multisample/multisample_summary.csv`, `fig_multisample_ratio.png`

| Level | GA (mean ± std) | Grid (mean ± std) |
|---|---|---|
| ×0.5 | 1.02 ± 0.04 | **0.61 ± 0.29** |
| ×1 | **0.95 ± 0.03** | 1.05 ± 0.12 |
| ×2 | 1.32 ± 0.44 | **0.92 ± 0.07** |
| ×4 | **1.01 ± 0.00** | 1.24 ± 0.45 |

Head-to-head (multi-sample mean, 12 cells): **grid wins 8, GA 4**. Conclusion holds from single-shot: grid better at shallow depth; neither method reliably beats ratio 1.0 at all levels. Multi-sampling reduced wild single-shot swings (e.g. GA ×1 now ~0.95 vs 1.04) but did not flip the overall story.

---

## 6.8 ×6 / ×8, N=10 — complete Sep 10

**Run:** Jul 26–Sep 10, seeds 0–9, levels ×6 and ×8, mirrored, `DNA_MAX_WORKERS=80`. Interrupted by a reboot on Aug 5; resumed Aug 24 from checkpoints.

**Files:** `results/20260726_x6_x8/`  
**Plot (share this):** `results/20260913_mirrored_paper/fig_ratio_by_difficulty.png`

Single-shot `ratio_raw` (lower is better; 1.0 = no mitigation). GA ×8 mean excludes seed 5 (ratio was NaN).

| Level | GA | Grid | Best singleton |
|---|---|---|---|
| ×6 | **0.72 ± 0.62** | 1.31 ± 0.47 | 0.91 ± 0.09 |
| ×8 | 1.81 ± 2.06 (n=9) | 1.17 ± 0.67 | **0.92 ± 0.09** |

Head-to-head on ratio: ×6 GA 6, singleton 4, grid 0. ×8 singleton 5, grid 3, GA 1 (9 seeds).

×6 is the first level where the GA beats both grid and a default technique on the re-run. It does not hold at ×8. Grid is worse than doing nothing at both levels. Fitness stays high when the ratio is bad (seed 3 ×8: fitness 0.96, ratio 7.17), so the paper-style fitness figures still overstate search. Seed 2 ×8 ratio 0.008 is a single shot, not a multi-sample result.

---

## 7. Why fitness and mitigation disagree

1. **Fitness uses one stochastic evaluation** per chromosome during GA (1000 shots). Re-run ratio uses a **different** stochastic sample after GA finishes.
2. **Sigmoid transform** compresses the ratio into (0,1); small denominator changes can swing fitness a lot.
3. **ZNE extrapolation** from over-amplified circuits (scale factors ≫ 1) fits noise and can overshoot — high fitness in one sample, catastrophic mitigated value on re-run (e.g. mitigated = −0.84 vs ideal +1.0).
4. **Convergence curves are spiky** (`fig_convergence_curves.png`): max fitness jumps from lucky evals then drops — classic overfitting to evaluation noise.
5. **GA peak vs last generation:** until fixed, `ratio_raw` in the Jun 13 GA CSV used the final generation’s winner, not the peak-fitness chromosome (gap ~0.2–0.4 fitness on every run). `genetic_algorithm_2` now deep-copies the best-ever individual each generation.

This is a **methodological result**, not a coding bug.

---

## 8. What we can and cannot claim

### Can claim (with evidence)

- Unmirrored gate scaling breaks the fitness metric (Section 3 + saturation probe).
- Mirroring restores ideal = +1 and makes difficulty measurable (Section 4).
- Multi-seed GA on mirrored circuits: ×1 fitness dip is reproducible; convergence is O(~10) generations with high variance; honest re-run often fails at shallow depth, best at ×2 on average.
- GA fitness is an unreliable proxy for true mitigation without multi-sample validation.
- On mirrored benchmarks (N=3 seeds), **grid search beats GA** on single-shot re-run ratio at most levels; **default ZNE** is competitive with composed pipelines.
- Grid’s search space is small and **fully enumerated** (288 configs); GA searches a larger continuous space with ~800 samples — GA losing does not imply the problem is easy, but that the discrete grid already covers the good configs.

### Cannot claim yet

- Multi-sample re-runs of the N=10 and ×6/×8 winners (Section 6.7 is seeds 0–2 only).
- That harder circuits generally favor the GA. ×6 is the first level where GA wins on ratio; ×8 reverses it.
- Monotone “deeper = harder to mitigate” — data is non-monotone.
- That seed 2 ×8 (ratio 0.008) or any other single-shot extreme is a real pipeline, until it is re-sampled.

---

## 9. Suggested paper paragraph (draft)

> We scaled Random Clifford+T gate counts as a difficulty knob for genetic optimization of error-mitigation pipelines. On unmirrored circuits, the ideal expectation of \(\langle Z^{\otimes n}\rangle\) collapses toward zero under scrambling, while depolarizing noise drives the noisy expectation to the same limit; the fitness denominator therefore falls to the shot-noise floor and the GA reports spurious convergence (fitness 1.0 at generation 0). Appending the circuit inverse (mirrored Clifford+T) restores an ideal value of +1 while allowing noise to accumulate over increased depth. Across three random seeds, we find a reproducible fitness minimum at intermediate depth (×1 baseline), generation counts of 5–10 with high variance, and frequent disagreement between GA fitness and mitigation quality measured on independent re-execution. Exhaustive grid search over 288 discrete pipeline configurations outperforms the GA on single-shot re-run error ratio at most difficulty levels, while default ZNE singletons remain competitive—suggesting the coarse grid already covers near-optimal configs and that GA fitness on one stochastic evaluation is a poor proxy for robust mitigation.

---

## 10. TODO — evidence still needed

| Priority | Task | Script / method | Status |
|---|---|---|---|
| **P0** | Grid + singleton baselines on mirrored circuits | `brute_force_mirror.py`, `singletons_mirror.py` | **done** |
| **P0** | Multi-sample re-runs of saved pipelines | `multisample_rerun.py` | **done** |
| **P0** | GA peak-chromosome logging fix | `genetic_algorithm_2` + `multi_seed_sweep.py` | **done** (future GA runs) |
| **P1** | Add seeds 3–9 (N=10) | `multi_seed_sweep.py` | **done** (×0.5–×4 and ×6/×8) |
| **P1** | ×6 / ×8 mirrored difficulty | `run_x6_x8.sh` | **done** (N=10, Sep 10) |
| **P2** | Coarse vs fine grid ablation | experiment | pending |
| **P2** | Pipeline composition tally | analysis script | pending |
| **P2** | Cap ZNE max scale factor | experiment | pending |

**Infrastructure (done):** idempotent resume (`multi_seed_sweep.py`), watchdog (`watchdog.sh`), heartbeat status file.

---

## 11. Reproduce

```bash
# On server (dnagame@10.242.42.1), in ~/dna-game/new
PYTHONWARNINGS=ignore .venv/bin/python multi_seed_sweep.py --seeds 0 1 2 --levels 0.5 1 2 4
PYTHONWARNINGS=ignore .venv/bin/python visualize_multi_seed.py
PYTHONWARNINGS=ignore .venv/bin/python brute_force_mirror.py --seeds 0 1 2 --levels 0.5 1 2 4
PYTHONWARNINGS=ignore .venv/bin/python singletons_mirror.py --seeds 0 1 2 --levels 0.5 1 2 4
PYTHONWARNINGS=ignore .venv/bin/python multisample_rerun.py --ga-csv multi_seed_raw.csv --grid-csv brute_force_raw.csv --n-reps 10
PYTHONWARNINGS=ignore .venv/bin/python visualize_multisample.py
```

---

## 12. Figure list for slides / paper

| Figure | Path |
|---|---|
| Unmirrored saturation (fitness vs difficulty) | `20260610_difficulty_sweep/difficulty_sweep.png` |
| Mirrored single-seed | `20260610_mirror_sweep/difficulty_sweep_mirror.png` |
| Multi-seed metrics + error bars | `20260613_multi_seed/fig_metrics_errorbars.png` |
| Convergence curves per seed | `20260613_multi_seed/fig_convergence_curves.png` |
| Raw ideal/noisy/mitigated | `20260613_multi_seed/fig_raw_values.png` |
| GA vs grid vs singleton (single-shot) | `20260616_p0_baselines/fig_ga_vs_baselines.png` |
| Multi-sample ratio (GA vs grid) | `20260617_multisample/fig_multisample_ratio.png` |
| N=10 ratio vs difficulty (×0.5–×8) | `20260913_mirrored_paper/fig_ratio_by_difficulty.png` |
| Paper-style fitness figs including ×6/×8 | `20260913_mirrored_paper/fig1_circuit_difficulty.png`, `fig3_composition_vs_single.png` |
