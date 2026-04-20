# Can Cell-Type-Specific Effects Be Recovered from Bulk Phosphoproteomics via Deconvolution?

## Motivation

The Song 2024 dataset pairs 72-animal bulk phosphoproteomics with matched snRNA-seq, yielding a 24-group composition matrix (2 sexes x 3 timepoints x 4 genotypes, one composition vector per group). A natural question is whether this paired structure enables deconvolution: given bulk kinase signals and cell-type composition, can we recover cell-type-specific condition effects?

This analysis investigates whether the answer depends on the deconvolution method chosen, or on properties of the data itself.

## Setup

The deconvolution problem is: given a bulk measurement vector **y** (24 groups x 1 for a single phosphosite) and a composition matrix **A** (24 groups x 10 cell types), recover the cell-type-specific effect vector **b** (10 x 1) such that:

```
y = A b + noise
```

This is a standard linear inverse problem. Its solvability depends on the information content of **A** — how much independent compositional variation exists across the 24 groups.

All code for this analysis is in `code/supplementary/deconvolution_infeasibility.py`. Figures are in `outputs/reports/deconvolution_infeasibility/`.

## 1. Characterizing the Composition Matrix

### 1.1 Structure

The observed composition matrix **A_obs** has 24 rows (factorial groups) and 10 columns (cell types: Astrocytes, Endothelial cells, Excitatory neurons, High MT, Interneurons, Medium spiny neurons, Microglia, Oligodendrocytes, OPCs, Other). Each row sums to 1.

The simplex constraint (rows sum to 1) means one column is linearly dependent on the others. This eliminates one degree of freedom, leaving at most 9 effective composition directions. All analyses below account for this by excluding the trivial zero singular value.

### 1.2 Singular Value Decomposition

![SVD spectrum of A_obs](../outputs/reports/deconvolution_infeasibility/fig1_svd_spectrum.png)

**Figure 1.** SVD of the centered composition matrix.
- **(A)** The singular value spectrum drops steeply: σ₁ = 0.183, σ₂ = 0.090, then a gradual decay to σ₉ = 0.016.
- **(B)** The first two components capture 73.5% of total composition variance. Six components are needed for 95%.
- **(C)** Noise amplification via the pseudoinverse (1/σ) reaches 62x on the weakest effective direction. Components 7–9 all exceed the 10x amplification threshold.

The effective condition number is κ = 11.3 — moderate, not extreme. This is important: the failure we observe below is not caused by a numerically singular matrix, but by the interaction of moderate ill-conditioning with the sample-to-parameter ratio and realistic noise levels.

### 1.3 Composition Heatmap

![Composition heatmap](../outputs/reports/deconvolution_infeasibility/fig2_composition_heatmap.png)

**Figure 2.** Cell-type fractions across all 24 factorial groups. The compositions are strikingly similar across groups — Excitatory neurons and Interneurons dominate (25–30% each), and between-group variation in any cell type rarely exceeds 5 percentage points. The deconvolution problem asks us to extract cell-type-specific effects from these small compositional differences.

## 2. Synthetic Ground Truth Experiments

### 2.1 Experimental Design

To test whether deconvolution is feasible, we need experiments where the ground truth is known. The setup:

1. **Ground truth**: Generate 500 phosphosites with known cell-type-specific effects. Each site has effects in 1–3 cell types (sparse), with magnitudes drawn from [0.5, 2.0] LFC — realistic for phosphoproteomics.

2. **Bulk synthesis**: Compute `y = A b_true + noise`, where noise σ = 0.3 (matching the empirical MAD of the real data).

3. **Recovery**: Attempt to recover **b** from **y** and **A** using standard methods.

4. **Evaluation**: Pearson correlation between true and recovered effects, both overall and per-site.

The critical variable is **A** — we compare recovery using a well-conditioned composition matrix (positive control) versus the actual **A_obs** (negative control), holding everything else constant.

### 2.2 Positive Control vs. Actual Composition

![Positive vs negative control](../outputs/reports/deconvolution_infeasibility/fig3_positive_negative_control.png)

**Figure 3.** OLS deconvolution with identical ground truth effects and noise, differing only in the composition matrix.
- **(A)** With a well-conditioned composition (κ = 3), true and recovered effects are correlated (r = 0.52). The scatter follows the diagonal with moderate dispersion.
- **(B)** With the actual composition (κ = 11), the relationship between true and recovered effects collapses (r = 0.09). Recovered values explode to ±40 while true effects are ±4 — a hallmark of noise amplification through the pseudoinverse.
- **(C)** Per-site correlation distributions: the well-conditioned matrix produces a right-shifted distribution (median r = 0.53), while A_obs produces a distribution centered near zero (median r = 0.10), indistinguishable from noise.
- **(D)** Per-cell-type recovery: the well-conditioned matrix recovers all cell types with r = 0.3–0.7; A_obs recovers none above r = 0.2.

The same ground truth, the same noise, the same method. Only the composition matrix differs.

## 3. Method Independence

A natural objection is that OLS is a poor deconvolution method. Perhaps regularized or constrained approaches would succeed where OLS fails.

![Multi-method comparison](../outputs/reports/deconvolution_infeasibility/fig4_multimethod_comparison.png)

**Figure 4.** Six deconvolution methods applied to the same synthetic data using the actual composition matrix.
- **(A)** Overall recovery correlation by method. OLS achieves r = 0.06. Ridge regression improves to r = 0.33 at best (α = 1.0), but this comes from shrinkage toward zero — it captures the direction of a few large effects while erasing smaller ones. Non-negative least squares (NNLS) with signed decomposition achieves r = 0.12.
- **(B)** Per-site correlation distributions. Ridge (α = 1.0) has median per-site r = 0.08. The apparent improvement in overall r from Ridge is driven by bias-variance tradeoff at the aggregate level, not by faithful per-site recovery.

No method tested — unregularized, regularized, or constrained — achieves meaningful per-site recovery from the actual composition matrix. The best overall r of 0.33 (Ridge) is achieved by shrinking everything toward zero, which is not deconvolution.

## 4. Statistical Robustness

### 4.1 Monte Carlo Distribution

Is the positive-vs-negative comparison above a fluke of the particular random seed?

![Monte Carlo distribution](../outputs/reports/deconvolution_infeasibility/fig6_monte_carlo.png)

**Figure 5.** Distribution of overall Pearson r across 200 independent trials (different random ground truths and noise realizations each time).
- Well-conditioned: μ = 0.480 ± 0.011 (tight distribution around r ≈ 0.5)
- Actual A_obs: μ = 0.066 ± 0.014 (tight distribution near zero)
- **The distributions do not overlap.** 100% of well-conditioned trials exceed the best A_obs trial.

The result is not stochastic. Every realization of the ground truth and noise produces the same qualitative outcome: well-conditioned compositions support deconvolution; the actual composition does not.

### 4.2 Noise Sensitivity

Does recovery with A_obs improve at lower noise levels?

![Noise sensitivity](../outputs/reports/deconvolution_infeasibility/fig7_noise_sensitivity.png)

**Figure 6.** Recovery as a function of measurement noise σ. The vertical dashed line marks the empirical noise level (σ = 0.3).
- The well-conditioned composition achieves near-perfect recovery (r → 1) as noise → 0, degrading gracefully with increasing noise.
- The actual composition shows a much steeper degradation curve. Even at σ = 0.01 (30x less noise than observed), A_obs recovery is substantially below the well-conditioned case.

The gap between the curves at low noise reflects the inherent information loss in the composition geometry. At the empirical noise level, A_obs recovery has already collapsed to near zero.

## 5. Parametric Transition

To demonstrate that recovery degrades continuously as composition quality worsens, we interpolate between the well-conditioned matrix and A_obs:

```
A(α) = (1 - α) A_good + α A_obs,  normalized to valid compositions
```

![Condition sweep](../outputs/reports/deconvolution_infeasibility/fig5_condition_sweep.png)

**Figure 7.** Recovery (Pearson r) as a function of interpolation parameter α.
- **(A)** As α increases from 0 (well-conditioned) to 1 (actual A_obs), both overall and per-site recovery decline monotonically.
- **(B)** The same data plotted against the effective condition number κ of the interpolated matrix. Recovery collapses as κ increases, with no recovery plateau — there is no regime where A_obs-like compositions support deconvolution.

## 6. Summary of Findings

| Property | Well-conditioned | Actual A_obs |
|:---|:---|:---|
| Effective condition number (κ) | 3 | 11.3 |
| Components with σ > 0.05 | 9 | 4 |
| Variance in top-2 components | — | 73.5% |
| Overall recovery r (OLS) | 0.517 | 0.087 |
| Median per-site r (OLS) | 0.527 | 0.102 |
| Best method r on A_obs | — | 0.330 (Ridge) |
| Best method per-site r on A_obs | — | 0.130 (Ridge) |
| Monte Carlo non-overlap | 100% of well-conditioned trials exceed best A_obs trial |

## 7. Interpretation

The composition geometry of A_obs does not support deconvolution: low effective rank (4 of 9 singular values > 0.05), small absolute compositional variation (σ₁ = 0.183), and unfavorable sample-to-parameter ratio (24 groups, 9 effective dimensions). These factors interact multiplicatively. The conclusion is method-independent (Figure 4), robust across 200 Monte Carlo trials (Figure 5), continuous in composition quality (Figure 7), and not a noise artifact (Figure 6).

For the design logic behind the pivot away from deconvolution, see [`analysis_rationale.md`](foundation/analysis_rationale.md). For the mathematical constraints, see [`statistical_constraints.md`](foundation/statistical_constraints.md).

## Reproducibility

```bash
python code/supplementary/deconvolution_infeasibility.py --run
```

All figures are generated from the observed composition matrix (`A_obs_fractions.tsv`) and synthetic data with fixed random seed (42). No external data beyond the composition matrix is required.
