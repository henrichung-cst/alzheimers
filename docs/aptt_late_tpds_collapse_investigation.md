# ApTt-late TPDS collapse — investigation notes

**Status:** open. Framing revised 2026-05-14 — the "ApTt-specific
collapse" is a downstream symptom of a global late-timepoint SigProb
collapse.
**Last updated:** 2026-05-14.

## 2026-05-14 finding — the problem is not ApTt-specific

Diagnostic at `alz/integration/diagnostics/aptt_late_collapse_m1_vs_m2.py`
on `outputs/reports/incytr_factorial_5xfad_kldata/receiver_cache/receiver=Astrocytes/`
revealed that the per-condition mean SigProb columns in the receiver
cache show a **global late-timepoint collapse**, not an ApTt-specific
one:

| Cell | max SigProb | mean SigProb |
|---|---|---|
| `SigProb_*_2mo` (any genotype) | ~0.295 | ~5e-5 |
| `SigProb_*_4mo` (any genotype) | ~2e-3 | ~2e-7 |
| `SigProb_*_6mo` (any genotype) | **0.0** | **0.0** |

All `*_6mo` SigProb values are **exactly zero** for the full 164k-path
universe in Astrocytes, including `SigProb_WTyp_6mo`. The 4mo values
are 2–3 orders of magnitude smaller than 2mo. So the late-ApTt TPDS
collapse is simply what falls out of the OLS when both the ApTt and WT
cells at 4mo/6mo have zero SigProb. Other "late" contrasts (App_4mo,
Tau_6mo, etc.) still show non-zero TPDS in some paths only because
their contrast vectors blend in 2mo cells via the time-invariant main
effects.

**Animal counts from `data/incytr_factorial_inputs/animal_metadata.csv`
(15 animals total, males-only, post-outlier-exclusion):**

| Genotype | 2mo | 4mo | 6mo |
|---|---|---|---|
| WTyp | 2 | 1 | 1 |
| AppP | 2 | 1 | 1 |
| Ttau | 2 | 1 | 1 |
| ApTt | 1 | 1 | 1 |

Every late-timepoint cell has n=1. n=1 alone is not sufficient to
explain SigProb = exactly 0 (it would produce noise, not zero) — so
there is a *second* mechanism beyond sample size collapsing the
late-timepoint per-animal mean expression of L/R/EM/T genes to a
regime where the Hill function (K=0.5, N=2) returns numerically zero.

## What this means for hypotheses (A) and (B) below

- (A) is correct that n=1 is the proximate sample-size constraint but
  is not the mechanism — n=1 with reasonable per-animal mean expression
  would give noisy TPDS, not zero TPDS.
- (B) is moot: design-matrix expansion (`+ Int_x_time4`, `+ Int_x_time6`)
  cannot rescue paths where the underlying SigProb is zero in both the
  ApTt and WT cells. `log(0) - log(0)` is undefined regardless of how
  many DOF the design has.

## 2026-05-14 finding 2 — input expression matrix is healthy at late timepoints

Read `data/incytr_factorial_inputs/expression_matrix.mtx` (30,567 genes
× 27,438 barcodes, genes-by-barcodes orientation) and computed
per-barcode total counts grouped by (timepoint, genotype, animal):

| Timepoint | mean counts/barcode | n_barcodes/animal range |
|---|---|---|
| 2mo | 3,060–3,250 | 881–1,334 |
| 4mo | 2,610–2,880 | 1,811–3,033 |
| 6mo | 2,710–2,800 | 1,626–3,473 |

Per-barcode sequencing depth is **essentially identical** across
timepoints (≤8% variation). Per-animal nucleus counts are **higher**
at 4mo/6mo than at 2mo, not lower. The raw expression input is
healthy at late timepoints — so the SigProb collapse is happening
**inside Incytr's per-animal / per-condition aggregation**, not in
the input data.

## New top hypothesis (C): dropout + n=1 fatal for product-form SigProb

Finding 2 above rules out an input-side expression collapse. The
remaining mechanism, consistent with all observations:

SigProb is a **product** of Hill terms over the 4 path components
(L, R, EM, T). If any one of the 4 genes has zero per-animal mean
expression in the relevant cluster, the entire path's SigProb is
zero. With **n=1 animal** per (genotype, timepoint) cell at 4mo and
6mo, dropout is fatal: scRNA dropout sparsity means a fraction of
genes will have zero counts in *any* single animal × cluster slice,
and for 4-gene paths the probability of at least one zero is high.

At 2mo, WT/App/Tau have **n=2**, so the per-condition average has a
chance to recover paths where one animal dropped out but the other
did not. ApTt at 2mo is **n=1** but still shows non-zero SigProb in
some paths because, by chance, that one ApTt animal expresses all 4
genes in enough barcodes for some paths to survive — whereas at 6mo
every cell is n=1 and the SigProb space drops to zero for the entire
path universe.

Predictions if (C) is correct:
- Per-animal SigProb at 2mo for the WT/App/Tau animals should show
  individual zeros that average up to non-zero per-condition means.
- Per-animal SigProb at 4mo/6mo should be zero for ≥99% of paths in
  every animal × cluster combination.
- For paths whose L/R/EM/T genes are all in the top-expressed tail
  (high prevalence across barcodes), per-animal SigProb should
  survive at 4mo/6mo. This is a small subset.

Implications:
- This is not fixable by changing the design matrix.
- This is not fixable by adding HEGs to the seed list.
- Possible fixes: (1) pool barcodes across animals within
  (genotype, timepoint) before computing per-condition expression so
  dropout averages out; (2) impute zeros with a small pseudocount
  before the Hill function; (3) collect more animals at 4mo/6mo;
  (4) use a different aggregation in `Expr_bygroup` (e.g. trimmed
  mean across non-zero barcodes).

## 2026-05-14 finding 3 — pipeline bug, not a statistical artifact

`alz/integration/diagnostics/recompute_sigprob_d095.py` hand-computed
per-animal SigProb for the top-10 high-|TPDS|-at-ApTt_2mo paths using
the exact Incytr Hill formula
`P = hill(L*R) * hill(R*EM) * hill(EM*T)` with K=0.5, N=2, against the
same expression matrix the wrapper exports
(`data/incytr_factorial_inputs/expression_matrix.mtx`).

For all 10 paths the hand-computed SigProb for D095 (6mo WT) and D092
(6mo ApTt) is **non-zero** — values in the 0.002 – 0.10 range, matching
the magnitudes seen at 2mo in the parquet. But the parquet's
`SigProb_WTyp_6mo` / `SigProb_ApTt_6mo` columns are **exactly 0.0** for
every one of those paths, and `SigProb_*_4mo` is **NaN** for the 4mo
animals on those same paths.

Example (path `Epha4>Fgfr1>Taf1>Chd9`, Sender=Endothelial-cell,
Receiver=Astrocytes):

| Animal | L (Epha4 in sender) | R (Fgfr1) | EM (Taf1) | T (Chd9) | Hand SigProb | Parquet `SigProb_<cond>` |
|---|---|---|---|---|---|---|
| C201 (2mo ApTt) | 1.02 | 1.40 | 0.42 | 1.86 | 0.369 | 0.203 |
| E137 (4mo ApTt) | 0.80 | 0.85 | 0.42 | 1.81 | 0.153 | **NaN** |
| D092 (6mo ApTt) | 0.80 | 0.93 | 0.33 | 1.70 | 0.107 | **0.0** |
| D095 (6mo WT)   | 0.48 | 1.01 | 0.29 | 1.77 | 0.063 | **0.0** |

The 2mo parquet value (0.203) is plausibly the mean of C201 plus other
ApTt-2mo animals (we have only one — so this is suspicious in its own
right; it may reflect a downstream normalization) but it is within an
order of magnitude of the hand value. The 4mo NaN and 6mo exact-0
have **no corresponding 0 or NaN in the underlying expression** — the
per-animal Astrocytes and Endothelial-cell mean expressions are all
healthy positive numbers, and our hand-computed Hill product is
strictly positive.

This means the late-timepoint per-condition SigProb collapse is a
**pipeline bug** somewhere between `Cal_SigProb_animal` (which is
called inside the upstream Incytr R package) and the per-condition
mean assembly written into the parquet (`sigprob_per_condition_df` or
the `populate_factorial_from_score` path in `factorial.R:2024+`). Not
a sample-size issue, not a dropout issue, not a design-matrix issue,
not an expression-magnitude issue. The animal-level SigProb values
being averaged into `SigProb_<geno>_<tp>` must be getting zeroed or
NaN'd for 4mo/6mo animals before that average is taken.

Status of prior hypotheses:
- (A), (B), (C) all falsified by this finding.
- New target: locate the code path in upstream `incytr` that zeros
  per-animal SigProb at 4mo/6mo. Candidate sites: `min_cells`
  filtering inside the streaming engine
  (`score_factorial_paths` in `~/Projects/work/incytr/R/factorial.R`
  line 1717), per-pair NA handling in the streaming sigprob_mat
  assembly (`populate_factorial_from_score` line 2024+), or the
  `cutoff_SigProb`-dependent gating in
  `duckdb_construct_candidates_receiver` (line 905+).

## Proposed next-step diagnostic (pipeline-bug hunt)

To localize the bug, the streaming pair pipeline needs to be re-run
once with the per-animal SigProb matrix preserved on disk:

1. Re-run a single (sender, receiver) pair (e.g. Endothelial-cell →
   Astrocytes) with `score_factorial_paths(..., return_sigprob_mat =
   TRUE)` so the per-animal sigprob_mat is dumped to disk for
   inspection.
2. Read the dumped sigprob_mat and check column D095 for the top-10
   high-TPDS paths. Compare to hand-computed values from
   `recompute_sigprob_d095.py`.
3. If sigprob_mat already shows zeros for D095, the bug is upstream
   of the per-condition mean step — somewhere in `Cal_SigProb_animal`
   or `Expr_bygroup_animal` as called from `score_factorial_paths`.
4. If sigprob_mat shows non-zero for D095 but the per-condition column
   shows zero, the bug is in `sigprob_per_condition_df` or
   `populate_factorial_from_score`'s reconstruction of `object@SigProb`.

The diagnostic scripts `aptt_late_collapse_m1_vs_m2.py`,
`dropout_coverage.py`, and `recompute_sigprob_d095.py` under
`alz/integration/diagnostics/` are preserved as audit trail.

## Observation

In the Song Incytr factorial output
(`outputs/reports/incytr_factorial_5xfad_kldata/receiver_cache/`), the count
of high-|PDS| pathways and the mean |TPDS| both **fall sharply** between
`ApTt_2mo` and `ApTt_4mo` / `ApTt_6mo`, despite:

- kinase activity (MEA NES) escalating monotonically across the same
  timepoints, and
- App-only and Tau-only contrasts showing real, growing signal at 4mo and
  6mo.

Expectation from the biology: combined-genotype phenotype should be
synergistic and should escalate with disease progression. The observed
collapse looks like sub-additivity, which is the same pattern flagged in
[[project_aptt_subadditivity]].

## What we've ruled out

1. **Seed-list mechanics.** The candidate-gene pool is constant across all
   9 contrasts (`L_deg = 267,348` for every contrast). Per-contrast TPDS
   variation is therefore entirely OLS-driven, not seed-driven.
2. **Pseudobulk magnitude.** For the 1,509 driver genes from the
   high-|TPDS|-at-ApTt_2mo pathway set, the per-gene pseudobulk diff
   (ApTt − WT, log1p-CP10K) is ~0.11 at every timepoint with <0.2% of
   genes crossing |Δ|>0.5. The transcriptome of the surviving ApTt
   animals at 4mo/6mo is genuinely close to WT at the gene level.
3. **HEG-augmentation hypothesis.** Adding the paper's HEG step
   (`docs/incytr_heg_seed_experiment_plan.md`) will enlarge the pathway
   universe but cannot rescue the late-ApTt collapse, because (1) above
   shows the seed list is not the bottleneck.

## What we believe is happening

Two non-exclusive mechanisms, in decreasing order of confidence:

### (A) Sample size at the ApTt × late-timepoint cells

`data/incytr_factorial_inputs/animal_metadata.csv` shows **n=1 ApTt animal
per (4mo, 6mo) cell** (males-only, post-outlier-exclusion; 13 females
dropped upstream). At n=1 the OLS contrast estimate equals the empirical
single-animal pseudobulk diff to the WT mean. There is no statistical
machinery that can lift a small empirical diff into a large contrast
estimate.

This explains *part* of the collapse but does not by itself explain why
ApTt_2mo — which also has small n — yields large TPDS. So (A) is
necessary but not sufficient.

### (B) Design misspecification: time-invariant `Int` term

`DESIGN_COLUMNS` in `alz/integration/config_integration.py` is:

```
const, App, Tau, Int, time_4mo, time_6mo,
App_x_time4, App_x_time6, Tau_x_time4, Tau_x_time6
```

Note: **no `Int_x_time4` / `Int_x_time6`**. The model assumes the
combined-genotype synergy (`Int`) is **time-invariant**. The
`ApTt_4mo` contrast vector `[0,1,1,1,0,0,1,0,1,0]` reconstructs:

```
ApTt_4mo_effect = App + Tau + Int + App_x_time4 + Tau_x_time4
```

`Int` is estimated primarily from the `ApTt_2mo − App_2mo − Tau_2mo`
residual (where each component has at most a handful of animals) and
then **carried forward unchanged** to 4mo and 6mo. If the true synergy
is time-varying (e.g. sub-additive at 2mo, super-additive at 6mo, or
vice versa), the constant `Int` will bias the ApTt-late contrast
estimate toward the App+Tau additive sum — visible as a "collapse"
relative to the App-only and Tau-only signals that *are* allowed to
grow via their `_x_time4` / `_x_time6` partners.

This would explain why the collapse appears specifically at 4mo/6mo
(where the model is forced to extrapolate the 2mo-estimated `Int`) and
not at 2mo (where `Int` is fit directly).

## Why (A) and (B) compose poorly

`Int` is estimated from the very cells where the data is sparsest (one
ApTt animal per timepoint). A noisy `Int` then propagates rigidly into
two more contrasts. The model has **no degrees of freedom left** to
register a 4mo- or 6mo-specific ApTt phenotype even if one exists.

## Proposed diagnostic (not yet run)

Refit each driver gene's per-animal log(SigProb) under two models:

- **M1** (current): 10-column design above.
- **M2**: M1 + `Int_x_time4`, `Int_x_time6` (12-column, saturated for
  the disease × timepoint cells).

Compare `ApTt_4mo` / `ApTt_6mo` contrast estimates under M1 vs M2:

- If **M2 estimates are materially larger** (and noisier — DOF cost is
  expected), (B) is confirmed and the collapse is partly a
  misspecification artifact. Next step would be a design-matrix
  expansion, accepting the variance hit.
- If **M2 lands where M1 did**, (A) is the whole story: the ApTt-late
  transcriptome is genuinely WT-like and there is no signal to recover
  without more animals.

Runs against `data/incytr_factorial_inputs/` plus one receiver's path
table (Astrocytes recommended). No full pipeline rerun needed.

## What this is not

- Not a kinase-pipeline / MEA / stoichiometry problem. The collapse is
  on the **TPDS** (transcript) side of Incytr, downstream of factorial
  OLS on per-animal SigProb.
- Not a seed-list problem (see "ruled out" §1).
- Not fixable by the HEG plan in
  `docs/incytr_heg_seed_experiment_plan.md` (orthogonal change).

## 2026-05-14 finding 4 — bug located (no rerun needed)

Disambiguating the two suspect sites (`sigprob_per_condition_df` vs
`populate_factorial_from_score`) by reading the streaming engine end-to-end
turned up the actual root cause one frame higher up the stack:

**`~/Projects/work/incytr/R/factorial.R:1900`** —

```r
keep <- rowSums(!is.na(sigprob_mat)) >= ncol(design)
receiver_candidates <- receiver_candidates[keep]
sigprob_mat <- sigprob_mat[keep, , drop = FALSE]
sigprob_mat[is.na(sigprob_mat)] <- 0     # silently coerce NAs to zero
```

Inside `score_factorial_paths` → `score_receiver`. Cascade:

1. `Expr_bygroup_animal(min_cells = 5)` (factorial.R:51-52) drops animal ×
   cluster combos with fewer than 5 cells →
   `animal_expr[[group]][[animal]] = NULL`.
2. Per-animal SigProb loop (factorial.R:1875-1895): when `r_expr` or
   `s_expr` is NULL the inner loop `next`s, leaving
   `sigprob_mat[path, animal] = NA`.
3. The `keep` gate (line 1897) requires ≥ `ncol(design) = 10` non-NA
   columns; rows with only a handful of NA animals still pass.
4. **Line 1900 turns the remaining NAs into 0.**
5. `factorial_wide_results` (line 1911) writes `SigProb_ref/alt_<contrast>`
   from the zeroed matrix → biased LFC/aFC.
6. `populate_factorial_from_score` (line 2092) calls
   `sigprob_per_condition_df` on the *same* zeroed matrix → `rowMeans(..., na.rm = TRUE)`
   now averages zeros, not NAs. For n=1 conditions (ApTt_4mo, ApTt_6mo,
   etc.) the lone animal *is* the mean, so its NA→0 collapse propagates
   directly into the per-condition column.
7. Line 1906 fits OLS on `log(0 + 1e-10) = -23.03` for the zeroed cells,
   pulling late-timepoint coefficients toward the floor — this is why
   the collapse is **global** (every late-timepoint contrast), not
   ApTt-specific.

`Cal_SigProb_animal` (factorial.R:78-106), used by the legacy
`Contrast_SigProb` code path, does **not** zero NAs — it preserves them
and lets `rowMeans(..., na.rm = TRUE)` correctly skip missing animals.
The streaming engine is the only path with the bug, and that is exactly
the path the wrapper uses (`alz/integration/factorial.R:145`).

Quantitative consistency check — for `Epha4>Fgfr1>Taf1>Chd9`,
Endothelial-cell → Astrocytes:

| Animal | Condition | Hand SigProb | Parquet | Mechanism |
|---|---|---|---|---|
| C201 | ApTt_2mo | 0.369 | 0.203 | n=2, second animal pulls mean down (ok) |
| D092 | ApTt_6mo | 0.107 | **0.0** | n=1, animal's NA was zeroed |
| D095 | WTyp_6mo | 0.063 | **0.0** | n=1, animal's NA was zeroed |
| E137 | ApTt_4mo | 0.153 | **NaN** | likely same root, path dropped at upstream gate; revisit |

`NaN` rendering (R `NA_real_` → pyarrow null → pandas NaN) at 4mo is
consistent with `cond_animals` being empty for that condition in the
per-pair slice — secondary effect of the same line-1900 mechanism plus
upstream filtering.

### Fix sketch (upstream `incytr`, not in this repo)

Two options, with the second strongly preferred:

1. **Remove the NA→0 step.** Drop line 1900 entirely and let `rowMeans`,
   `factorial_fit_ols`, and `factorial_wide_results` handle NA via
   `na.rm = TRUE` / pattern-keyed OLS (as `Contrast_SigProb` already does).
   `factorial_fit_ols` would need to be hardened to skip NAs.
2. **Tighten the `keep` gate.** Require `rowSums(!is.na(sigprob_mat)) ==
   ncol(sigprob_mat)` (no missing animals at all), and on that condition
   line 1900 becomes a no-op. Loses paths where any one animal × cluster
   has < min_cells, which is the honest answer for n=1 conditions.

Either fix must be applied at the upstream package, not via a wrapper
patch — the bug is inside `score_factorial_paths`'s closed scope.

## 2026-05-14 finding 5 — blast radius (upper bound)

`alz/integration/diagnostics/na_zero_blast_radius.py` swept all 19
receivers in `outputs/reports/incytr_factorial_5xfad_kldata/` and
counted, per (receiver, condition), how many `SigProb_<cond>` cells in
the per-condition parquet columns are exactly `0.0` or `NaN`.

Per-condition `SigProb_<cond>` fraction-that-is-(0.0 or NaN), pooled
across all 19 receivers (one row per path × receiver):

| Condition | n_animals | mean frac (0 or NaN) |
|---|---|---|
| Ttau_4mo | 1 | 0.9999 |
| WTyp_6mo | 1 | 0.9997 |
| AppP_4mo | 1 | 0.9980 |
| Ttau_6mo | 1 | 0.9979 |
| ApTt_4mo | 1 | 0.9976 |
| ApTt_6mo | 1 | 0.9973 |
| WTyp_4mo | 1 | 0.9971 |
| AppP_6mo | 1 | 0.9971 |
| ApTt_2mo | 1 | 0.9869 |
| WTyp_2mo | 2 | 0.9867 |
| Ttau_2mo | 2 | 0.9840 |
| AppP_2mo | 2 | 0.9793 |

This is an **upper bound**, not a clean blast-radius. The two
populations are intermingled in the zeroes:

- **Bug-zero** — per-animal SigProb was NA before line 1900 zeroed it.
  Mechanism in this dataset is dropout (a path gene has zero
  expression in *some* cells of the animal × cluster), not
  `min_cells = 1` (every animal × cluster has ≥ 1 cell, so the
  upstream `Expr_bygroup_animal` gate fires for nobody).
- **Bio-zero** — per-animal SigProb was legitimately `0` because at
  least one of the 4 path genes truly has zero per-animal-cluster mean
  expression. Hill is strictly positive on positive inputs, so the
  product is zero only when an input is zero.

The hand-computation case study (finding 3) confirms at least *some*
of the parquet-zeros are bug-zeros (D092 at 6mo, hand-SigProb=0.107,
parquet=0.0). The 98–99% upper bound is loose; the true bug share is
unknown without either re-running with the fix or doing per-path
forensic hand-computation. The split matters because legitimate
zero-SigProb paths are uninteresting; bug-zero paths are the ones
whose downstream TPDS / log2FC are wrong.

### Implication for downstream consumers

Until the upstream fix lands and the factorial pipeline is re-run,
**any per-condition `SigProb_<cond>` value that is exactly `0.0` or
`NaN` should be treated as "data quality unknown,"** not "no signal."
The factorial OLS coefficients (TPDS, log2FC, aFC) downstream of those
cells are biased toward the floor for every contrast that loads on
the affected animals — the bias is strongest in n=1 conditions
(every late-timepoint condition in this design, plus ApTt_2mo).

The high-|TPDS| set (`|TPDS| ≥ 0.5`) is partially insulated: those
are paths whose contrast estimate survived the NA→0 contamination,
typically because some animals on at least one side of the contrast
had non-zero hand-SigProb. The per-condition cell of a high-|TPDS|
path can still be a bug-zero (per finding 3), but the *contrast
direction* is more trustworthy than the per-condition values
themselves.

### Diagnostic outputs

- `outputs/reports/incytr_factorial_5xfad_kldata/diagnostics/na_zero_blast_radius_animal_cluster.csv` —
  per-(animal, cluster) cell counts; confirms no `min_cells=1` filter
  trips in this dataset.
- `outputs/reports/incytr_factorial_5xfad_kldata/diagnostics/na_zero_blast_radius_per_receiver.csv` —
  per-(receiver, condition) zero/NaN counts.
- `outputs/reports/incytr_factorial_5xfad_kldata/diagnostics/na_zero_blast_radius_summary.csv` —
  the table above.

### Next sharper diagnostic (deferred)

To split bug-zero vs bio-zero without a rerun, hand-compute per-animal
SigProb for every high-|TPDS| path's lone-animal cells and compare to
parquet. A cell with all 4 path-gene expressions > 0 in the animal ×
cluster but parquet = 0.0 is a definite bug-zero. The
`recompute_sigprob_d095.py` script does this for 10 paths in
Astrocytes → all 4 lone-animal cells were bug-zeros. Scaling to the
full high-|TPDS| set is mechanical and bounded by ~50k paths.

## 2026-05-14 finding 6 — upstream patch filed

Bug + fix proposal written to:
`~/Projects/work/incytr/docs/incytr_proposals/score_factorial_paths_na_to_zero_bug.md`.
Recommends pattern-keyed NA-aware OLS (port from `Contrast_SigProb`)
with the tighten-the-keep-gate fallback. Awaiting upstream review.

## 2026-05-14 finding 7 — Hill regime mismatch (larger than the bug)

While quantifying the blast radius of the NA→0 bug (finding 5), a
second and larger problem became visible: **most per-animal SigProb
values are near zero for a structural arithmetic reason that has
nothing to do with the bug.**

### The setup

`SigProb = hill(L·R) × hill(R·EM) × hill(EM·T)`, with `hill(x) = x² /
(x² + K²)` at K=0.5, N=2. Hill needs inputs of order ~1+ to "turn on":
`hill(0.5) = 0.5`, `hill(1) = 0.8`, `hill(2) = 0.94`, and `hill(0.1) ≈
0.04`.

### Problem 1 — input regime

Expression matrix is `log1p(CP10K)`. The per-animal-cluster mean
across the ~100–500 cells in one (animal, cluster) is dominated by
dropout — for any given gene, most cells in that cluster are zero, so
the mean is small.

Concrete distribution, C201 × Astrocytes (129 cells, typical case):

| percentile | per-animal-cluster mean expression |
|---|---|
| 50% | 0.002 |
| 90% | 0.26 |
| 95% | 0.78 |
| max | 5.6 |
| fraction == 0 | 49.8% |

Plug those into the Hill product:

| All 4 genes at... | SigProb |
|---|---|
| 0.07 (median nonzero) | ≈ 4 × 10⁻¹² |
| 0.78 (p95) | 0.21 |
| 1.5 | 0.87 |
| 2.5 | 0.98 |

To get a SigProb ≥ 0.1, you need all four path genes simultaneously
in the top ~5% of expressed genes in their respective clusters. The
probability of that is roughly `(0.05)⁴ ≈ 6×10⁻⁶`. Most paths
mechanically produce SigProb ≈ 0; the high-SigProb tail rides on the
handful of paths routed through highly-expressed gene quartets.

This is the dominant reason ~98% of per-condition `SigProb_<cond>`
values are near zero — **not the NA→0 bug**. The bug adds fabricated
zeros on top of legitimate near-zeros, but the near-zeros are mostly
real arithmetic.

### Why the paper doesn't show this

The paper's Incytr aggregates expression at the **condition** level —
pool all cells across all animals in one (condition, cluster) into a
single mean. Deeper averaging (tens of thousands of cells per group)
washes out dropout and pulls per-gene means up into the ~1+ range
where Hill is active. The factorial mode aggregates at the
**animal** level so OLS across animals is possible, which keeps cell
counts at ~100–500 per group — squarely in Hill's dead zone.

### Does going pairwise fix this?

No. "Paper-mode" pairwise still aggregates over (condition, cluster).
In this dataset:

| Conditions | n animals | typical pooled cells per cluster |
|---|---|---|
| 2mo WTyp / AppP / Ttau | 2 | ~200–600 |
| Every other condition (all 4mo + 6mo + ApTt_2mo) | 1 | ~100–500 |

For the 9 of 12 conditions with n_animals == 1, "pooling all animals
in the group" *is* the per-animal value — same cell budget factorial
mode uses, same Hill dead-zone. The three n=2 cells double cell
counts, which marginally lifts means but doesn't pull a
mostly-zero distribution past K=0.5.

The Hill regime problem is **inherent to the cell-count budget of
this dataset, not to factorial mode**. Switching to pairwise loses
the contrast machinery without gaining cells.

### Possible structural fixes (deferred, not in scope for now)

1. **Re-calibrate K downward** (e.g. K = 0.1 or K = 0.05) so Hill
   saturates at the input range actually produced by log1p(CP10K)
   per-animal pseudobulk means. One parameter, upstream-localized,
   recoverable. Best bang-for-buck if upstream is open to it.
2. **Aggregate expression at the condition level for SigProb**,
   recover per-animal variance separately. Loses the per-animal
   SigProb signal that drives OLS — would require a more invasive
   redesign than (1).
3. **Pool across timepoints within genotype**, or across genotypes
   within timepoint, to fatten cell counts. Sacrifices the temporal
   or disease-axis resolution the contrast set was designed for. Not
   a real option for this analysis.
4. **Accept current setup, treat SigProb as relative-only.** The few
   non-zero paths are rank-able, and contrast direction (TPDS sign)
   is meaningful when both sides have signal. The NA→0 fix becomes
   more important under this stance because it's the thing
   distinguishing real-zero from bug-zero among the small set of
   non-zero paths that carry information.

### Implications for the NA→0 fix

The NA→0 fix from `~/Projects/work/incytr/docs/incytr_proposals/score_factorial_paths_na_to_zero_bug.md`
is still worth landing — it removes a falsification — but it will
**not** make the late-timepoint TPDS collapse go away. Most of that
collapse is Hill regime, not the bug. Anyone re-running after the fix
should expect modest changes in the high-|TPDS| paths' per-condition
columns and TPDS coefficients, not a wholesale recovery of late-
timepoint signal.

### Reference for follow-up

Upstream design discussion of the per-animal vs paper aggregation
trade-off lives in `~/Projects/work/incytr/docs/factorial_vs_paper.md`
and `~/Projects/work/incytr/docs/incytr_proposals/per_animal_factorial_design.md`.
Re-read both before any structural change to the factorial mode's
SigProb step.

## 2026-05-14 finding 8 — ApTt-late collapse has a second cause: contrast geometry

While debugging the Hill regime (finding 7), the diagnostic showed a
specific asymmetry the Hill explanation cannot account for. From the
per-contrast high-|TPDS| counts:

| Contrast | high-|TPDS| paths (|TPDS| ≥ 0.5) |
|---|---|
| App_2mo, App_4mo, App_6mo | ~20,000 each |
| Tau_2mo, Tau_4mo, Tau_6mo | ~20,000 each |
| ApTt_2mo | 19,758 |
| **ApTt_4mo** | **1,812** |
| **ApTt_6mo** | **1,502** |

The ~10× drop **only** at ApTt_4mo and ApTt_6mo (while ApTt_2mo is in
the same range as the single-genotype contrasts) is not explained by
the Hill regime (symmetric across contrasts) or by the NA→0 bug
(symmetric across n=1 conditions).

### Source: missing `Int_x_time` interactions in the design

Current design columns (`alz/integration/config_integration.py:33-38`):

```
const, App, Tau, Int, time_4mo, time_6mo,
App_x_time4, App_x_time6, Tau_x_time4, Tau_x_time6
```

No `Int_x_time4` / `Int_x_time6`. The model assumes the App+Tau
synergy is constant across time. Each contrast vector sums a
different number of coefficients:

| Contrast | Coefficients summed | # nonzero |
|---|---|---|
| App_4mo | App + App_x_time4 | 2 |
| Tau_4mo | Tau + Tau_x_time4 | 2 |
| ApTt_2mo | App + Tau + Int | 3 |
| **ApTt_4mo** | App + Tau + Int + App_x_time4 + Tau_x_time4 | **5** |
| **ApTt_6mo** | App + Tau + Int + App_x_time6 + Tau_x_time6 | **5** |

ApTt at 4mo/6mo is the only contrast that sums 5 coefficients.
Combined with `Int` being constrained time-invariant, the OLS-fit
`β_Int` is a shared parameter across all three ApTt observations
(C201, E137, D092). The reported TPDS for ApTt_4mo is therefore a
**shrunken estimate** pulled toward the average of ApTt across all
timepoints, not what E137 alone showed at 4mo. The number does not
correspond to E137's data, nor to any well-defined population
quantity that the design can support.

The summed-5-coefficients arithmetic also produces extra cancellation
toward zero in the contrast estimate, which the `logi()` map then
flattens further. Result: an order-of-magnitude drop in
high-|TPDS| path counts at ApTt_4mo / ApTt_6mo.

## 2026-05-14 finding 9 — proposed fix: cell-means parameterization

Discussed and recorded for later evaluation. Not yet implemented.

### Why not "remove" or "drop" the ApTt-late contrasts

- Dropping ApTt_4mo / ApTt_6mo from the contrast set produces a table
  with no answer for the late-timepoint combined-genotype effect,
  which is one of the experiment's central questions. Not viable.
- Reporting some contrasts via factorial OLS and others via a
  fundamentally different calculation (e.g. pairwise log-ratios for
  the n=1 cells, OLS for the rest) breaks cross-contrast
  comparability — TPDS values are no longer on the same scale and
  the output table is no longer self-consistent.

### Cell-means coding (the proposed fix)

Replace the current 10-column factorial-coded design with one column
per (genotype × timepoint) cell. 12 cells, 12 columns, no intercept:

```
WTyp_2mo, AppP_2mo, Ttau_2mo, ApTt_2mo,
WTyp_4mo, AppP_4mo, Ttau_4mo, ApTt_4mo,
WTyp_6mo, AppP_6mo, Ttau_6mo, ApTt_6mo
```

Each animal's row has a 1 in its own cell, 0 elsewhere. With 15
animals and 12 columns, df_resid = 3 — variance pooled from the
three n=2 cells (WTyp_2mo, AppP_2mo, Ttau_2mo).

Every contrast becomes a two-cell difference, computed identically
across all contrasts:

| contrast | cell-means vector |
|---|---|
| App_2mo | `AppP_2mo − WTyp_2mo` |
| App_4mo | `AppP_4mo − WTyp_4mo` |
| ApTt_2mo | `ApTt_2mo − WTyp_2mo` |
| ApTt_4mo | `ApTt_4mo − WTyp_4mo` |
| ApTt_6mo | `ApTt_6mo − WTyp_6mo` |
| … (all 9) | always (alt_cell − ref_cell) |

### What this fixes

- **OLS fit for each cell-means coefficient = mean SigProb in that
  cell.** For n=1 cells, the coefficient is literally the single
  animal's value. No shrinkage, no time-invariance contamination, no
  silent extrapolation.
- **ApTt_4mo TPDS = `logi(log(E137) − log(E050))`** — exactly what
  the data says, in the same scale and via the same formula as
  App_2mo TPDS. Fully comparable across contrasts because the
  calculation is identical for all 9.
- **No asymmetric collapse**: every contrast is a 2-cell difference,
  same coefficient count, same logi-compression regime.

### What it preserves

- Same OLS framework, same SE/t-test/permutation machinery.
  Variance pooled from the 3 n=2 cells and applied to all
  contrasts — that's an explicit homoscedasticity assumption,
  same one already baked into the current design but now visible.
- Same parquet schema (TPDS, SE, pvalue, log2FC, aFC, perm_pvalue
  per contrast). Drop-in replacement.

### What it loses

- The named-effect parameterization (`App`, `Tau`, `Int`,
  `App_x_time4`, etc.) is gone from the design matrix. To recover
  synergy, derive it as a 4-cell contrast:

  ```
  Int_2mo = ApTt_2mo − AppP_2mo − Ttau_2mo + WTyp_2mo
  Int_4mo = ApTt_4mo − AppP_4mo − Ttau_4mo + WTyp_4mo
  Int_6mo = ApTt_6mo − AppP_6mo − Ttau_6mo + WTyp_6mo
  ```

  Three time-resolved synergy contrasts. All under the same method.
  All comparable. Each uses pooled variance from the 3 n=2 cells.
  Time-varying synergy can be expressed as `Int_4mo − Int_2mo` etc.
  if desired (8-cell combinations — noisiest of the set, honestly).

- One residual DOF compared to the current factorial-coded design (3
  vs 5). The current design spends those 2 df enforcing
  time-invariant Int and additive interactions — exactly the
  constraint that produces the ApTt-late collapse. Spending them on
  per-cell identification instead is the trade.

### Implementation scope (estimate)

Wrapper-side only. Upstream Incytr engine already takes a generic
design matrix and generic contrast vectors.

1. `alz/integration/config_integration.py` — replace `DESIGN_COLUMNS`,
   `MUTANT_TO_DESIGN`, `TIMEPOINT_TO_DESIGN`, and `FACTORIAL_CONTRASTS`
   with cell-means versions.
2. `alz/integration/export_factorial_inputs.py` — generates the
   design matrix from the new config; emits 12 columns instead of 10.
3. `alz/integration/factorial.R` — no change; consumes the new design
   via `inputs$design`.
4. Add synergy contrasts (`Int_2mo`, `Int_4mo`, `Int_6mo`) if
   desired; optionally `dInt_4mo_vs_2mo`, `dInt_6mo_vs_2mo` for the
   time-varying synergy question (these will be the noisiest — that
   is the honest answer for n=1 ApTt per late timepoint, not a
   modeling failure).

### Open questions before implementing

- Confirm `factorial_fit_ols` (upstream `R/factorial.R:1230-1276`)
  works without an intercept column. It expects `crossprod(design)`
  to be invertible — cell-means design should be full rank, but
  worth a smoke test.
- Confirm the permutation null is well-defined under cell-means
  (the permutation reshuffles animal labels; the design is
  invariant to within-cell relabeling, so the null is identical to
  the factorial-coded case modulo column ordering).
- Decide policy for synergy contrasts: report `Int_2mo` / `_4mo` /
  `_6mo` as standard contrasts, or treat them as derived quantities
  in a separate post-hoc step?
- The NA→0 fix from
  `~/Projects/work/incytr/docs/incytr_proposals/score_factorial_paths_na_to_zero_bug.md`
  is still required regardless of design choice — without it, n=1
  cells with any path-gene NA still get falsified to zero before
  the cell-means OLS sees them.

### Status

Recorded for later evaluation. Not yet implemented. Order of
operations once committed: (a) land upstream NA→0 fix; (b) implement
cell-means config; (c) full rerun with both fixes; (d) compare
high-|TPDS| path counts across all 9 contrasts to confirm the
asymmetric collapse is gone.

## Pointers

- Contrast geometry: `alz/integration/config_integration.py` lines
  33–64.
- Per-animal SigProb OLS: `~/Projects/work/incytr/R/factorial.R`
  `Contrast_SigProb()` (~lines 100–260).
- `logi()` saturation: `~/Projects/work/incytr/R/math.R` line 8.
- Animal counts: `data/incytr_factorial_inputs/animal_metadata.csv`.
- Output under audit:
  `outputs/reports/incytr_factorial_5xfad_kldata/receiver_cache/`.
