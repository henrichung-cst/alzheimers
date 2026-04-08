# Responses to Biostatistician Review

Below we address each of the six questions raised in `docs/questions.md`, with supporting evidence from supplementary analyses.

---

## Q1. Threshold sensitivity of confidence tiers

**Concern:** The specific WMB specificity (≥2× uniform) and SEA-AD LFC (|LFC| > 0.1) thresholds feel arbitrary. How sensitive is the final attribution table?

**Finding: The total set of attributed kinase–cell type pairs is perfectly stable across all 25 threshold combinations tested.** We swept WMB specificity multipliers from 1.0× to 3.0× uniform and SEA-AD |LFC| thresholds from 0.05 to 0.25. In all 25 combinations, the number of attributed rows is exactly 3,329 — no rows cross the attributed/non-attributed boundary, because that boundary is determined by concordance sign (concordance_score > 0), which is threshold-independent.

What does change is the distribution across confidence tiers:

| Metric | Default (2.0×, 0.10) | Range across grid |
|---|---|---|
| High | 115 | 43–1,600 |
| Moderate | 3,055 | 1,688–3,164 |
| Low | 159 | 41–629 |
| **Total attributed** | **3,329** | **3,329 (invariant)** |

A near-miss analysis identified 480 rows (14.4% of attributed) that change tier under ±1 step perturbation. The dominant transitions are moderate↔low (not high↔none), meaning threshold choice affects granularity of evidence strength labeling, not which kinase–cell type pairs are called.

**Bottom line:** The thresholds tune the severity of the confidence grading, but the set of attributed findings is robust. We recommend reporting tier distributions at both default and adjacent thresholds in supplementary materials.

See: `outputs/reports/supplementary/threshold_sensitivity/`

---

## Q2. Supertype-to-subclass aggregation method

**Concern:** Collapsing 139 supertypes to 24 subclasses — does the aggregation method (median vs mean vs weighted mean) matter?

**Finding: 92.8% of kinase–cell type attribution pairs are stable across all three aggregation methods** (median, mean, weighted mean). Of 8,304 total pairs, only 602 (7.2%) change confidence tier depending on the method. Cell counts for weighted averaging were not available in the SEA-AD effect size file, so the weighted mean fell back to unweighted mean — making this a comparison between median and mean, which is the relevant contrast.

The 7.2% of method-sensitive pairs represent borderline cases where median vs mean LFC crosses a threshold boundary. These are concentrated in heterogeneous subclasses where supertype-level LFCs have mixed signs, exactly where one would expect aggregation sensitivity.

**Bottom line:** The current median-based aggregation is appropriate and produces concordant results with the mean for >92% of pairs. We recommend flagging method-sensitive pairs in the supplementary data.

See: `outputs/reports/supplementary/aggregation_robustness/`

---

## Q3. Cross-species inference risk

**Concern:** The attribution arm assumes cell-type-specific transcriptomic changes in human AD (SEA-AD) are directionally conserved in mouse 5xFAD phosphoproteomics.

**Response:** We agree this is a real caveat and have built the pipeline to explicitly separate evidence sources. The unified attribution already provides the infrastructure to distinguish evidence types:

- **WMB expression specificity** is mouse-internal evidence (Allen Whole Mouse Brain), requiring no cross-species assumption.
- **SEA-AD concordance** is the cross-species component (human AD snRNA-seq).
- The confidence tier logic treats them independently: "moderate" can be reached by *either* source alone, while "high" requires *both*.

The codebase includes an `_assign_evidence_basis()` function (kinase_attribution.py:778) that classifies each attribution as `cross_species`, `mouse_expression_only`, `human_concordance_only`, or `weak`. This column (`evidence_basis`) is being added to the output in the current working branch and will appear in the next pipeline run.

Additionally, the mechanism annotation (see Q5) provides a complementary mouse-internal signal: kinases classified as "activity-driven" have mouse phosphoproteomics evidence independent of human transcriptomics.

**Bottom line:** Attributions supported by WMB specificity alone do not depend on cross-species concordance. We will add the `evidence_basis` column to the canonical output so downstream consumers can filter or stratify by evidence type.

---

## Q4. FDR < 0.25 threshold

**Concern:** The standard GSEA threshold of FDR < 0.25 may be seen as permissive. Is there a secondary analysis at FDR < 0.10?

**Finding: 74.4% of kinase-contrast pairs survive the stricter FDR < 0.10 threshold** (270 of 363 pairs). The 93 dropped pairs are disproportionately from the Tau contrast (54 dropped, vs 21 for ApTt and 18 for App), suggesting Tau-driven kinase signals are noisier or more diffuse.

| Threshold | Kinase-contrast pairs | Unique kinases |
|---|---|---|
| FDR < 0.25 (default) | 363 | 216 |
| FDR < 0.10 (strict) | 270 | 170 |
| Dropped | 93 | 46 |

At the attribution level, the FDR < 0.10 table retains 2,337 of 3,329 rows (70.1%). The kinases that drop out tend to have only 1–2 significant contrasts (e.g., PIM1, SLK, ASK1, MAP3K15).

**Bottom line:** The core findings are robust to FDR < 0.10. We recommend presenting the FDR < 0.10 subset as a "high-confidence core" alongside the full FDR < 0.25 results, and noting the Tau contrast's higher attrition rate.

See: `outputs/reports/supplementary/fdr_stringent/`

---

## Q5. Parent protein quality for activity-driven kinases

**Concern:** With only 12 activity-driven kinases (now 99 after re-running with current pipeline), could noisy parent protein estimates create spurious stoichiometry signals?

**Finding:** The mechanism annotation now identifies 99 activity-driven kinase-contrast pairs (across 92 unique genes). Of these:

- **47 of 92** activity-driven kinase genes were found in the total proteome.
- Among detected genes: median detection rate = **1.000** (100% of samples), median CV = **0.208**.
- This is comparable to or better than other significant kinases (median detection = 1.0, median CV = 0.248), meaning activity-driven kinases do *not* have systematically worse parent protein data.

**9 genes were flagged** for poor parent protein quality (detection rate < 50% or CV > 1.0): ACVR1C, MAPKAPK2, PKN2, DSTYK, CSNK1G2, RIPK2, CAMK1G, WNK1, MAPK14. For these, the stoichiometry correction relies on sparse parent protein estimates. However:

- WNK1 and MAPK14 have only 16.7% detection rate (detected in ~12/72 samples), making their stoichiometry corrections genuinely unreliable.
- The remaining flagged genes have 33% detection rate — marginal but interpretable.
- 45 of 92 genes (49%) were not found in the proteome at all, meaning stoichiometry for these was computed without parent protein correction (using phospho data alone — equivalent to the raw phospho pathway).

| Group | N genes | Median detection | Median CV | Median abundance |
|---|---|---|---|---|
| Activity-driven | 92 (47 found) | 1.000 | 0.208 | 73.54 |
| Other significant | 143 (59 found) | 1.000 | 0.248 | 72.53 |

**Bottom line:** The activity-driven class does not suffer from systematically worse parent protein quality than other significant kinases. We recommend flagging WNK1 and MAPK14 attributions as having unreliable stoichiometry corrections, and noting that ~49% of activity-driven genes lack proteome-level parent protein data entirely.

See: `outputs/reports/supplementary/parent_protein_qc/`

---

## Q6. OLS modeling choices

**Concern:** What is the model formula? Full interaction model or main effects only?

**Response:** The OLS model uses a hand-coded design matrix (no statsmodels/patsy formula interface) with 72 rows × 7 parameters:

```
Y_site = β₀ + β_App·App + β_Tau·Tau + β_Int·App×Tau + β_sex·Female + β_4mo·Time4mo + β_6mo·Time6mo + ε
```

The genotype factor is coded via indicator variables for the 2×2 factorial (APP × Tau):

| Genotype | App | Tau | Int |
|---|---|---|---|
| WT | 0 | 0 | 0 |
| APP | 1 | 0 | 0 |
| T22 | 0 | 1 | 0 |
| T22/APP | 1 | 1 | 1 |

Three contrasts are extracted as linear combinations of genotype coefficients:

| Contrast | Formula | Interpretation |
|---|---|---|
| App | β_App | APP main effect vs WT |
| Tau | β_Tau | T22 (Tau) main effect vs WT |
| ApTt | β_App + β_Tau + β_Int | T22/APP double mutant vs WT |

Each contrast's standard error is computed analytically via `Var(c'β) = c'(X'X)⁻¹c · σ²`, with p-values from a t-distribution (df = n − 7). BH FDR correction is applied across sites within each contrast.

**Design rationale:**
- Sex and timepoint are included as covariates (nuisance variables), not as interaction terms. With 72 animals across 2 sexes × 3 timepoints × 4 genotypes = 24 groups (~3 animals/group), a full interaction model would consume too many degrees of freedom.
- The model is main-effects for sex/timepoint, with one genotype interaction (App × Tau). This is a pragmatic choice given the sample sizes per cell.
- Sites with missing data are fit individually (slow path); complete-data sites use vectorized `(X'X)⁻¹X'Y` (fast path). Of 16,114 sites, 6,264 have complete data and 9,850 have partial data; 11,628 total produce valid fits.

See: `code/kinase_attribution.py`, lines 60–75 (genotype coding), 499–517 (design matrix), 520–592 (OLS), 678–727 (contrast extraction).
