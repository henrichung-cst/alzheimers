# Concordance Model

This document specifies how transcriptomic concordance evidence is computed,
weighted, and used throughout the kinase attribution pipeline. Concordance is
the mechanism by which kinase activity changes (from phosphoproteomics) are
linked to specific cell types (from snRNA-seq).

## Core question

For a given kinase showing altered activity in a disease model, *is the gene
encoding that kinase also differentially expressed in the same direction in a
particular cell type?* If so, that cell type is a candidate source of the
observed kinase activity change.

## Evidence sources

Two independent transcriptomic references provide concordance evidence:

| Source | Species | Cohort | Design | Weight |
|--------|---------|--------|--------|--------|
| **Song snRNA-seq** | Mouse | Same cohort (28 animals, paired with proteomics) | Factorial OLS on pseudobulk, males only, pooled across timepoints | **3×** |
| **SEA-AD** | Human | Allen Institute MTG (postmortem AD brain, 139 supertypes) | Differential expression, AD vs. control donors | **1×** |

Song receives 3× weight because it is same-species, same-cohort, and paired
with the proteomics animals. SEA-AD is a cross-species human proxy. Neither
source has absolute veto power — each contributes proportional to its weight.

Weights are configurable:
- `config.SONG_CONCORDANCE_WEIGHT` (default 3.0)
- `config.SEA_AD_CONCORDANCE_WEIGHT` (default 1.0)

## Computing concordance

### Step 1: Raw concordance scores

For each (kinase, cell type, contrast) triple, a directional concordance
score is computed from each available source:

```
sea_ad_cs = sign(NES) × sea_ad_lfc
song_cs   = sign(NES) × song_lfc
```

- `NES`: normalized enrichment score from MEA (kinase activity direction in
  the mouse phosphoproteomics)
- `sea_ad_lfc`: log2 fold change of the kinase gene in human AD snRNA-seq
  for this cell type
- `song_lfc`: log2 fold change of the kinase gene in the paired mouse
  snRNA-seq for this cell type

A **positive** concordance score means the kinase activity direction (up or
down) matches the gene expression direction in that cell type — the cell type
is a plausible source. A **negative** score means they disagree.

### Step 2: Pathway matching

Both sources are pathway-matched to the mouse contrast to avoid comparing
amyloid-pathway kinases against a tau-dominated transcriptomic signature:

**SEA-AD** uses stage-stratified effect sizes:
- App contrasts → `effect_sizes_early.h5ad` (early/low-CPS donors, amyloid-dominant)
- Tau contrasts → `effect_sizes_late.h5ad` (late/high-CPS, tau-dominant)
- ApTt contrasts → `effect_sizes.h5ad` (full CPS range)

Rationale: early and late SEA-AD strata have Pearson r ≈ −0.12 with ~48%
sign flips. Without stratification, an App kinase could be checked against a
tau-dominated late-stage signature that shows the opposite direction.

**Song** uses pathway-specific LFCs from factorial OLS:
- App contrasts → App genotype effect (β_App)
- Tau contrasts → Tau genotype effect (β_Tau)
- ApTt contrasts → combined genotype effect (β_App + β_Tau + β_Interaction)

Mapping: `config.SEA_AD_PATHWAY_MAP` (for SEA-AD) and
`contrast.split("_")[0]` (for Song, which maps to factorial OLS pathways
directly).

### Step 3: Weighted effective concordance

The two raw scores are combined into a single effective concordance value
using the configured weights:

```
If both available:
    effective = (3 × song_cs + 1 × sea_ad_cs) / (3 + 1)

If Song only:
    effective = song_cs

If SEA-AD only:
    effective = sea_ad_cs

If neither:
    effective = 0.0
```

The `concordance_source` is recorded as `"both"`, `"song"`, `"sea_ad"`, or
`"none"`.

**Gate:** If `effective_concordance ≤ 0`, the attribution is rejected. The
kinase activity direction contradicts the expression evidence for this cell
type.

### Step 4: Combined score (for ranking)

```
combined_score = effective_concordance × (0.5 + wmb_specificity)
```

WMB specificity amplifies concordance — a kinase that is both concordant and
cell-type-specific ranks highest. The 0.5 offset prevents zero-specificity
cell types from being completely suppressed.

## Confidence tiers

After effective concordance passes the positivity gate, confidence is
assigned based on the strength and source of evidence:

### High confidence
- Song contributed to concordance (`source` is `"song"` or `"both"`)
- WMB specificity ≥ 2× uniform (SPECIFICITY_HIGH)
- LFC evidence from Song or SEA-AD exceeds minimum threshold

### Moderate confidence
- Song contributed but WMB specificity is lower, OR
- SEA-AD-only concordance (capped at moderate regardless of WMB tier)

### Low confidence
- Weak evidence from all sources

### None (rejected)
- `effective_concordance ≤ 0` (direction mismatch)

**Key constraint:** SEA-AD-only concordance can never reach "high" confidence.
Only Song-supported concordance (same-species, same-cohort) qualifies.

## Evidence basis labels

Each attributed (kinase, cell type) pair receives an evidence basis label
describing which sources contributed:

| Label | WMB | SEA-AD | Song | Interpretation |
|-------|-----|--------|------|----------------|
| `three_way` | ✓ | ✓ | ✓ | Strongest: expression-specific, concordant in both human and mouse transcriptomics |
| `within_cohort` | ✓ | — | ✓ | Strong: same-cohort mouse evidence + expression specificity |
| `cross_species` | ✓ | ✓ | — | Moderate: human concordance + expression specificity, no paired mouse data |
| `mouse_expression_only` | ✓ | — | — | Weak: expressed in cell type but no concordance evidence |
| `song_only` | — | ✓ | — | Song concordance without WMB expression specificity |
| `human_concordance_only` | — | ✓ | — | SEA-AD concordance without WMB expression specificity |
| `weak` | — | — | — | Below all thresholds |

Thresholds:
- WMB: `wmb_specificity ≥ SPECIFICITY_LOW` (1/24 ≈ 0.042, i.e., above uniform)
- SEA-AD: `|sea_ad_lfc| > SEA_AD_LFC_MIN` (default 0.1)
- Song: `|song_lfc| > SONG_LFC_MIN` (default 0.1)

## Downstream usage

### Unified attribution table (`unified_attribution.csv`)

Contains per-row effective concordance, concordance source, combined score,
confidence tier, and evidence basis for every (kinase, cell type, contrast)
triple. This is the full granular output.

### Hypothesis tables (`attribution_recovery.py`)

**Table 2 (celltype_evidence_table.csv):** Static per-(kinase, cell type)
evidence. WMB fold, SEA-AD LFC, Song LFC, evidence basis, WMB tier.

**Table 3 (kinase_hypothesis_table.csv):** Top 3 cell types per kinase,
ranked by WMB fold then weighted concordance magnitude:

```
weighted_concordance = (3 × |song_lfc| + 1 × |sea_ad_lfc|) / 4
```

`has_high_conf_attribution` is `True` if any cell type has WMB tier "high"
AND concordance evidence from either source (`|song_lfc| > 0.1` OR
`|sea_ad_lfc| > 0.1`).

### Interactive viewer (`build_unified_viewer.py`)

The composite score builder exposes Song and SEA-AD as independent
dimensions with configurable weights. Default "balanced" preset:
Song concordance = 30, SEA-AD concordance = 10 (3:1 ratio).

## Gene mapping chain

Concordance requires mapping from kinase activity → gene symbol → cell-type
expression. The mapping chain:

```
Kinase abbreviation (e.g., PKCG)
    → gene symbol (e.g., Prkcg)          [MyGene.info, cached]
    → .upper() → PRKCG                   [case normalization]
    → match in SEA-AD obs_names           [human 1:1 orthology by name]
    → match in Song pseudobulk genes      [mouse, direct symbol match]
    → match in WMB expression matrix      [mouse, direct symbol match]
```

All matching is done via `.upper()` normalization at every junction. The
pipeline assumes 1:1 orthology between mouse and human kinase genes by
symbol name, which holds for the conserved kinome.

## Cell type taxonomy

All three evidence sources are mapped to a common spine of **34 WMB
classes** defined in `config.WMB_CLASSES` and `data/external/allen_abc/
wmb_class_manifest.csv`. Each evidence source provides what it can and
contributes `n/a` for cell types it cannot witness, rather than being
silently dropped.

- **WMB (spine):** 338 region-specific subclasses → 34 classes via direct
  group-by on `wmb_meta["class"]` (no keyword matching, no silent drops).
  Subclass-level data preserved in `wmb_kinase_expression_subclass.csv`
  for audit/tooltip.
- **SEA-AD:** 139 supertypes → 9 WMB classes via the chained mapping
  `var["Subclass"]` (24 SEA-AD subclasses) → `seaad_subclass_to_wmb_class.csv`
  (cortical neurons + glia + vascular + immune). Median LFC across all
  contributing supertypes per WMB class. SEA-AD has no MTG coverage of
  hippocampal CA, dentate granule, striatum, olfactory bulb, or cerebellum,
  so those classes carry `sea_ad_lfc = n/a`.
- **Song:** Allen Cell Type Mapper `class_name` column maps directly into
  the 34-class vocabulary (the prefix code is added via the manifest).
  Approximately 21 of 34 classes pass Song's confidence + animal-count
  gates; the rest carry `song_lfc = n/a` honestly. See
  `outputs/reports/snrna_integration/active_classes.csv` for the per-class
  Song coverage manifest.

## Implementation locations

| Component | File | Function/Lines |
|-----------|------|----------------|
| Raw concordance scores | `alz/kinase_attribute.py` | `_compute_effective_concordance()` |
| Confidence + evidence basis | `alz/kinase_attribute.py` | `_assign_confidence_and_basis_vectorized()` |
| Vectorized weighted blend | `alz/kinase_attribute.py` | `step_attribute()` |
| SEA-AD pathway matching | `alz/kinase_attribute.py` | `step_attribute()` |
| Pipeline orchestration | `alz/pipelines/attribute/{nodes,pipeline}.py` | Kedro nodes wrap pure helpers |
| CLI shim | `alz/kinase_attribute.py` | `main()` (delegates to `KedroSession`) |
| Song pathway-specific LFCs | `alz/snrna_integration.py` | `step_concordance()` |
| Song Allen-Cell-Type → WMB class mapping | `alz/config.py` | `SONG_TO_WMB_CLASS_MAP` |
| Top cell-type ranking | `alz/attribution_recovery.py` | `_build_kinase_hypothesis_table()` |
| High-confidence flag | `alz/attribution_recovery.py` | `_build_kinase_hypothesis_table()` |
| Viewer score presets | `alz/build_unified_viewer.py` | `SCORE_PRESETS` (Kinase Explorer side panel) |
| Weight configuration | `alz/config.py` | `SONG_CONCORDANCE_WEIGHT`, `SEA_AD_CONCORDANCE_WEIGHT` |
