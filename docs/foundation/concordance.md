# Concordance Model

This document specifies how transcriptomic concordance evidence is computed,
weighted, and used throughout the kinase attribution pipeline. Concordance is
the mechanism by which kinase activity changes (from phosphoproteomics) are
linked to specific cell types (from snRNA-seq).

> **Headline vs direction (read first).** The tiers this document computes are the
> **disease-direction** concordance tiers. Since the exclusivity-pill refactor they
> are snapshotted into `direction_tier` / `direction_basis` and are **info-only**.
> The headline `confidence_tier` shown in the viewer is the **cell-type exclusivity
> pill**, computed downstream in `alz/bulk_mea/specificity_class.py` and specified in
> [`specificity_confidence.md`](./specificity_confidence.md). Everywhere below, read
> "confidence tier" as the `direction_tier` computation (`alz/bulk_mea/confidence.py`).

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

### Step 1: Direction checks

For each (kinase, cell type, contrast) triple, a directional concordance
check is computed from each available source:

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

A **positive** value means the kinase activity direction (up or
down) matches the gene expression direction in that cell type — the cell type
is a plausible source. A **negative** value means they disagree. These values
are internal gate inputs, not exported evidence scores.

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

### Step 3: Internal direction gate

The two direction checks are combined internally using the configured weights:

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

**Gate:** If the internal weighted direction check is `≤ 0`, the attribution is rejected. The
kinase activity direction contradicts the expression evidence for this cell
type.

### Step 4: Direction tier

The tier this computation exports is the categorical `direction_tier` (with
`direction_basis`). `assign_confidence` writes it as `confidence_tier`, and
`assign_specificity_class` then snapshots it into `direction_tier` before
overwriting the headline `confidence_tier` with the exclusivity pill. For display
ordering, code sorts by that tier and then by explicit evidence columns: Song
location specificity, decomposition agreement, WMB/human location support, and raw
Song/SEA-AD LFC magnitudes. WMB no longer gates the evidence table or drives a
synthetic score.

## Direction tiers

After the internal direction gate passes, the direction tier is
assigned based on the strength and source of evidence:

### High confidence
- Song contributed to concordance (`source` is `"song"` or `"both"`)
- Song LFC exceeds the minimum threshold
- Song cell-type specificity ≥ 2× uniform over the Levy-T5 spine

### Moderate confidence
- Song contributed but one strict Song gate is missing, OR
- SEA-AD-only concordance (capped at moderate regardless of WMB tier)

### Low confidence
- Weak evidence from all sources

### None (rejected)
- internal weighted direction check `≤ 0` (direction mismatch)

**Key constraint:** SEA-AD-only and WMB-only evidence can never reach "high"
confidence. Only Song-supported concordance with Song cell-type localization
(same-species, same-cohort) qualifies.

## Evidence basis labels

Each attributed (kinase, cell type) pair receives an evidence basis label
describing which sources contributed:

| Label | WMB | SEA-AD | Song | Interpretation |
|-------|-----|--------|------|----------------|
| `three_way` | ✓ | ✓ | ✓ | Strongest: Song-localized, WMB cross-checked, concordant in both human and mouse transcriptomics |
| `within_cohort` | optional | — | ✓ | Strong: same-cohort mouse evidence + Song cell-type specificity |
| `cross_species` | ✓ | ✓ | — | Moderate: human concordance + expression specificity, no paired mouse data |
| `mouse_expression_only` | ✓ | — | — | Weak: expressed in cell type but no concordance evidence |
| `song_only` | — | — | ✓ | Song concordance without WMB expression specificity |
| `human_concordance_only` | — | ✓ | — | SEA-AD concordance without WMB expression specificity |
| `weak` | — | — | — | Below all thresholds |

Thresholds:
- Song location: `song_specificity ≥ 1/N_CELL_TYPES` (above uniform), high at `2/N_CELL_TYPES`
- WMB: `wmb_specificity ≥ wmb_specificity_uniform()` (above retained-class uniform; cross-check)
- SEA-AD: `|sea_ad_lfc| > SEA_AD_LFC_MIN` (default 0.1)
- Song: `|song_lfc| > SONG_LFC_MIN` (default 0.1)

## Downstream usage

### Unified attribution table (`unified_attribution.csv`)

Contains the categorical confidence tier, confidence basis, concordance source,
and raw evidence columns for every (kinase, cell type, contrast) triple. It
does not export a synthetic score or the internal direction-gate value.

### Hypothesis tables (`attribution_recovery.py`)

**Table 2 (celltype_evidence_table.csv):** Static per-(kinase, cell type)
evidence. WMB fold, SEA-AD LFC, Song LFC, evidence basis, WMB tier.

**Table 3 (kinase_hypothesis_table.csv):** Top 3 cell types per kinase,
ranked by confidence tier and explicit evidence columns.

`has_high_conf_attribution` is `True` if any cell type has WMB tier "high"
AND concordance evidence from either source (`|song_lfc| > 0.1` OR
`|sea_ad_lfc| > 0.1`).

### Interactive viewer (`build_unified_viewer.py`)

The viewer renders the categorical confidence tier and raw support columns.
It does not expose a concordance score or an attribution evidence score.

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

All three evidence sources land on the **levy_t5 31-cluster spine**
(`config.CLUSTER_SPINE`, `CLUSTER_SPINE_NAME = "levy_t5"`) — the Song snRNA-seq
cluster taxonomy. Each source provides what it can and contributes `n/a` for
clusters it cannot witness, rather than being silently dropped. References attach
to the spine through **1-hop** bridges only; chained mappings (e.g. SEA-AD → WMB →
cluster) are forbidden — see [`cohort_contract.md`](./cohort_contract.md) §6.3.

- **Song (spine):** native 31 clusters. For the exclusivity pill the clusters are
  folded into 17 curated specificity units (see
  [`specificity_confidence.md`](./specificity_confidence.md)); for direction
  concordance the per-cluster disease LFC is used directly.
- **WMB:** 338 region-specific subclasses → 34 WMB classes; corroborates at the
  WMB **class** of the spine cluster's home, via
  `data/derived/bridges/cluster_to_wmb_class.csv`.
- **SEA-AD:** 139 MTG supertypes roll onto the spine via
  `data/derived/bridges/cluster_to_seaad_supertype.csv` (weighted). SEA-AD has no
  MTG coverage of hippocampal CA, dentate granule, striatum, olfactory bulb, or
  cerebellum, so spine clusters that map only to those carry `sea_ad_lfc = n/a`.
- **HBCA:** whole-brain superclusters roll onto the spine via
  `data/derived/bridges/cluster_to_hbca_supercluster.csv` (weighted).

## Implementation locations

| Component | File | Function/Lines |
|-----------|------|----------------|
| Internal direction gate | `alz/bulk_mea/attribute.py` | `_assemble_unified()` |
| Confidence + evidence basis | `alz/bulk_mea/confidence.py` | `assign_confidence()` |
| Direction-source weighting | `alz/bulk_mea/attribute.py` | `_assemble_unified()` |
| SEA-AD pathway matching | `alz/bulk_mea/attribute.py` | `_assemble_unified()` |
| Pipeline orchestration | `pixi.toml` | `attribute` task invokes `alz/bulk_mea/attribute.py` |
| Song pathway-specific LFCs | `alz/reference/snrna_integration.py` | `step_concordance()` |
| Song Allen-Cell-Type → WMB class mapping | `alz/shared/config.py` | `SONG_TO_WMB_CLASS_MAP` |
| Top cell-type ranking | `alz/bulk_mea/recover.py` | `_build_kinase_hypothesis_table()` |
| High-confidence flag | `alz/bulk_mea/recover.py` | `_build_kinase_hypothesis_table()` |
| Viewer payload assembly | `alz/build_unified_viewer.py` | attribution payload block |
| Weight configuration | `alz/shared/config.py` | `SONG_CONCORDANCE_WEIGHT`, `SEA_AD_CONCORDANCE_WEIGHT` |
