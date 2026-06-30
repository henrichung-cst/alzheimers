# Specificity / confidence calculation

## The shared standard metric

The repo has **one cell-type attribution metric**, defined in
`alz/cross_reference/specificity.py` (`specificity.compute`), used by every
cohort. Full spec: [`docs/plans/attribution/standard_attribution_metric.md`](../plans/attribution/standard_attribution_metric.md).

In brief: a cell type is **detected** when `fraction_cells_expressing ≥ 0.10`
(count-based, normalization-free). Expression weights are de-logged to linear
(`max(2^mean_log2 − 1, 0)`). From those two foundations:

| Output | Definition |
|---|---|
| `detected` | `fraction_cells_expressing ≥ 0.10` |
| `concentration_c` | `w_c / Σ_{detected} w` — share among detected cell types only |
| `concentration_of_total` | `w_c / Σ_{all} w` — share of total linear expression |
| `concentration_tier` | first `t` in `(10, 5, 2, 1)` where `concentration_of_total ≥ t / N_total`; else 0 |
| `effective_n_celltypes` | `1 / Σ concentration_c²` over detected cell types |
| `top_celltype` / `top_concentration` | dominant detected cell type and its share |

**Cohorts on the shared metric (all call `specificity.compute`):**

- **Song within-cohort snRNA** — `alz/reference/snrna_integration.py:291`.
  Outputs: `song_detection.csv`, `song_expression_specificity.csv`.
  Per-kinase columns (prefixed `song_`): `song_detected`, `song_concentration`,
  `song_concentration_of_total`, `song_concentration_tier`,
  `song_fraction_cells_expressing`, `song_effective_n`, `song_top_celltype`,
  `song_top_concentration`.
- **WMB mouse atlas** — `alz/reference/wmb_expression.py:561` (corroborator).
- **NSCLC T-cell reference** — `alz/reference/nsclc_expression.py:417`.
- **T-cell within-cohort** — `alz/cross_reference/tcell_within_cohort.py:181`.
  Per-kinase columns: `tcell_detected`, `tcell_fraction_expressing`,
  `tcell_concentration`, `tcell_concentration_tier`, `tcell_effective_n`,
  `tcell_top_celltype`.
- **5xFAD within-cohort snRNA** — producer
  `alz/ingest/build_5xfad_snrna_attribution.R` exports per-(gene, tissue, cell
  type) detection + mean log2(x+1) to `fivexfad_snrna_expression.csv`;
  `alz/cohorts/fivexfad/snrna_specificity.py` runs `specificity.compute` per
  tissue → `fivexfad_expression_specificity.csv`. Per-kinase columns (prefixed
  `fivexfad_`): `fivexfad_detected`, `fivexfad_fraction_cells_expressing`,
  `fivexfad_concentration`, `fivexfad_concentration_of_total`,
  `fivexfad_concentration_tier`, `fivexfad_effective_n`, `fivexfad_top_celltype`.
  The legacy share / τ localizer (`fivexfad_specificity`, `fivexfad_tau`,
  `fivexfad_fold_over_uniform`) is removed.

---

## The exclusivity confidence pill

The confidence pill (`none / low / moderate / high / very_high`) answers:

> **How exclusively is this kinase expressed in a single cell type, and does an
> independent reference agree?**

The pill now spans **all three kinase-explorer cohorts** (Song, 5xFAD, T-cell)
using the same tier formula. The single shared implementation is:

```text
alz/bulk_mea/exclusivity_tier.py::exclusivity_tier(detected, eff, corroborated) → (tier, basis)
```

Constants: `EXCLUSIVE_EFF_MAX = 1.5`, `BROAD_EFF_MAX = 3.0`.

Per-cohort wiring:

| Cohort | Caller | `eff` reads | Corroborator |
|---|---|---|---|
| Song | `alz/bulk_mea/specificity_class.py::assign_specificity_class` | `song_unit_effective_n` | WMB class OR human SEA-AD/HBCA (`human_location_score ≥ 1.0`) |
| 5xFAD | `alz/viewer/cohorts/fivexfad.py::_apply_fivexfad_exclusivity_confidence` | `fivexfad_effective_n` | WMB class OR SEA-AD at the home cell type |
| T-cell | `alz/build_tcell_viewer.py::_build_tcell_attribution_index` | `tcell_effective_n` | NSCLC detection at the crosswalked home state (panel-absent → uncorroborated, caps at `moderate`) |

Direction concordance — bulk-MEA significance, snRNA LFC direction, decomp
agreement — is **info-only for all three cohorts**. For 5xFAD these are stored
as row fields `bulk_mea_significant`, `direction_concordant`,
`decomp_agrees_bulk` and shown in the detail section but do not gate the pill.
T-cell concordance (`tcell_concordant`, `tcell_consistency`) is similarly
info-only.

Song's Song-specific wiring (`assign_specificity_class`) runs after
`assign_confidence` in `alz/bulk_mea/attribute.py`. Design history of the Song
model:
[`docs/plans/attribution/cross_reference_exclusivity_regrouping.md`](../plans/attribution/cross_reference_exclusivity_regrouping.md).

---

## Song-specific mechanics

Sections 1–6 below describe the mechanics specific to the Song cohort (curated
specificity units, corroboration logic, output columns). The tier formula in §4
is identical for all three cohorts; the `eff` input and corroborator are
per-cohort as shown above.

## 1. Resolution: curated specificity units

The pill runs at **curated specificity units** (`config.load_specificity_unit_map`),
not blindly at the 31 native Song clusters nor a flat coarse rollup. The native
clusters over-split some cell types — excitatory neurons into 6 pyramidal subtypes,
interneurons into 8 — which dilutes a pan-class kinase's signal across subtypes and
makes it look non-specific. So per WMB class:

- **Collapse** (`SPECIFICITY_COLLAPSE_WMB_CLASSES`) — when a WMB class's Song
  clusters are subtypes of *one* cell type, they fold into the WMB-class **parent**
  unit. `01 IT-ET Glut` (6 excitatory), `06 CTX-CGE GABA` (8 interneuron),
  `09 CNU-LGE GABA` (striatal GABAergic), `02 NP-CT-L6b Glut` (deep excitatory).
- **Combined label** — a WMB class with exactly one Song cluster keeps that cluster
  as its unit, labelled `"cluster · class"` (e.g. `Microglia · 34 Immune`).
- **Split** — a WMB class whose Song clusters are *genuinely distinct* cell types
  stays split; each cluster is its own unit. `33 Vascular` (endothelial / pericyte /
  choroid / leptomeningeal), `30 Astro-Epen`, `31 OPC-Oligo`.

This gives **17 units over the 31 clusters**. The dominant unit is reported per
kinase (`specificity_unit_label`); its top child cluster is `specificity_celltype`;
`specificity_collapsed` flags a parent over several clusters. **The collapse is never
silent** — the viewer renders a collapsed unit as an expandable parent listing its
child clusters (§6).

The reference atlases have no cluster vocabulary, so they corroborate at the WMB
**class** level: the dominant unit's home cluster is projected to its WMB class
(`wmb_class`, the `cluster_to_wmb_class` crosswalk applied in `attribute.py`) and the
references are checked for agreement there.

---

## 2. Within-cohort Song is primary

Song is the within-cohort snRNA-seq from the same animals as the bulk MEA, so it is
the primary signal — it **sets** the tier. The pill folds the detected-set
`song_concentration` (from the shared metric — `alz/cross_reference/specificity.py`,
over detected cell types only) onto the 17 curated specificity units, then measures
exclusivity as the **effective number of units**:

```
eff = 1 / Σ (unit_share)²
```

`unit_share` is the sum of `song_concentration` values for all Song clusters that map
to that unit. `eff ≈ 1` → expressed in essentially one cell type; `eff` large → spread
across many. The **dominant unit** is the one with the largest aggregate share; its top
detected child cluster is the home cell type (`specificity_celltype`).

The prior inputs `song_location_high` / `specificity_score` / `song_tau` are retired.
The pill now reads `song_concentration` from the shared standard metric exclusively.

---

## 3. Reference data only corroborates

WMB (mouse atlas) and human (SEA-AD + HBCA) are corroborators. They can **promote**
the tier by one step when they agree with the WMB class of Song's dominant cluster,
but they are **never required** and **never veto**.

- **WMB** agrees when its true top class (`wmb_top_celltype`, forwarded from
  `alz/cross_reference/evidence.py`) equals the WMB class of the dominant unit's home
  cluster.
- **Human** agrees when the cluster with the maximum `human_location_score` has the
  same WMB class as the dominant unit's home cluster, counted **only when that score
  is strong** (≥ 1.0 log2 over the brain mean — `HUMAN_STRONG_LOG2_SPECIFICITY`).
  Human has no detection gate, only a relative location score, so a weak signal does
  not vote.

```
corroborated = (WMB top class == WMB class of the dominant unit's home cluster)
            OR (human strong AND human's top class == that same WMB class)
```

---

## 4. The tier

`eff` = Song's effective number of units; thresholds
`EXCLUSIVE_EFF_MAX = 1.5`, `ENRICHED_EFF_MAX = BROAD_EFF_MAX = 3.0`.

| Tier | Condition |
|---|---|
| **`very_high`** | Song detected, `eff ≤ 1.5`, **and** corroborated |
| **`high`** | (Song detected, `eff ≤ 3`, corroborated) **or** (Song detected, `eff ≤ 1.5`, uncorroborated) |
| **`moderate`** | Song detected, `eff ≤ 3`, no reference corroborates |
| **`low`** | Song detected but `eff > 3` (broadly expressed — not cell-type-specific) |
| **`none`** | not detected in the within-cohort Song data |

Within-cohort Song alone can reach `high` (a kinase pinned to essentially one unit,
`eff ≤ 1.5`). References only ever raise the tier by confirming the cell class.

---

## 5. Worked examples

| Kinase | Dominant unit | `eff` | Corroborated | Tier |
|---|---|---|---|---|
| **BTK** | Microglia · 34 Immune | 1.0 | WMB + human | `very_high` |
| **ALK1** | Endothelial-cell | 1.0 | WMB + human | `very_high` |
| **SYK** | Microglia · 34 Immune | 1.7 | WMB + human | `high` |
| **CDK9 / DYRK2** | Excitatory neurons (IT/ET) *(collapsed)* | ≤ 3 | WMB | `high` |
| **AKT1** | Cholinergic-Neurons · 13 CNU-HYa Glut | 2.4 | — | `moderate` |
| **CAMK2A** | Excitatory neurons (IT/ET) *(collapsed)* | > 3 | — | `low` |
| **GRK5 / LRRK2** | Interneurons (CGE) *(collapsed)* | > 3 | — | `low` |

Observed per-kinase distribution across 389 kinases:
**very_high 18 · high 39 · moderate 15 · low 246 · none 71.** The collapse rescues
kinases over-split *within* one WMB class (e.g. CDK9/DYRK2, concentrated across
IT/ET excitatory subtypes, now read as one unit → `high`). **CAMK2A stays `low`**:
even after collapse it is genuinely expressed across *several distinct* units
(excitatory IT/ET, dentate, interneurons, striatal), which the design deliberately
does not merge — so no single cell type dominates.

---

## 6. Output columns

`assign_specificity_class` writes, per row of `unified_attribution_full.csv`
(constant across a kinase's rows, since the signal is contrast-invariant):

| Column | Meaning |
|---|---|
| `confidence_tier` | the pill (`none…very_high`) |
| `confidence_basis` | human-readable reason (unit, `eff`, which references corroborate) |
| `specificity_unit` | the dominant unit's id (a WMB class for collapsed units, else a Song cluster) |
| `specificity_unit_label` | the dominant unit's display label (the pill's named cell type) |
| `specificity_celltype` | the dominant unit's top child Song cluster (home cell type) |
| `specificity_collapsed` | `True` when the unit collapses several Song clusters (viewer shows an expandable parent) |
| `song_unit_effective_n` | `eff` as used by the pill (1 / Σ unit_share², over 17 curated units; distinct from the native-cluster `song_effective_n`) |
| `direction_tier` / `direction_basis` | the prior disease-direction tier, preserved (§7) |

These flow into the viewer payload `attribution_index`, plus a static
`specificity_units` section (`cluster_to_unit` + per-unit `label` / `collapsed` /
`children`) so the viewer can render a collapsed unit as an expandable parent over
its child Song clusters. They drive the audit verdict and Kinase Explorer pills —
**no grouping is ever shown without the child clusters being one click away.**

---

## 7. Relationship to direction concordance

Direction concordance — does the kinase's activity move in the **same disease
direction** across bulk MEA, within-cohort expression, and the decomposition
layer — is **info-only for all three cohorts**. It is never the pill.

For **Song**: `assign_confidence` (`alz/bulk_mea/confidence.py`) computes the
concordance tier and snapshots it into `direction_tier` / `direction_basis`,
shown in the pill's tooltip. Both questions remain answerable:

- **confidence_tier** → *where* the kinase is expressed (cell-type exclusivity).
- **direction_tier** → *whether its activity moves with disease* (concordance).

For **5xFAD**: bulk-MEA significance and snRNA direction gates that previously
gated the pill are now stored as row fields (`bulk_mea_significant`,
`direction_concordant`, `decomp_agrees_bulk`) shown in the accordion detail.
They do not gate the confidence tier.

For **T-cell**: `tcell_concordant` and `tcell_consistency` are displayed as
info-only columns. The single-donor caveat (no reliable p-value) applies to
direction only; the exclusivity pill uses pooled within-cohort detection, not
p-values.

> **Note:** [`concordance.md`](./concordance.md) still documents the older model in
> which the confidence tier *was* the direction-concordance tier. That tier now
> lives in `direction_tier`; the headline `confidence_tier` is the cell-type-exclusivity
> calculation described here.
