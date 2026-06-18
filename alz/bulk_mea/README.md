# `alz/bulk_mea/` — Mode 1: Stoichiometry-corrected bulk kinase attribution

This package is the mouse **bulk MEA** chain: it turns the Song cohort's raw TMT
total-proteome and phospho workbooks into the project's primary deliverable,
`kinase_hypothesis_table.csv` — a ranked, cell-type-attributed table of kinase
activity changes across Alzheimer's genotypes and timepoints.

The same code also serves the NBB human cohort indirectly: `alz/cohorts/mukesh/
mea.py` reshapes human data into Song-shaped artifacts and reuses the
Stage-2 OLS/MEA helpers exported from `enrich.py`.

---

## The core idea: stoichiometry, not raw phospho

A rise in a phosphosite's signal can mean two things: the kinase got more active,
or the protein simply became more abundant. To separate them we work in
**stoichiometry space**:

```
stoichiometry = log2(phospho_intensity) − log2(total_protein_intensity)
```

A change in stoichiometry is a change in *occupancy* (fraction of the protein that
is phosphorylated) — i.e. activity — with abundance divided out. Kinase enrichment
(MEA) runs on the stoichiometry β values, so the hits reflect activity. Stage 4
(`mechanism.py`) re-runs the same enrichment on the *raw* phospho LFC and labels
each hit `activity_driven` / `abundance_driven` / `both`, which is the explicit
check that the stoichiometry correction is doing real work.

---

## Stage order

```
normalize → enrich → attribute → mechanism → recover
                                             (summary = read-only viewer)
```

| Stage | Script | Role | Key output |
|------:|--------|------|------------|
| 1 | `normalize.py` | IRS cross-plex normalization (**all 72 samples**) + stoichiometry | `stoichiometry_matrix.csv`, `raw_phospho_normalized.csv` |
| 2 | `enrich.py` | Sample filter → factorial OLS per site (9 contrasts) → MEA on stoichiometry β | `mea_stoichiometry.csv`, `site_level_ols.csv` |
| 3 | `attribute.py` | Cross-join sig kinases × 31-cluster spine; layer SEA-AD + WMB + Song evidence; confidence tiers | `unified_attribution{,_full}.csv` |
| 4 | `mechanism.py` | Raw-phospho MEA + abundance/activity/both classification; **merges `mechanism_annotation` back into `unified_attribution.csv` → must run after `attribute`** | `mechanism_annotation.csv`, `mea_raw_phospho{,_pY}.csv` |
| 5 | `recover.py` | Cross-contrast trajectory classification + final hypothesis tables | **`kinase_hypothesis_table.csv`**, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv` |
| ⸻ | `summary.py` | Read-only: prints cached results across all stages (runs no analysis) | — |

Run via pixi tasks (`pixi run normalize / enrich / attribute / mechanism /
recover`), or the bundled `pixi run live` to chain
`ingest → normalize → enrich → attribute → mechanism → recover`.

---

## How each stage works

### Stage 1 — `normalize.py`

1. **IRS normalization** (Internal Reference Scaling) corrects for cross-plex
   batch effects in the TMT data. Each plex carries a reference channel (`126`,
   `Ref_Pool`); intensities are scaled so the reference is comparable across
   plexes. If fewer than 4 reference channels are present it falls back to
   per-plex median centering. Applied independently to the total proteome and to
   the phospho sitequant.
2. **Stoichiometry** is computed by matching each phosphosite to its parent
   protein on gene symbol (vectorized), then `log2(phospho) − log2(protein)` per
   sample, NaN where either side is missing/non-positive.
3. **PCA QC** before/after normalization, plotted by plex / genotype / sex /
   timepoint, plus per-gene spot-checks (Mapt, Gsk3b, Akt1, Mapk1, Camk2a).

Stage 1 **always uses all 72 samples** — no cohort filtering here. Outputs are
*track-suffixed* (see "Two phospho tracks" below).

### Stage 2 — `enrich.py`

1. **Sample filter** — drops outliers (`sample_exclusions.csv`) and, in
   `males_only` mode, females.
2. **Factorial OLS** — builds a design matrix with App / Tau / Int genotype
   coding × timepoint, plus their interactions (a `female` column is added in
   `full_cohort` mode). `_run_ols_all_sites` fits every site at once. The
   statistical care point: complete-data sites share the global `(XᵀX)⁻¹`, but
   **partial-data sites get their own `(XᵢᵀXᵢ)⁻¹`** — using the global covariance
   on a partial site understates the contrast SE and inflates the t-stat.
3. **Contrasts** — the 9 contrasts (App/Tau/ApTt × 2mo/4mo/6mo) are computed by
   `_contrast_stats`, a single helper applied to both the stoichiometry and the
   raw-phospho fits (LFC, two-sided p, BH-FDR).
4. **MEA** — `_run_mea` median-centers and winsorizes the per-contrast site LFCs
   (1st/99th percentile), then runs GSEA-prerank kinase enrichment via the
   `kinase_library` package, emitting NES + FDR per kinase per contrast.

### Stage 3 — `attribute.py`

Cross-joins the significant kinases against the **Levy-t5 31-cluster spine**
(`config.CLUSTER_SPINE`) and layers in three independent evidence sources via
single-hop crosswalks:

- **SEA-AD** human per-supertype effect sizes, pathway-matched (App→early CPS,
  Tau→late CPS, ApTt→full CPS), joined cluster → supertype.
- **WMB** mouse expression specificity, joined cluster → parent WMB class (1:1
  lineage-level; clusters sharing a parent inherit the same WMB score).
- **Song** within-cohort snRNA concordance + specificity, keyed directly on
  `cluster_name`.

These feed the canonical **confidence tier**. Song within-cohort direction plus
Song location specificity is the route to `high`; decomposition agreement
promotes `high` rows to `very_high`. WMB, SEA-AD, HBCA, and decomposition are
retained as explicit cross-check columns, with `confidence_basis` summarizing
the route. Sorting uses the explicit tier and evidence columns directly, not a
synthetic numeric score. A hard row-count assertion guards against silent drops
in the merge chain.

### Stage 4 — `mechanism.py`

Re-runs Stage-2 MEA on the **raw** (uncorrected) phospho LFCs and compares the
hit-set against the stoichiometry MEA. Each (kinase, contrast) becomes
`activity_driven` (stoich only), `abundance_driven` (raw only), `both`, or
non-significant. It then **merges a `mechanism_annotation` column back into
`unified_attribution.csv`** — which is why it must run *after* `attribute.py`; on
a fresh build with no `unified_attribution.csv` present yet, that merge is
silently skipped and the annotation is lost.

### Stage 5 — `recover.py`

Synthesizes the final hypothesis tables:

- **`kinase_activity_matrix.csv`** — wide NES/FDR per kinase across all 9
  contrasts, plus a `trajectory_label` (progressive / declining / peaked /
  sustained / early / late / single_contrast / mixed / none) computed within the
  kinase's peak condition.
- **`celltype_evidence_table.csv`** — one row per (kinase, cell type) above the
  WMB expression gate, deduped to the contrast with the strongest joint evidence.
- **`kinase_hypothesis_table.csv`** — the primary deliverable: kinase-first
  synthesis joining the activity profile to its top-3 candidate cell types, with
  a binary "has ≥1 high-confidence attribution" flag.

`recover.py` treats `data/derived/caches/kinase_to_gene_mapping.csv` as the
authoritative kinase-abbreviation to gene-symbol map. If
`unified_attribution_full.csv` was built against an older or inconsistent map,
recovery fails fast and instructs you to rerun `attribute.py` before producing
final tables. This prevents stale kinase labels from propagating into the
viewer payload.

---

## Design patterns

**Two phospho tracks.** Everything is parametrized over `config.PHOSPHO_TRACKS`:
`st` (IMAC Ser/Thr, unsuffixed filenames) and `py` (tyrosine-enriched, `_pY`
suffix). Each stage loops both tracks; outputs never collide because of the
suffix. The S/T and Y kinome name-spaces are disjoint, so any per-kinase lookup
downstream is unambiguous.

**Dual CLI / Kedro shape.** Each stage has a side-effect-free compute function
(`step_normalize`, `step_attribution_recovery`, …) that takes in-memory inputs
and returns DataFrames, plus a `main()` CLI shim that loads from / writes to
disk. Kedro nodes call the `step_*` functions; the runners call `main()`.

**Shared helpers live in `config.py`.** Cohort/track plumbing is centralized in
`alz/shared/config.py`, not duplicated per stage:

- `config.load_params()` — read `conf/base/parameters.yml` with `KEDRO_ENV` overlay
- `config.resolve_track(track)` — name → track-config dict
- `config.track_output(filename, track_cfg)` — compose a suffixed output path
- `config.load_sample_mapping()` — load the data-ingest sample mapping

Stage-2-specific OLS/MEA helpers (`_run_ols_all_sites`, `_run_mea`,
`_build_design_matrix`, `_filter_samples`, `CONTRAST_COEFS`, …) are exported from
`enrich.py` and reused by `mechanism.py` and `decomposition_mea/enrich_celltype.py`.

---

## Inputs

- `outputs/reports/data_ingest/sample_mapping.csv` (from `alz/ingest/song.py --mapping`)
- `outputs/reports/data_ingest/sample_exclusions.csv` (from `alz/ingest/song.py --outliers`)
- Raw TMT total-proteome + phospho IMAC/pY workbooks under
  `data/datasets/song/primary/` (read by `normalize.py`)
- `data/derived/caches/kinase_to_gene_mapping.csv` (`config.MAPPING_CACHE_FILE`;
  used by `attribute.py` and `recover.py`)
- WMB per-class expression matrix at
  `outputs/reports/wmb_expression/wmb_kinase_expression.csv` (built by
  `alz/reference/wmb_expression.py`; required for `attribute.py`)
- SEA-AD effect-size h5ads under `data/external/sea_ad/` (required for `attribute.py`)
- Song snRNA specificity/concordance CSVs under
  `outputs/reports/snrna_integration/` (optional; enables the within-cohort tier)

## Outputs

Stages 1–4 write under `outputs/reports/kinase_attribution/`; Stage 5 writes under
`outputs/reports/attribution_recovery/`. See
`docs/foundation/live_pipeline_contract.md` for the per-file schema contract.

## Sample filtering & cohort mode

`analysis_mode` lives in `conf/base/parameters.yml` (default `males_only`, to
avoid hormonal confounds). Set `KEDRO_ENV=full_cohort` to overlay
`conf/full_cohort/parameters.yml` and run the both-sexes sensitivity analysis.
`normalize.py` always uses all 72 samples; cohort filtering applies starting at
`enrich.py`. `config.ANALYSIS_MODE` is itself overlay-aware, so every stage agrees
on the active cohort.

## Verification

There is no unit-test suite. The designated checks are:

```bash
python alz/bulk_mea/summary.py                       # read-only: prints cached results across stages
python alz/bulk_mea/audit_kinase_gene_mapping.py     # fails on generated kinase/gene mapping drift
python alz/decomposition_mea/verify_decomposition.py --all   # decomposition harness
```
