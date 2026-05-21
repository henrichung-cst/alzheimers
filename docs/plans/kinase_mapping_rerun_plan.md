# Song analysis full rerun — post kinase-mapping update

Audit pass after refreshing the kinase→substrate mapping. Goal: regenerate
every downstream deliverable from the new mapping and confirm consistency
across the bulk track, the per-cluster decomposition track, and Incytr.

## Inputs assumed fresh

- Updated `kinase_to_gene_mapping.csv` (in `data/datasets/song/analysis_cache/`)
- Kinase-library substrate annotations unchanged (only the gene-symbol bridge moved)

If the kinase-library version also changed, re-pin in `pixi.toml` first and
run `pixi install` before Stage 0.

## Stage 0 — sanity check on the mapping change

1. `git diff -- data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv`
   (or wherever the mapping lives) — record added/removed/renamed kinases.
2. Grep for hard-coded kinase names in `config.py`, `kinase_enrich.py`,
   `kinase_attribute.py`, `build_unified_viewer.py`. Any name no longer in
   the mapping needs a follow-up.
3. Note the count delta — used later as a smoke check on MEA output row counts.

## Stage 1 — bulk pipeline (males-only primary)

Runs from `outputs/reports/kinase_attribution/` upward. Stoichiometry inputs
(Stage 1 normalize) are mapping-agnostic; the rerun starts at Stage 2.

```bash
# Stage 1 — only if normalized matrices are stale or missing
pixi run normalize

# Stage 2 — OLS + MEA on stoichiometry β (consumes new mapping)
pixi run enrich

# Stage 3 — unified cell-type attribution (WMB + SEA-AD + Song snRNA LFC)
pixi run attribute

# Optional: mechanism classification
pixi run python alz/kinase_mechanism.py
```

Expected outputs to diff against the prior run:
- `mea_stoichiometry.csv` — NES/FDR per (kinase, contrast); row count = |kinases ∩ mapping|
- `unified_attribution.csv` — kinase × WMB-class evidence rows
- `attribution_summary.json` — confidence-tier counts

Sensitivity rerun:
```bash
KEDRO_ENV=full_cohort pixi run enrich
KEDRO_ENV=full_cohort pixi run attribute
```

Final hypothesis tables:
```bash
pixi run python alz/attribution_recovery.py --run
```

## Stage 2 — bulk decomposition (Levy-19 spine)

Forward projection `P_c = f_c × bulk`, both proteomics and phospho tracks
(st = IMAC pS/pT, py = pY). Mapping-independent — but rerun anyway since
downstream MEA depends on it.

```bash
# Stage 5 — per-cluster proportions f_c
pixi run python alz/snrna_proportions.py --spine levy19

# Stage 6 — decompose bulk phospho (st + py) and proteome onto 19 clusters
pixi run python alz/decomposition/build_celltype_decomposition.py \
    --spine levy19 --track both
```

Outputs:
- `outputs/reports/decomposition/levy19/protein_per_cluster.parquet`
- `outputs/reports/decomposition/levy19/phospho_per_cluster.parquet` (st)
- `outputs/reports/decomposition/levy19/phospho_per_cluster_pY.parquet` (py)

Verify before moving on:
```bash
pixi run python alz/decomposition/verify_decomposition.py \
    --spine levy19 --checks mass coverage
```
Must pass: per-cell-rate mass identity + all 19 clusters present.

## Stage 3 — per-cluster MEA and bulk-vs-cluster comparison

Per-cluster MEA consumes the new mapping; this is the second mapping-sensitive
step after Stage 1.

```bash
pixi run python -m alz.decomposition.enrich_celltype --spine levy19 --track st
pixi run python -m alz.decomposition.enrich_celltype --spine levy19 --track py
```

Output: `outputs/reports/decomposition/levy19/mea_per_cluster.parquet`
(columns: `kinase, cluster, contrast, NES, FDR, n_substrates`).

Comparison check:
```bash
pixi run python alz/decomposition/verify_decomposition.py \
    --spine levy19 --checks mea
```
Per-contrast Spearman ρ(per-cluster f_c-weighted NES vs bulk NES) ≥ 0.7.
A drop below this threshold after the mapping change is the audit's primary
red flag — investigate which kinases moved before continuing.

Manual diff worth running:
- Join `mea_stoichiometry.csv` (bulk) ↔ `mea_per_cluster.parquet`
  (cluster-aggregated) on `(kinase, contrast)`, scatter NES, flag |Δ| > 1.

## Stage 4 — Incytr factorial across all omics layers

Per-cluster Incytr scoring on the Levy-19 spine. Reads decomposition outputs
(Stage 2) for the pr/ps/py layers; kinase activity layer reads per-cluster MEA
from Stage 3.

```bash
pixi run python alz/integration/export_factorial_inputs.py
pixi run incytr-factorial
```

Outputs (per-cluster parquets):
- `outputs/reports/incytr_factorial/per_cluster/pr/<cluster>.parquet`
- `outputs/reports/incytr_factorial/per_cluster/ps/<cluster>.parquet`
- `outputs/reports/incytr_factorial/per_cluster/py/<cluster>.parquet`
- `outputs/reports/incytr_factorial/pair_metadata.parquet` (19² = 361 pairs)

Verify:
```bash
pixi run python alz/decomposition/verify_decomposition.py \
    --spine levy19 --checks incytr
```

## Stage 5 — viewer rebuild + spot-check

```bash
pixi run python alz/build_unified_viewer.py
```

Open `outputs/reports/unified_viewer/index.html` (hard refresh) and spot-check:
- Kinase tab: any kinase added/removed by the new mapping appears/disappears
- Pathway tab: backbone edge counts plausible
- Incytr tab: per-cluster panel populated for all 19 clusters

## One-shot runner

The existing smoke runner covers Stages 2–4:
```bash
bash alz/runners/main/run_pivot_smoke.sh --skip-normalize
```

Pair with the bulk pipeline at the front:
```bash
pixi run dual                              # Stage 1 (males + full-cohort)
bash alz/runners/main/run_pivot_smoke.sh   # Stages 2–4
pixi run python alz/build_unified_viewer.py
```

## Audit deliverables

Capture in `outputs/reports/audits/kinase_mapping_rerun_<date>/`:
1. Diff of mapping file (kinase adds/removes/renames)
2. Pre/post diff of `mea_stoichiometry.csv` — NES delta histogram
3. `verification.json` from Stage 13 harness (mass, coverage, mea, incytr)
4. Pre/post diff of `kinase_hypothesis_table.csv` top-N
5. Notes on any kinase whose attribution confidence tier changed

## Stage 1 audit results (2026-05-13)

Snapshots and diff under `outputs/reports/audits/kinase_mapping_rerun_2026-05-13/`.

**MEA stoichiometry: no change.** Males-only pre and post `mea_stoichiometry.csv`
are byte-identical (md5 `94fd64f...`). All 389 kinases, 2799 rows, 3501
significant; zero NES movement, zero trajectory-label flips. The mapping
refresh did not move kinase-level enrichment, as expected — the kinase-library
substrate sets are stable and the bridge file only renames the gene-symbol
column. Stage 0 grep confirmed no hard-coded kinase names in the four critical
modules.

**Two bugs surfaced during the audit:**

1. **Cohort-mode audit confound.** `audit_diff.py` initially read
   `outputs/reports/kinase_attribution/`, which `run_dual_analysis.sh`
   overwrites with the full-cohort run as the final step. The pre snapshot
   was males-only, so the first audit pass compared
   males-only (pre) vs full-cohort (post). Fixed: audit now reads the
   `*_males_only/` archive explicitly.

2. **Missing WMB expression file — silent failure.**
   `outputs/reports/wmb_expression/wmb_kinase_expression.csv` is absent.
   `kinase_attribute.py:430-438` falls into the `else` branch, sets
   `wmb_specificity = NaN`, then `fillna(0.0)` zeroes the whole column.
   `wmb_specificity` is uniformly 0.0 across 9000 post-run rows
   (max 0.0000 vs pre max 0.726). Downstream consequence:
   `has_wmb_high (wmb_specificity ≥ 0.105)` is never True →
   the `high` confidence tier collapses from 242 → 0. Total attributed
   rows simultaneously grew (7137 → 9000) because the relaxed specificity
   admits more rows at the `moderate`/`low` boundary, so the regression
   doesn't look like a failure on the surface.

**Required follow-ups before treating any Stage 1 deliverable as final:**

- Run `bash alz/runners/supporting/run_wmb_expression.sh` to regenerate
  `wmb_kinase_expression.csv`, then re-run `pixi run dual`.
- Harden `kinase_attribute.py`: when `wmb_top` is None, hard-fail with a
  clear error rather than silently zeroing `wmb_specificity`.
- Add a WMB prerequisite gate to `run_dual_analysis.sh` to mirror the
  check already present in `run_live_pipeline.sh` (per CLAUDE.md Gotchas).
- `attribution_summary.json` confidence-tier counts and `kinase_hypothesis_table.csv`
  cell-type evidence columns from the current post-run are **not trustworthy**
  until WMB is restored.

## Gotchas

- Stage 1 normalize is **not** mapping-sensitive; skip it unless inputs changed
- `analysis_mode=males_only` is the primary; full-cohort is the sensitivity check
- `run_dual_analysis.sh` overwrites `outputs/reports/kinase_attribution/`
  twice — pre/post audits must read the `*_males_only/` archive, not the
  unsuffixed directory (full-cohort wins the final overwrite)
- WMB expression file (`wmb_kinase_expression.csv`) is a hard prerequisite
  for `kinase_attribute.py`; absence silently zeroes `wmb_specificity` and
  collapses the `high` confidence tier — see Stage 1 audit results above
- Decomposition mass identity is per-cell-rate (`Σ_c P_c × N_c/N_total ≈ bulk`),
  not literal sum
- Per-cluster MEA reflects **abundance change** of phosphosites in a cluster,
  not activity (no per-cluster stoichiometry — algebraically collapses)
- Incytr is gated on upstream `Incytr::construct_factorial_paths` /
  `score_factorial_paths` exports; `pixi run incytr-factorial` will refuse to
  start otherwise

## Addendum (2026-05-14): cortex_hpf swap

User requested replacing the whole-brain WMB reference with the Allen "Mouse
Whole Cortex and Hippocampus 10x" dataset to match the cortical scope of the
phosphoproteomics. That standalone dataset is the cortical slice of the same
WMB-10Xv3 cells under a different release name (same Allen taxonomy), so
onboarding it as a parallel dataset would mean a 5 GB HDF5 download, new
schema, new crosswalk, and a rewrite of `wmb_expression.py` for no new
biology.

Implementation taken instead: restrict the WMB streaming step to four
cortical regions — `Isocortex-1`, `Isocortex-2`, `HPF`, `CTXsp` — via a new
`WMB_REGION_SCOPE` env var (default `cortex_hpf`; set `whole_brain` to
revert). Output filename is unchanged; a sidecar `.scope.json` records
which scope produced the matrix. The Levy-19 ↔ WMB-class crosswalk and all
downstream joins are unaffected.

Action required: back up the existing whole-brain
`outputs/reports/wmb_expression/wmb_kinase_expression.csv` if you want to
compare, then re-run `bash alz/runners/supporting/run_wmb_expression.sh`.
Expect the `high` confidence tier counts in the next audit to shift again
relative to the current snapshot — cortex_hpf specificity will be sharper
on cortical lineages and softer on glia that have whole-brain abundance
elsewhere.

## Reversal (2026-05-14, same day): cortex_hpf was the wrong knob

The cortex_hpf swap implemented above was based on a misread of what the
specificity score is doing. The score is a ratio:

    specificity(kinase, class) = mean_expr(class) / Σ_class mean_expr(class)

Its denominator is the brain-wide reference. Restricting the WMB cell pool
to cortex+HPF shrinks both numerator and denominator simultaneously, so it
does **not** make the score "more cortical" — it just degrades the estimate
for any class whose cells mostly live outside the cortical mask. The
post-swap audit showed exactly this:

- `09 CNU-LGE GABA` lost 24× of its cells (108k → 4.6k); a Levy-19 cluster
  (Striatal-MSN) legitimately targets that class, so we cannot drop it
- the largest specificity deltas (Dusp21 0.66→0.00, Ptpn7 0.87→0.36, etc.)
  were sampling-variance artifacts in low-n classes, not biology
- only ~144 of 4,923 (kinase × class) pairs crossed the `SPECIFICITY_HIGH`
  threshold; the rest of the matrix moved by abs-mean Δ=0.012

Correct framing the user supplied: the Levy-19 spine is itself a cortex/HPF
taxonomy, so simply asking *"how enriched is this kinase in cluster X vs
the rest of the brain?"* with a whole-brain denominator already answers the
cortical-attribution question. No region restriction needed.

Action taken: changed `WMB_REGION_SCOPE` default back to `whole_brain` and
restored `wmb_kinase_expression.csv` from the `_whole_brain.csv` backup;
stamped a matching scope sidecar. The `cortex_hpf` machinery (config
selector, scope-aware cache check, sidecar manifest) is kept as a
sensitivity-check toggle — opt-in via `WMB_REGION_SCOPE=cortex_hpf`. A
parallel cortex_hpf copy is preserved as `wmb_kinase_expression_cortex_hpf.csv`
for diff audits.

To regenerate the whole-brain proteome expression matrix (the cortex_hpf
run overwrote it):

    WMB_REGION_SCOPE=whole_brain bash alz/runners/supporting/run_wmb_expression.sh

The scope-aware cache check will force `--proteome` to recompute because
its sidecar still says `cortex_hpf`. The kinase matrix is already restored
and will be skipped (sidecar matches).
