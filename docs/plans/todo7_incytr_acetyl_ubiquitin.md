# Plan: Extend Incytr to Acetylation and Ubiquitination

**Date:** 2026-06-18  
**Status:** IMPLEMENTED — code authored in worktree (branch: main); verify after box frees.

---

## 1. How Phosphorylation Flows Through Incytr Today

### 1.1 Where PTM data enters

The driver `alz/incytr_pair/incytr_commandline.R` loads three PTM tracks via env-parameterized filenames:

```
PR_FILE  → pr_yuyu_deconvoluted.csv   (total proteome — the `pr` channel)
PS_FILE  → ps_yuyu_deconvoluted.csv   (pSer/pThr IMAC — the `ps` channel)
PY_FILE  → py_yuyu_deconvoluted.csv   (pTyr — the `py` channel)
```

These are wide CSVs with schema: `gene_symbol | <condition>_<cluster>_<suffix>` where suffix is `pr`, `ps`, or `py`. `ps`/`py` are site-level with a `gene_symbol` key; `pr` is gene-level with a configurable `PR_GENE_COL` key (`"Gene Symbol"` for Song, `"gene_symbol"` for T-cells).

The active channels are declared via `CHANNELS=pr,py,ps` (T-cell donor2 uses `CHANNELS=pr,py` — no IMAC). `pr` is required (drives `prG` receiver gene set). `ps`/`py` are optional.

### 1.2 The `slice_omics` function

```r
slice_omics(df, gene_col, condition, suffix)
```
Anchored-prefix-selects columns `^<condition>_`, strips the prefix, appends `_<suffix>`, and returns one row per `gene_symbol`. This is **PTM-agnostic** — it operates on column names and a suffix label, not on biology.

### 1.3 The `floor_pr` floor

```r
floor_pr <- function(df) { for (cc ...) df[[cc]] <- pmax(df[[cc]], 1); df }
```
Applied **only to `pr_1`/`pr_2`** (driver lines 195–196). This is the sce4 parity constant. `ps`/`py` are not floored. The floor is semantically about deconvolution residuals in the total proteome, not about phospho — new PTM channels (`Ack`, `KGG`) should NOT be floored.

### 1.4 The `multiomics_args` list

```r
multiomics_args <- list(
  pr.data_condition1 = pr_1, pr.data_condition2 = pr_2, pr.correction = 0.001, pr.q = NULL,
  ps.data_condition1 = ps_1, ps.data_condition2 = ps_2, ps.correction = 0.001, ps.q = NULL,
  py.data_condition1 = py_1, py.data_condition2 = py_2, py.correction = 0.001, py.q = NULL
)
```
This is forwarded to `Incytr::Cal_pairwise_grid(multiomics = multiomics_args, ...)` which calls `Integr_multiomics`.

### 1.5 Inside the Incytr package

`Integr_multiomics` (`incytr/R/analysis.R:597`) accepts a named list keyed by `pr | ps | py | Ack | KGG | Rme1`. It calls `integrate_omics_layer()` for each non-NULL layer. That function:
- Pulls `<sender>_<suffix>` and `<receiver>_<suffix>` columns  
- Runs `limma::normalizeBetweenArrays()` across the pair  
- Calls `Cal_foldchange()` to produce `log2FC` and `aFC`  
- Emits 8 columns: `{Ligand,Receptor,EM,Target}_{suffix}_{log2FC,aFC}`  
- Stores into `object@<suffix>_FC`

**The Incytr package already has slots `Ack_FC` and `KGG_FC`** in the class definition (`Incytr_class.R:34–36`). `Pathway_evaluation()` already reads them (`evaluation.R:70–71`), includes them in `multimodel_score`, and `Export_results()` already emits them (`evaluation.R:279–288`). The label `Rme1` is also pre-wired.

**The package-side changes for Ack/KGG are ZERO** — the package already handles them end-to-end. The gap is entirely on the application side: the driver does not load/pass Ack or KGG data.

### 1.6 What is phospho-specific vs PTM-agnostic

| Code location | Phospho-specific? | Notes |
|---|---|---|
| `PR_FILE`/`PS_FILE`/`PY_FILE` env vars | Assumed | Naming convention, not logic |
| `CHANNELS=pr,py,ps` | Assumed | Could be generalized |
| `slice_omics()` function | **PTM-agnostic** | Works on any suffix |
| `floor_pr()` | **pr-specific** | sce4 parity, do not extend |
| `multiomics_args` list | Assumed | Only has 3 keys today |
| `Integr_multiomics` (package) | **PTM-agnostic** | Iterates `.OMICS_LABELS` |
| `Pathway_evaluation` score terms | **PTM-agnostic** | `s4`/`s5` for Ack/KGG already |
| `Export_results` | **PTM-agnostic** | Already handles Ack/KGG slots |
| `prG` receiver gene set | **pr-specific** | `proteomics_gene()` on `pr_*` |
| `proteomics_gene()` | **pr-specific** | aFC on total proteome only |
| `drop_pat` regex | Partially assumed | Drops `_pr_aFC` etc.; Ack/KGG have no raw aFC columns in current output |

### 1.7 The `drop_pat` column filter

```r
drop_pat <- "^(Ligand|Receptor|EM|Target)_(pr|ps|py)_aFC$|^SiK_(R|EM|T)_of_(EM|T|R)(_EI_.*)?$"
```
This drops the `_aFC` raw mirror columns from the output. If Ack/KGG produce `Ligand_Ack_aFC` etc. columns, this pattern **will not drop them** — they will appear in the shard. That is acceptable (same as ps/py aFC columns being present before this filter dropped them). When Ack/KGG are added, update `drop_pat` to also match `_Ack_aFC` and `_KGG_aFC`.

### 1.8 The `num_pat` coercion regex

```r
num_pat <- "^(SigProb|p_value|SiK|log2FC|aFC|PDS|TPDS|PPDS|PhPDS|Ack_score|KGG_score|Rme1_score|multimodel_score|pr_|ps_|py_)"
```
This regex already includes `Ack_score` and `KGG_score` (the evaluation scalars) but does **not** cover `Ack_|KGG_` column prefixes (the per-node fold-change columns like `Ligand_Ack_log2FC`). These would be written as strings since they start with `Ligand_`, not `Ack_`. In practice `Export_results` returns them as numeric already, so this may not matter. Watch for type coercion issues if Ack/KGG FC columns land as character in the shard.

---

## 2. PTM Data Availability by Cohort

### 2.1 Song (AD mouse cohort)

**Source:** `data/datasets/song/primary/phospho/` — contains only:
- `song_IMAC_compositeSites_merged_labeled.xlsx` (pSer/pThr — maps to `ps`)
- `song_IMAC_sitequant_merged_labeled.xlsx`  
- `song_pY_compositeSites_merged_labeled.xlsx` (pTyr — maps to `py`)
- `song_pY_sitequant_merged_labeled.xlsx`

**No AcK (Ack/acetylation) or KGG (ubiquitination) files exist for Song.** The existing derived inputs (`data/derived/incytr_inputs/`) confirm this: `pr_yuyu_deconvoluted.csv`, `ps_yuyu_deconvoluted.csv`, `py_yuyu_deconvoluted.csv` only.

**Verdict for Song: ❌ Ack — no data. ❌ KGG — no data.**  
Song Incytr results cannot include Ack/KGG tracks. Existing phospho results are complete.

### 2.2 5xFAD cohort

**Source:** `data/datasets/5xFAD/primary/{cortex,hippocampus}/proteomics/`

Cortex:
- `ack/31Mar2026_MaleCtx5xFAD_AcK_PTMSiteReport.tsv` (77 MB)
- `kgg/20260501_102617_260203_LD_cortex_KGG_Report.tsv` (101 MB)
- Plus existing `imac/`, `py/`, `total/`

Hippocampus:
- `ack/20260501_092018_011926_Lucie_Hippocampus_male_Mo6-12_5xFAD_Report.tsv` (23 MB — note: Mo6-12 only, so 3mo may be absent)
- `kgg/20260501_093053_…_KGG_EnsembleDB_LFS-v1_Report.tsv` (19 MB)
- Plus existing `imac/`, `py/`, `total/`

**Status per `ingest.py`:** The `5xFAD/ingest.py` `REPORT_SPECS` already registers these files with `source_priority = "available_not_kinase_mea_v1"` and `analysis_scope = "provenance_only"`. They are on disk and parseable via the existing report-reading infrastructure. The `KINASE_TRACKS` dict covers only `st`/`py` today; AcK/KGG are not in the MEA pipeline.

**RAISE — 5xFAD cohort on hold:** Per `project_5xfad_cohort_on_hold` memory entry, the 5xFAD Incytr pipeline itself (deconvolution → Incytr driver) has not been run yet. The `run_pair_mode_5xfad.sh` exists but no Incytr pair-mode results exist in `outputs/reports/`. 5xFAD Incytr (including Ack/KGG) is blocked on the cohort-level gate being lifted.

**Verdict for 5xFAD: ✅ Ack data on disk (cortex: full; hippocampus: 6–12 mo only). ✅ KGG data on disk (both tissues). But Incytr for 5xFAD is on hold pending cohort gate.**

### 2.3 T-cell cohort

**Source:** `data/datasets/tcells/{donor1,donor2}/proteomics/`

- `donor1/`: `TotalProteome_ForPerseus.txt`, `pY_ForPerseus.txt`, `IMAC_IMACSiteReport.tsv` — no AcK/KGG files
- `donor2/`: `TotalProteome_ForPerseus.txt`, `pY_ForPerseus.txt` — no AcK/KGG files

**No AcK or KGG files exist for T-cells.** Existing `data/derived/tcells_incytr_inputs/{donor1,donor2}/` confirms: `pr_deconvoluted.csv`, `ps_deconvoluted.csv` (donor1 only), `py_deconvoluted.csv` only.

**Verdict for T-cells: ❌ Ack — no data. ❌ KGG — no data.**

### 2.4 Mukesh cohort

Not mentioned in the task scope. Not examined.

### Summary table

| Cohort | Ack | KGG | Notes |
|---|---|---|---|
| Song (AD mouse) | ❌ none | ❌ none | Only pSer/pThr + pTyr on disk |
| 5xFAD cortex | ✅ on disk | ✅ on disk | Cohort on hold; 3mo may be missing from hippo AcK |
| 5xFAD hippocampus | ⚠️ 6–12 mo only | ✅ on disk | Mo3 absent from AcK report |
| T-cell donor1/donor2 | ❌ none | ❌ none | ForPerseus bundles have no AcK/KGG |

---

## 3. Minimal Extension Design

### 3.1 Key insight

The Incytr **package** already handles Ack and KGG end-to-end (slots, integration, scoring, export). The extension is **entirely a driver change**: load the new PTM files, slice them like ps/py, and inject them into `multiomics_args`. No new code paths, no parallel pipelines.

### 3.2 New env parameters in `incytr_commandline.R`

Extend the existing channel/file env-parameter block with two new channels:

```r
# Add Ack and KGG to the recognized channel set
stopifnot(all(CHANNELS %in% c("pr", "py", "ps", "Ack", "KGG")))
# (keep stopifnot("pr" %in% CHANNELS) unchanged)

ACK_FILE    <- Sys.getenv("ACK_FILE",    unset = "")
KGG_FILE    <- Sys.getenv("KGG_FILE",    unset = "")
ACK_GENE_COL <- Sys.getenv("ACK_GENE_COL", unset = "gene_symbol")
KGG_GENE_COL <- Sys.getenv("KGG_GENE_COL", unset = "gene_symbol")
```

Load conditionally (same pattern as `py`/`ps`):

```r
ack <- if ("Ack" %in% CHANNELS && nzchar(ACK_FILE)) read_csv(file.path(INPUTS_DIR, ACK_FILE)) else NULL
kgg <- if ("KGG" %in% CHANNELS && nzchar(KGG_FILE)) read_csv(file.path(INPUTS_DIR, KGG_FILE)) else NULL
```

Slice (same `slice_omics` function, no changes):

```r
ack_1 <- if (!is.null(ack)) slice_omics(ack, ACK_GENE_COL, condition1, "Ack") else NULL
ack_2 <- if (!is.null(ack)) slice_omics(ack, ACK_GENE_COL, condition2, "Ack") else NULL
kgg_1 <- if (!is.null(kgg)) slice_omics(kgg, KGG_GENE_COL, condition1, "KGG") else NULL
kgg_2 <- if (!is.null(kgg)) slice_omics(kgg, KGG_GENE_COL, condition2, "KGG") else NULL
```

Do **NOT** floor Ack/KGG. The floor is a `pr`-specific sce4 parity constant, not a general PTM rule.

Inject into `multiomics_args`:

```r
multiomics_args <- list(
  pr.data_condition1 = pr_1,  pr.data_condition2 = pr_2,  pr.correction = 0.001, pr.q = NULL,
  ps.data_condition1 = ps_1,  ps.data_condition2 = ps_2,  ps.correction = 0.001, ps.q = NULL,
  py.data_condition1 = py_1,  py.data_condition2 = py_2,  py.correction = 0.001, py.q = NULL,
  Ack.data_condition1 = ack_1, Ack.data_condition2 = ack_2, Ack.correction = 0.001, Ack.q = NULL,
  KGG.data_condition1 = kgg_1, KGG.data_condition2 = kgg_2, KGG.correction = 0.001, KGG.q = NULL
)
```

When `ack_1`/`ack_2` are NULL, `Integr_multiomics` already handles NULL gracefully (skips the layer with a message). This is the existing pattern for `ps`/`py` on donor2 T-cells.

### 3.3 Update `drop_pat`

```r
drop_pat <- paste0(
  "^(Ligand|Receptor|EM|Target)_(pr|ps|py|Ack|KGG)_aFC$",
  "|^SiK_(R|EM|T)_of_(EM|T|R)(_EI_.*)?$"
)
```

The `Ack_score`/`KGG_score` evaluation columns should be **kept** (they are in `num_pat` already).

### 3.4 Deconvolution preprocessing for 5xFAD Ack/KGG — IMPLEMENTED

**`alz/cohorts/fivexfad/ingest.py` — `run_export_bulk()` extension:**

A new helper `_read_ptm_bulk_linear(tissue, assay, manifest, group_map)` reads the
AcK/KGG Spectronaut TSVs directly (not via `KINASE_TRACKS` — those are only for
kinase MEA, which AcK/KGG do not feed). It reuses `_quantity_columns`,
`_median_center_log2`, `_linear_group_bulk`, `_site_id`, `_extract_gene`, and
`_motif` from the same file. No residue-type filter is applied — each report is
already single-PTM-type from Spectronaut.

`run_export_bulk()` now loops over `("ack", "ack_bulk_linear.csv")` and
`("kgg", "kgg_bulk_linear.csv")` after the existing `pr`/`ps`/`py` block,
calling `_read_ptm_bulk_linear` per tissue and writing the bulk CSVs. Self-gating:
returns None (with a printed note) when the report is absent. Existing pr/ps/py
outputs are untouched.

**`alz/ingest/fivexfad_decompose.py` — channel extension:**

`KEY_COLS`, `BULK_FILE`, and `OUT_FILE` dicts now include `"ack"` and `"kgg"` entries
with site-level keys `["site_id", "gene_symbol", "motif"]` — the same as `ps`/`py`.
The `_run()` inner function is PTM-agnostic (operates on bulk file + key cols),
so no new logic is needed. The channel loop is extended to
`("pr", "ps", "py", "ack", "kgg")`. Self-gating via the existing
`if not os.path.exists(bulk_path): skip` path.

**Note on 5xFAD AcK hippocampus (3mo):** The AcK hippocampus report is labeled
`Mo6-12` — 3-month samples will be absent from `ack_bulk_linear.csv` for
hippocampus. The deconvolution's `_deconvolve` function uses only conditions
present in both bulk and scRNA (`shared = set(bulk_conditions) & set(shares)`),
so 3mo groups that are in scRNA but not in the AcK bulk are silently skipped
(listed in `bulk_conditions_skipped_no_scrna`). The resulting `ack_deconvoluted.csv`
will have no `TG_3mo_*` or `WT_3mo_*` columns for hippocampus AcK.
`slice_omics` produces an all-NA slice for those conditions; `Integr_multiomics`
treats NA FC as 0 in `score_omics_layer` — acceptable.

---

## 4. Output Isolation — Not Overwriting Phospho Results — IMPLEMENTED

### 4.1 Naming scheme

The runner uses **distinct output subdirectories** to prevent the PTM-extended
results from ever overwriting the phospho-only results:

```
# Phospho-only (canonical; never touched by --ptm):
outputs/reports/incytr_pair_mode_5xfad/cortex/wide/
outputs/reports/incytr_pair_mode_5xfad/hippocampus/wide/

# PTM-extended (Ack + KGG additive; written only by --ptm):
outputs/reports/incytr_pair_mode_5xfad/cortex/wide_ptm/
outputs/reports/incytr_pair_mode_5xfad/hippocampus/wide_ptm/
```

Per-contrast filename schema is unchanged in both:
```
<condition1>_<condition2>_incytr_output.parquet
```

The `--smoke` flag respects the same separation:
`wide_smoke/` (default) vs `wide_ptm_smoke/` (--ptm --smoke).

**Anti-shim note:** phospho-only (`wide/`) and phospho+PTM (`wide_ptm/`) are two
coexisting data products the user explicitly wants kept separate. They are NOT an
old/new toggle — the phospho result is the canonical AD-parity product; the PTM
result is an additive 5xFAD-specific extension. Both are permanent. This is a
legitimate anti-shim exemption (§ "genuine external compatibility" — the two
products have different scientific interpretations and different consumers).

Song and T-cell Incytr output directories (`incytr_pair_mode/wide/` and
`incytr_pair_mode_tcells/`) are completely unaffected.

### 4.2 Shell runner — IMPLEMENTED

`alz/incytr_pair/run_pair_mode_5xfad.sh` now accepts two flags:

- **(no flag):** `CHANNELS=pr,ps,py`, output `…/<tissue>/wide/`.  
  **Behavior is byte-identical to pre-extension.** Phospho preflight does NOT
  require ack/kgg files, so the running #6 job is unaffected.

- **`--ptm`:** `CHANNELS=pr,ps,py,Ack,KGG`, sets `ACK_FILE/KGG_FILE/ACK_GENE_COL/KGG_GENE_COL`
  env vars, output `…/<tissue>/wide_ptm/`. Preflight checks ack/kgg deconvoluted
  CSVs only in this mode.

Modes can be combined: `--ptm --smoke cortex` runs a 2-pair, nboot=2 smoke
into `wide_ptm_smoke/`.

### 4.3 Derived input directories

No changes to Song or T-cell derived input directories:
- `data/derived/incytr_inputs/` — Song only (pr/ps/py, unchanged)
- `data/derived/tcells_incytr_inputs/` — T-cells only (unchanged)
- `data/derived/5xfad_incytr_inputs/cortex/` — gains `ack_bulk_linear.csv`,
  `kgg_bulk_linear.csv`, `ack_deconvoluted.csv`, `kgg_deconvoluted.csv`
- `data/derived/5xfad_incytr_inputs/hippocampus/` — same (hippocampus AcK
  bulk/deconvoluted will have no 3mo columns — documented in §3.4)

### 4.4 No viewer changes for phospho

The viewer reads `INCYTR_PAIR_MODE_OUTPUTS_DIR` (currently `outputs/reports/incytr_pair_mode`) for Song pathways. The new 5xFAD outputs live in a different directory tree and are not yet wired to the viewer. When viewer integration is needed, the 5xFAD cohort slice in `build_unified_viewer.py` would be extended with a parallel `_write_incytr_pair_pathways_5xfad()` function reading from the tissue-specific output dirs.

---

## 5. Viewer Surface

The existing viewer is Song-centric for Incytr pathways. When 5xFAD Incytr results are ready, the extension follows the existing Song cohort pattern in `alz/viewer/cohorts/song.py`:

- Add a 5xFAD Incytr pathways function that reads `outputs/reports/incytr_pair_mode_5xfad/{tissue}/wide/*.parquet`
- Shard by (tissue, sender, receiver) — tissue is an additional dimension vs. Song
- Register under a new PAYLOAD key, e.g. `"incytr_pathways_5xfad"`, to avoid collisions with `"incytr_pathways"` (Song)
- The existing `_INCYTR_SCORE_COLS` / `_INCYTR_FC_COLS` sets already include `Ack_score`, `KGG_score` — verify once output schema is confirmed

**No viewer changes are required before generating the 5xFAD results.** Viewer integration is a follow-on step.

---

## 6. Verification: Phospho Path Byte-Identity

### 6.1 The existing sce4 gate

`pixi run verify-incytr-sce4` runs `alz/incytr_pair/verify_incytr_sce4.sh` → `verify_sce4_parity.py`. This gate:
- Runs the driver with the frozen 46-cluster Song inputs, NBOOT=0, nboot=0, Song channels only (`CHANNELS=pr,ps,py`)
- Checks Microglia→Cholinergic-Neurons recall ≥ 599/600
- Checks max `|Δ sclog2FC|` = 0 on R/T

**The gate is unaffected** by adding Ack/KGG env params because:
1. `verify_incytr_sce4.sh` sets `CHANNELS=pr,ps,py` explicitly — no Ack/KGG channels active
2. `ACK_FILE`/`KGG_FILE` default to `""` → `ack=NULL` / `kgg=NULL` → `multiomics_args` Ack/KGG keys map to NULL → `Integr_multiomics` skips them silently
3. No code path in the driver is modified for the phospho run, only extended for the Ack/KGG run

**Parity guarantee:** When `Ack` and `KGG` are not in `CHANNELS`, the driver behavior is byte-identical to the pre-extension code. The new env vars (`ACK_FILE`, `KGG_FILE`, etc.) are read-and-ignored (empty strings → NULL data). The `multiomics_args` list gets two extra NULL entries — `Integr_multiomics` iterates over named keys and skips NULL layers. Output is unchanged.

### 6.2 Post-implementation verification checklist

Before shipping:

1. `pixi run verify-incytr-sce4` — must still pass (599/600 Micro→Cholin, max |Δ|=0)
2. Smoke run with `CHANNELS=pr,ps,py` on Song (no-op extension check): diff output against the existing wide parquet — should be byte-identical
3. Smoke run on 5xFAD cortex with `CHANNELS=pr,ps,py,Ack,KGG` (after deconvolution is built): spot-check that `Ack_score`, `KGG_score`, `Ligand_Ack_log2FC`, `Ligand_KGG_log2FC` columns are present and numeric (not all-NA)

---

## 7. Touch Points Summary — IMPLEMENTED

### Driver (`alz/incytr_pair/incytr_commandline.R`)

| Location | Change |
|---|---|
| `stopifnot(all(CHANNELS %in% c("pr", "py", "ps")))` | Extended to `c("pr", "py", "ps", "Ack", "KGG")` |
| Env-param block after `PS_GENE_COL` | Added `ACK_FILE`, `KGG_FILE`, `ACK_GENE_COL`, `KGG_GENE_COL` (all default to `""` / `"gene_symbol"`) |
| After `ps <- ...` | Added `ack <- ...` and `kgg <- ...` with `nzchar(ACK_FILE)` / `nzchar(KGG_FILE)` guard |
| After `ps_1/ps_2 <- slice_omics(...)` | Added `ack_1/ack_2` and `kgg_1/kgg_2` via same `slice_omics` |
| `multiomics_args` list | Added `Ack.data_condition1/2`, `Ack.correction`, `Ack.q`, and KGG equivalents |
| `drop_pat` | Added `Ack` and `KGG` to the per-node aFC drop pattern |

### 5xFAD ingest (`alz/cohorts/fivexfad/ingest.py`)

| Location | Change |
|---|---|
| New `_read_ptm_bulk_linear(tissue, assay, manifest, group_map)` | Reads AcK/KGG Spectronaut TSVs, applies median-center log2 + linear-group-bulk, returns DataFrame or None |
| `run_export_bulk()` — end of per-tissue loop | Added loop over `("ack", "ack_bulk_linear.csv")`, `("kgg", "kgg_bulk_linear.csv")` calling `_read_ptm_bulk_linear` |

### 5xFAD decompose (`alz/ingest/fivexfad_decompose.py`)

| Location | Change |
|---|---|
| `KEY_COLS`, `BULK_FILE`, `OUT_FILE` dicts | Added `"ack"` and `"kgg"` entries (site-keyed) |
| `_decompose_tissue()` channel loop | Extended `("pr", "ps", "py")` to `("pr", "ps", "py", "ack", "kgg")` |

### 5xFAD runner (`alz/incytr_pair/run_pair_mode_5xfad.sh`)

| Location | Change |
|---|---|
| New `--ptm` flag | Sets `CHANNELS_ENV=pr,ps,py,Ack,KGG`, `WIDE_SUBDIR=wide_ptm`, `SMOKE_SUBDIR=wide_ptm_smoke` |
| `run_one()` | Assembles `ptm_env` array with `ACK_FILE/KGG_FILE/ACK_GENE_COL/KGG_GENE_COL`; empty in phospho-only mode |
| `preflight()` | Checks `ack_deconvoluted.csv` / `kgg_deconvoluted.csv` only when `$PTM == "yes"` |
| Default mode | Unchanged: `CHANNELS=pr,ps,py`, output `wide/`, no ack/kgg in preflight |

### Incytr package (`~/Projects/work/incytr/R/`)

**No changes.** The package already supports Ack and KGG as first-class omics layers.

### Song and T-cell runners

**No changes.** The extension is additive via env params; existing runners do not set `ACK_FILE`/`KGG_FILE`, so Ack/KGG channels stay inactive.

---

## VERIFY LATER (needs the box — DO NOT RUN NOW)

The box is running a 13–19h Incytr job consuming most of RAM. Run these after it finishes:

**(a) Phospho byte-identity gate (highest priority):**
```bash
pixi run verify-incytr-sce4
```
Must still pass: 599/600 Micro→Cholin recall, max |Δ sclog2FC| = 0 on R/T.
Expected: PASS — the new code is additive; when `CHANNELS=pr,ps,py` and
`ACK_FILE/KGG_FILE` are unset (empty), `ack=NULL`, `kgg=NULL`, and
`Integr_multiomics` skips those slots silently. No existing code path is modified.

**(b) Build 5xFAD AcK/KGG deconvoluted CSVs:**
```bash
# Step 1: produce ack_bulk_linear.csv / kgg_bulk_linear.csv
pixi run python -m alz.cohorts.fivexfad.ingest --export-bulk

# Step 2: deconvolve into ack_deconvoluted.csv / kgg_deconvoluted.csv
pixi run python alz/ingest/fivexfad_decompose.py

# Sanity: check mass-identity max |rel err| printed for each channel/tissue.
# ack/kgg should be ≲ 0.01 (same expectation as ps/py).
```

**(c) PTM smoke run (5xFAD cortex, 1 pair, nboot=2):**
```bash
bash alz/incytr_pair/run_pair_mode_5xfad.sh --ptm --smoke cortex
```
Confirm:
- Output lands in `outputs/reports/incytr_pair_mode_5xfad/cortex/wide_ptm_smoke/`
- `wide/` is NOT modified
- Output parquet has columns `Ack_score`, `KGG_score`, `Ligand_Ack_log2FC`,
  `Ligand_KGG_log2FC` that are numeric (not all-NA)
- `Ligand_Ack_aFC` and `Ligand_KGG_aFC` are ABSENT (dropped by `drop_pat`)

---

## 8. Blocked Items / Raises

1. **Song Ack/KGG — data does not exist.** Cannot extend Song Incytr to include Ack/KGG. If data arrives in future, the deconvolution path is analogous to Song `ps`/`py` in `export_decomposition_for_pair.py`.

2. **T-cell Ack/KGG — data does not exist.** Cannot extend T-cell Incytr.

3. **5xFAD cohort gate.** Per `project_5xfad_cohort_on_hold` memory: 5xFAD Incytr is on hold. All 5xFAD Ack/KGG Incytr work is blocked until that hold is lifted. The driver extension (touch point #1 above) can be implemented now and is safe since it changes nothing for Song/T-cell runs.

4. **5xFAD hippocampus AcK 3mo samples.** The AcK hippocampus report (`Mo6-12`) may lack 3-month samples. The 3mo TG vs WT contrast for hippocampus AcK will produce NA FC values for every gene. This is acceptable (treated as 0 by the scorer) but should be flagged in the deconvolution output manifest.

5. **5xFAD deconvolution not yet run for Ack/KGG.** The `5xfad_incytr_inputs/cortex/` and `hippocampus/` dirs have `pr`, `ps`, `py` only. Ack/KGG deconvolution must be built before the runner can execute.

6. **5xFAD Seurat object condition encoding.** The condition names in the 5xFAD `incytr_obj.rds` (e.g. `"TG_3mo"`, `"WT_3mo"`) need to be confirmed before writing the contrast loop in `run_pair_mode_5xfad.sh`. Check `build_5xfad_seurat.R`.
