# 5xFAD evidence panel — full Song mirror (transcript + protein + phospho pS/pY)

Wire the Incytr Pathways **Evidence panel** (row-expand 4×4 node×layer matrix,
`EvidencePanel` in `alz/viewer_shared/template/js/widgets/evidence_row.js`) for the two
5xFAD contexts (`fivexfad_cortex`, `fivexfad_hippocampus`) so every layer renders
underlying per-cell-type evidence **mirroring the Song-AD implementation**: per-replicate
dot-bars for protein + phospho, a single pseudobulk bar for transcript, each alongside the
Incytr-stored LFC chip. Today those contexts ship no evidence shards → all cells "n/a".

## What the data supports (revised — all four layers are derivable)

Per-cell-type **per-sample** matrices are derivable for 5xFAD for all four layers, exactly
the way Song gets its per-animal dots. Song's mechanism (confirmed):
`P_c(gene, sample) = bulk(gene, sample) × share_c(condition) × (N_total/N_c)`, where the
per-replicate axis comes from the **per-sample bulk** measurement and the cell-type
`share` is condition-pooled scRNA (Song stratum-pools the share for samples without
matched scRNA — identical design point here).

| Layer | 5xFAD source (already on disk) | Granularity | Render |
|---|---|---|---|
| **Protein** | `kinase_attribution_5xfad/{tissue}_total_proteome_normalized.csv` (per-sample log2) | per-sample (n=3–5/arm) | dot-bar |
| **Phospho pS** | `{tissue}_st_raw_phospho_normalized.csv` (per-sample log2, site-keyed) | per-sample | dot-bar + per-site popover |
| **Phospho pY** | `{tissue}_py_raw_phospho_normalized.csv` | per-sample | dot-bar + per-site popover |
| **Transcript** | `kinase_attribution_5xfad/celltype_mea/fivexfad_snrna_pseudobulk_linear.csv.gz` (per-cell-type per-sample, ≤2/arm) | condition mean (single value/arm) | single bar |

**Why transcript is a single bar:** Song's transcript layer is pseudobulk n=1 per arm
(one pooled scRNA library); mirroring Song, 5xFAD transcript collapses its ≤2 scRNA
samples per condition to a mean. (Deviation available as a follow-up: 5xFAD *could* show
its ≤2 replicate dots since the data is per-sample — flagging, defaulting to the Song
single-bar to mirror.)

## Correctness gate (load-bearing)

The condition-level deconvoluted CSVs (`data/derived/5xfad_incytr_inputs/<tissue>/{pr,ps,py}_deconvoluted.csv`)
are what the Incytr pair-mode consumed to compute the stored node LFCs shown in the chip.
`_linear_group_bulk` builds the condition bulk as `mean_sample(2^log2)`, and the share is
identical, so **`mean over a condition's per-sample P_c == the condition-level deconvoluted
value behind the chip`**. The new builder asserts this reconciliation per (gene/site,
condition) to 1e-6 — guaranteeing the dot-bars and the LFC chip describe the same data.
This is the 5xFAD analog of Song's `verify_pathway_round_trip.py` gate.

**Non-goal:** do NOT modify `fivexfad_decompose.py`, `run_export_bulk`, the condition-level
deconvoluted CSVs, or the Incytr pair-mode. The per-sample deconvolution is a **separate,
read-only-input** artifact built solely for the evidence panel; the chips are unchanged.

## Components

### 1. New builder — `alz/integration/build_omics_trace_fivexfad.py`
The 5xFAD analog of `build_omics_trace.py` (protein + phospho_ps + phospho_py), per tissue:
- **Reuse** `_shares_by_condition` / `_load_counts` / `_load_aggexp` from
  `alz/ingest/fivexfad_decompose.py` (import, don't duplicate) — the same share + counts
  that produced the condition-level values.
- **Reuse** `_sample_group_map` from `alz/cohorts/fivexfad/ingest.py` for the
  `biological_sample_id → <geno>_<age>mo` condition map.
- Read the per-sample **log2** matrix on disk; `lin = 2.0 ** value` (matching
  `_linear_group_bulk`); keep only columns ∈ `_sample_group_map` (drops pools/ungenotyped).
- Route + gate exactly like the Song builder: routed clusters + evidence genes from
  `edge_slices/incytr_pathways_fivexfad_<tissue>/index.json` (`present` pairs) and the pair
  shards' `Ligand/Receptor/EM/Target` columns. Filter rows to evidence genes **before**
  expanding per-cluster (memory).
- Deconvolve per sample: for each sample column (condition `cond`), for each routed cluster
  `cl` present in `cond`: `value = (N_total[cond]/N_per[(cl,cond)]) × lin(sample) × share[cond][cl]`.
- Assert the reconciliation gate vs the condition-level `*_deconvoluted.csv`.
- Write one parquet shard per cluster under `audit_sources/omics_trace_fivexfad_<tissue>/<slug>.parquet`,
  in the **Song omics_trace schema** so the JS store reads it unchanged:
  ```
  layer: "protein"|"phospho_ps"|"phospho_py"; gene_symbol; site_id (null for protein);
  motif (null for protein); animal_id (= biological_sample_id); genotype ("TG"|"WT");
  timepoint ("3mo".."12mo"); value (linear deconvolved); log2_value (log2(value), NaN@0)
  ```
  (Phospho motif joined from the same site-keyed source, mirroring the Song builder.)
- Write `index.json` (`omics_schema_version`, `relative_path`, `clusters`,
  `layers:["protein","phospho_ps","phospho_py"]`, `filename_template`, `sanitize_rule`).

### 2. New builder — `alz/integration/build_transcript_trace_fivexfad.py`
5xFAD analog of `build_transcript_trace.py`, per tissue:
- Read `fivexfad_snrna_pseudobulk_linear.csv.gz`; restrict to routed clusters + evidence genes.
- Aggregate scRNA samples → condition mean per (cluster, gene, condition); reshape to the
  **Song transcript_trace schema**: `gene, group (=<geno>_<age>mo), genotype, timepoint, value`.
- Write per-cluster shards under `audit_sources/transcript_trace_fivexfad_<tissue>/` + index.

### 3. `alz/viewer/paths.py`
Add per-tissue dir + index + schema-version constants:
`OMICS_TRACE_FIVEXFAD_{CORTEX,HIPPO}_DIR/_INDEX`, `OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION=1`,
`TRANSCRIPT_TRACE_FIVEXFAD_{CORTEX,HIPPO}_DIR/_INDEX`, `TRANSCRIPT_TRACE_FIVEXFAD_SCHEMA_VERSION=1`.

### 4. `alz/build_unified_viewer.py`
- Add `ensure_5xfad_omics_trace_sources(tissue)` / `ensure_5xfad_transcript_trace_sources(tissue)`,
  mirroring the Song `ensure_*` (version-gated rebuild, return the per-context block).
- Convert `meta["omics_trace"]` and `meta["transcript_trace"]` to **by_context** (JS already
  has the `m.by_context` branch — never populated today):
  ```python
  meta["omics_trace"]     = {"by_context": {"song_ad": ensure_omics_trace_sources(), ...5xfad}}
  meta["transcript_trace"]= {"by_context": {"song_ad": ensure_transcript_trace_sources(), ...5xfad}}
  ```
  (Required even though both cohorts now populate it — flat would return Song's index under
  a 5xFAD context, and cluster names overlap, so the wrong shards would load.)
  `meta["omics_trace_normalized"]` stays flat (build-time gate only; not read by the viewer).
- Capabilities (~line 1039): flip `omics_trace: True` AND `transcript_trace: True` for the
  two 5xFAD contexts.

### 5. `alz/viewer_shared/template/js/widgets/evidence_row.js` + `transcript_trace.js`
Make cohort detection **context-aware** (the unified viewer hosts 5xFAD as contexts under a
`song_ad` payload, so `PAYLOAD.meta.cohort` is the wrong axis):
- `_contextCohort()`: resolve `ViewerPayload.activeContext()` in `ViewerPayload.contexts()`,
  return its `.cohort`; fall back to `PAYLOAD.meta.cohort`. Replace `_cohort()`/`_panelCohort()`
  bodies (in both files).
- `contrastToArms`: add `fivexfad` branch — `TG_<age>` →
  `[{arm:"TG", group:"TG_<age>mo"}, {arm:"WT", group:"WT_<age>mo"}]`.
- `rowGroupKey`: `fivexfad` → `<genotype>_<timepoint>` (no males-only sex filter).
- `_FIVEXFAD_LAYER_UNITS`: protein "deconvoluted abundance"; phospho "deconvoluted
  abundance, gene-mean"; transcript "log-norm pseudobulk mean". Route via `_layerUnit`.
- `render()` header: arms label "TG vs WT @ <age>"; n-label "deconvoluted · per-sample
  (TG vs WT)". WT-baseline colour already keys on `arm === "WT"` — works as-is.

### 6. Runner / orchestration
`build_unified_viewer` calls the two `ensure_5xfad_*` during the payload build; shards write
directly into `audit_sources/`. No new pixi task. The per-sample deconvolution reads only
already-on-disk per-sample matrices — `run_export_bulk` / the condition-level path are untouched.

## Memory safety
Inputs are small (total proteome ~3.9 MB; per-sample phospho 39,666 sites × ~26 cols; scRNA
pseudobulk 831 × 5,119). The blow-up is per-sample × per-cluster expansion — filter to routed
evidence genes (a few thousand) **before** expanding, process one tissue at a time, and run
the standalone builders under `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`.
The 7.99 M-row `fivexfad_celltype_site_level_ols.parquet` is NOT read.

## Verification
1. Run both builders (capped) per tissue; assert 31 shards each, spot-check schema/values,
   and that the **reconciliation gate** (per-sample mean == condition-level deconvoluted)
   passes < 1e-6 for pr/ps/py.
2. Rebuild payload + `index.html`. Assert `meta.{omics,transcript}_trace.by_context` each have
   3 keys with non-empty `clusters` for all three, and 5xFAD contexts have
   `omics_trace: True` / `transcript_trace: True`.
3. `node --check` on `evidence_row.js` + `transcript_trace.js`.
4. Node logic harness (like the ui_chrome check): feed a 5xFAD pathway row → assert protein/
   phospho cells produce multi-dot bars, transcript a single bar, all four with the stored
   chip, and `contrastToArms("TG_6mo")` → TG_6mo/WT_6mo.
5. Serve + hard-refresh: 5xFAD mode → Incytr Pathways → expand a row → all four layers render
   with TG-vs-WT bars + chips; per-site phospho popover works.

## Files
- `alz/integration/build_omics_trace_fivexfad.py` (new)
- `alz/integration/build_transcript_trace_fivexfad.py` (new)
- `alz/viewer/paths.py`
- `alz/build_unified_viewer.py`
- `alz/viewer_shared/template/js/widgets/evidence_row.js`
- `alz/viewer_shared/template/js/widgets/transcript_trace.js`
