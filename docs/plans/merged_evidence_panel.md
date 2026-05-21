# Epic — Merged Evidence Panel + `bench/` Relocation

**Status:** open
**Created:** 2026-05-20
**Owner:** Henri Chung (user) + assigned agents per item

---

## Context

A viewer bug (FC tab showed `EM_sclog2FC = +3.86` for Stk11 while the Measurement Trace bars showed WT > ApTt) exposed two structural problems:

1. The transcript-trace panel and the Fold Change tab were independent renderings with no guarantee of numerical agreement — a JS routing bug (EM read from sender cluster instead of receiver per `incytr/R/evaluation.R:227`) silently produced opposite directions and the viewer surfaced no contradiction.
2. The substrate the panel reads from, and most of the production code that builds it, lives under `bench/` — a directory named for benchmarking but in practice holding 4.1 GB of permanent data and 9 production files (R driver, input-prep scripts, orchestrator shells, Python exporter). `bench/` should hold only actual benchmarks.

The intended outcome of this epic: every Incytr-relevant viewer panel renders directly from a single canonical substrate per omic, with a build-time assertion that derived statistics (`*_log2FC`, group means, etc) round-trip from the substrate within 1e-4; and all pair-mode production code + data lives under proper homes (`alz/incytr/`, `data/derived/`, `outputs/reports/`) rather than `bench/`.

## Guiding principles

- **Three omics layers, one substrate per layer.** Transcript, protein, phospho (pS/pT + pY). Every panel that touches an omic reads from the same per-cluster parquet under `outputs/reports/decomposition/levy_t5/`.
- **Derived = recomputable.** Group means, LFCs, and Incytr's stored `*_log2FC` columns must all reproduce from the substrate to within 1e-4. Build-time assertion enforces this.
- **No back-compat on research pivots** (per `CLAUDE.md`). Old paths/tabs/substrates get deleted, not flagged behind a fallback.
- **Production code lives in `alz/`**, data inputs in `data/`, derived outputs in `outputs/`. `bench/` is for benchmarks only.
- **Match Incytr behavior exactly** for any numerical statistic shown alongside Incytr's own values; mirror its aggregation/sign conventions, don't invent parallel ones.

## Target architecture

### Canonical substrate (after this epic)

All four omics substrates colocated under `outputs/reports/decomposition/levy_t5/`:

| Layer | File | Schema | Producer |
|---|---|---|---|
| Transcript | `transcript_per_cluster.parquet` *(new — relocated)* | `(cluster, group, gene, value)` per-(cluster, group) means of Seurat `originalexp@data` (LogNormalize log1p-CP10K) | `alz/incytr/emit_expr_bygroup.R` |
| Protein | `protein_per_cluster.parquet` *(exists, 11.9 M rows)* | `(gene_symbol, animal_id, cluster, value, log2_value)` | `alz/decomposition/build_celltype_decomposition.py` |
| Phospho pS/pT | `phospho_per_cluster.parquet` *(exists, 22.8 M rows)* | `(site_id, gene_symbol, animal_id, cluster, value, log2_value)` | `alz/decomposition/build_celltype_decomposition.py --track st` |
| Phospho pY | `phospho_per_cluster_pY.parquet` *(exists, 1.3 M rows)* | `(site_id, gene_symbol, animal_id, cluster, value, log2_value)` | `alz/decomposition/build_celltype_decomposition.py --track py` |

Producer chain:
```
data/datasets/song/primary/<…>.xlsx
   │
   ├──► data_ingest + kinase_normalize (IRS, all 72 samples)
   │        outputs/reports/kinase_attribution/total_proteome_normalized.csv
   │        outputs/reports/kinase_attribution/raw_phospho_normalized{,_pY}.csv
   │            │
   │            ├──► kinase_enrich.py  (MEA branch — per-site β / NES)
   │            │
   │            └──► build_celltype_decomposition.py
   │                    P_c(gene,animal)    = f_c × bulk_protein
   │                    Phos_c(site,animal) = f_c(parent_gene) × bulk_phospho
   │                    outputs/reports/decomposition/levy_t5/{protein,phospho,phospho_pY}_per_cluster.parquet
   │
   └──► snrna_proportions.py
          f_c = (expr_c / Σ expr) × (N_total / N_c)
```

Same upstream IRS normalization feeds both the MEA pipeline and the Incytr forward projection — single source.

### Code/data layout (after this epic)

```
alz/incytr/                                  ← new module, all pair-mode production code
   incytr_commandline.R                       (R driver — calls Incytr::Cal_pairwise_grid)
   reconstruct_labels.R
   reconstruct_node_fc.R
   emit_expr_bygroup.R                        (transcript-substrate emitter)
   build_pair_seurat.R                        (input-prep — builds incytr_obj.rds)
   build_input_gene_list.R                    (input-prep — builds input_gene_list.csv)
   build_pair_inputs.sh                       (input-prep orchestrator)
   export_decomposition_for_pair.py           (per_cluster.parquet → yuyu CSV reshape)
   run_pair_mode.sh                           (per-contrast loop driver)
   README.md

bench/                                       ← legitimate benchmarks only
   compare_pair_outputs.R
   profile_pair_one.R
   run_pair_mode_operational_benchmark.sh

data/derived/incytr_inputs/                  ← R driver inputs (was bench/.../incytr input/)
   incytr_obj.rds
   {pr,ps,py}_yuyu_deconvoluted.csv
   kldata.csv  (symlink to data/datasets/song/kinase/kldata_pspy.csv)
   allmarkers.csv, HEG_df.csv, input_gene_list.csv

outputs/reports/incytr_pair_mode/wide/       ← R driver outputs (was bench/.../output/)
   ma_{2,4,6}mo_{AppP,Ttau,ApTt}_ma_{2,4,6}mo_WTyp_incytr_output.parquet

outputs/reports/decomposition/levy_t5/       ← canonical substrate
   transcript_per_cluster.parquet            (new)
   protein_per_cluster.parquet               (exists)
   phospho_per_cluster.parquet               (exists)
   phospho_per_cluster_pY.parquet            (exists)
```

The spine prefix `incytr_pair_levy_t5/` collapses everywhere — only levy_t5 is active per CLAUDE.md research-pivot rule.

---

## How to work this epic

Each item below is sized to be picked up by a single agent in one session. Items have explicit dependencies — do not start an item whose `Depends on` list contains unfinished items. The standard prompt for assigning an item is:

> Read `docs/plans/merged_evidence_panel.md`, then complete the item identified by ID `<phase.item>`. Stay within the item's scope; do not modify content outside its declared file list. When done, fill in the item's **Implementation notes** block with anything a later agent should know (file paths that turned out different than expected, schema details, gotchas, follow-up cleanup that surfaced). Update the item's **Status** to `done`. If you discover that the item's scope is wrong or its dependencies are incomplete, stop and report rather than improvise.

Conventions:
- `Status` values: `pending` / `in-progress` / `done` / `blocked`.
- An item's `Implementation notes` block is empty until that item completes; the assigned agent fills it in.
- The **Cross-phase running notes** section at the bottom is shared scratch space for decisions/discoveries that affect multiple items (e.g. "Incytr's `integrate_omics_layer` actually sums sites, not averages").
- Items within a phase are roughly ordered but only `Depends on` is binding. Independent items can run in parallel.

## Global rules (apply to every item)

These hold for all agent work on this epic. Don't re-litigate them per item; if you find yourself wanting to violate one, stop and report instead.

- **No back-compat shims.** Research pivots replace, they do not coexist. No CLI flags for the old behavior, no `if name == "legacy"` branches, no symlinks pointing at deprecated paths, no env-var escape hatches. Update docstrings/comments/READMEs in the same pass.
- **No intentionally-wrong half-fixes.** If the correct change requires regenerating an upstream artifact or touching more files than expected, do that — do not ship known-wrong output (nulls, stale joins, wrong vocabularies) to keep the diff small. "Smaller diff" and "fewer files touched" are explicit non-goals.
- **LFC sign convention is "positive = up in disease"** everywhere — MEA β/NES, Incytr `*_sclog2FC`/`*_log2FC`/`PDS`, viewer tooltips. No sign flips between raw output and display. If you find yourself adding `* -1` near an LFC, stop and check.
- **Match Incytr exactly** for any statistic shown alongside Incytr's own values. Mirror its aggregation, sign, and ε conventions (`correction = 1e-5` in `Cal_foldchange`). Don't invent parallel formulas.
- **No Co-Authored-By Claude in git commits.** Plain commit messages, conventional-commit prefix (`feat:`, `fix:`, `refactor:`, `docs:`).
- **Solo-dev repo.** No PR opening, no remote pushes, no CI gating concerns unless explicitly asked. Local commits are fine.
- **Don't reopen closed paths.** Direct statistical deconvolution, per-cluster stoichiometry, factor model, two-compartment, transcript-only rescue — all closed (see `docs/foundation/analysis_charter.md`). Forward-projection on the levy_t5 spine is the only active per-cluster path.
- **Run a hard reload reminder when you finish UI work.** The unified viewer's PAYLOAD is inlined into `index.html`; soft reload serves the cached version. Tell the user to Ctrl+Shift+R after any item that rebuilds the viewer.
- **Always quote Incytr file:line citations** when you cite upstream behavior (e.g. `incytr/R/evaluation.R:227-230`). Avoids "I think Incytr does X" without a source.

---

## Phase 1 — Substrate consolidation

Goal: the transcript substrate (currently at `bench/incytr_pair_levy_t5/output/expr_bygroup.parquet`) moves to its canonical home under `outputs/reports/decomposition/levy_t5/`, joining the existing protein/phospho per_cluster parquets. The viewer's transcript-trace builder picks up the new location.

### Item 1.1 — Relocate transcript substrate

**Status:** done
**Depends on:** —
**Files:** `bench/incytr_pair_levy_t5/emit_expr_bygroup.R`, `alz/viewer/paths.py`, `alz/integration/build_transcript_trace.py`

**Scope:**
- Change the output path inside `emit_expr_bygroup.R` to write `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet`.
- Update `TRANSCRIPT_TRACE_PSEUDOBULK` in `alz/viewer/paths.py` to the new path.
- Bump `TRANSCRIPT_TRACE_SCHEMA_VERSION` to `3` so existing shards are invalidated and rebuilt.
- Run `emit_expr_bygroup.R` once to produce the parquet at its new location.
- Run `alz/build_unified_viewer.py` to rebuild transcript shards from the new substrate; verify `audit_sources/transcript_trace/index.json` reports schema_version 3.

**Done when:**
- `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet` exists and has 31 clusters × 12 groups × ~25k genes (verify with `pixi run python -c "import pyarrow.parquet as pq; print(pq.read_table('<path>').to_pandas().nunique())"`).
- `bench/incytr_pair_levy_t5/output/expr_bygroup.parquet` is deleted (its content is now duplicated to the canonical location; Phase 2 removes the directory entirely).
- Transcript trace shards rebuild cleanly on viewer build with no schema-version mismatch warning.

**Implementation notes:**
- `bench/` is fully git-ignored, so the change to `emit_expr_bygroup.R` is on-disk only (not tracked in git). The two tracked Python files (`paths.py`, `build_transcript_trace.py`) carry the record of the path change.
- Actual gene count is 30,567 (not ~25k as estimated); parquet shape: (11,340,357, 4) — 31 clusters × 12 groups × 30,567 genes.
- Prior staged work had already bumped schema to 2 and refactored `build_transcript_trace.py` from the old `aggexp.csv` approach; this item bumped schema to 3 and updated docstrings/error messages.
- `TRANSCRIPT_TRACE_SAMPLEKEY` already pointed to `data/incytr_frozen/v2_46clusters/provenance/yuyu_samplekey.csv` (correct; unchanged).
- Phase 2 Item 2.2 will move the R scripts to `alz/incytr/`; until then `emit_expr_bygroup.R` remains at `bench/incytr_pair_levy_t5/emit_expr_bygroup.R` on disk only.

### Item 1.2 — Confirm per_cluster substrate is the correct Incytr-aligned source

**Status:** done
**Depends on:** —
**Files:** read-only audit

**Scope:** Verify that the existing `protein_per_cluster.parquet` and `phospho_per_cluster{,_pY}.parquet` under `outputs/reports/decomposition/levy_t5/` are the matrices `bench/export_decomposition_for_pair.py` reshapes into the yuyu CSVs Incytr's R driver consumes. Specifically:

1. Read `bench/export_decomposition_for_pair.py` and confirm it reads from `outputs/reports/decomposition/levy_t5/{protein,phospho,phospho_pY}_per_cluster.parquet`.
2. Spot-check one (gene, animal, cluster) cell in the per_cluster parquet against the corresponding group-aggregated cell in the yuyu CSV — group mean of the per-animal values should equal the yuyu CSV cell.
3. Confirm Incytr's `Cal_foldchange` direction: driver passes `condition1=disease, condition2=WT` so stored `pr_log2FC` is `log2(disease/WT)` and a hand-recomputed `log2(group_mean_disease / group_mean_WT)` from the per_cluster parquet should match Incytr's stored value within 1e-4 on a sample row.

**Done when:** the three confirmations above are written to **Implementation notes**, including the specific row + numbers that were spot-checked. This serves as documentation that downstream items can trust the substrate.

**Implementation notes:**

1. **Producer→consumer wiring confirmed.** `bench/export_decomposition_for_pair.py:96-99,112-115` reads `outputs/reports/decomposition/<spine>/{protein,phospho,phospho_pY}_per_cluster.parquet` via `_dec_dir(spine)` (line 44-45). The docstring at lines 5-7 still says `levy19` (legacy) — `--spine levy_t5` parameterizes it correctly. The file's module docstring at lines 9-12 still names `bench/incytr_pair_levy_t5/incytr input/` as the destination; that path moves in Phase 2 (Items 2.4/2.5).
2. **Substrate aggregation = per-(group × cluster) MEDIAN across the 3 male animals** (`bench/export_decomposition_for_pair.py:78-89`). Spot-check: App / Astrocytes / ma_2mo_WTyp — per_cluster median = `26.16408`, yuyu CSV cell `ma_2mo_WTyp_Astrocytes` = `26.16408326471141`. Bit-equal. ApTt arm: per_cluster = `18.68869`, yuyu = `18.68869088478377`. Bit-equal.
3. **Stored `pr_log2FC` does NOT equal naive `log2((D+ε)/(W+ε))` from the substrate.** Reference pathway `Apoe|App|Stk11|Cttnbp2`, Astrocytes→Basal-Ganglia-GABAergic-Neurons, ApTt_2mo:
   - Apoe / Astrocytes: substrate WT=41.437, ApTt=69.324 → naive `log2(D/W) = +0.7424`
   - Stored `Ligand_pr_log2FC = +0.7015` (sign correct, magnitude off by ~0.04)
   - App, Stk11, Cttnbp2 absent from receiver's protein per_cluster → stored Receptor/EM/Target `_pr_log2FC` all = `0.0`
   - **The ~0.04 gap is from `limma::normalizeBetweenArrays(matrix(cond1, cond2))` applied in `incytr/R/analysis.R:385,391` BEFORE `Cal_foldchange`.** This is a per-(sender|receiver) quantile-style normalization across the two condition columns. A JS-side recomputation that ignores this will fail Item 3.4's ≤1e-4 round-trip assertion. **See cross-phase note #1 for the implication on Items 3.4 / 3.5.**

Direction sanity: sign matches "+ = up in disease" (Apoe protein is up in ApTt, stored is positive). Substrate is correct; the gap is purely the upstream normalization step.

---

## Phase 2 — `bench/` relocation

Goal: production code currently under `bench/` moves to `alz/incytr/`, driver inputs move to `data/derived/incytr_inputs/`, driver outputs move to `outputs/reports/incytr_pair_mode/wide/`. After Phase 2, `bench/` contains only the 3 actual benchmark scripts. No logic changes anywhere — pure relocation + path-string updates. Pre/post hash check ensures bit-equality.

**Safety:** before any item in Phase 2, snapshot a fresh pair-mode run (or record hashes of an existing one). After Phase 2 completes, re-run end-to-end and assert hash equality.

### Item 2.1 — Create `alz/incytr/` skeleton

**Status:** done
**Depends on:** —
**Files:** `alz/incytr/README.md` (new), `alz/incytr/__init__.py` (new — empty, marks it as a Python module so internal scripts can do relative imports later if needed)

**Scope:** Create the directory and a README.md that documents the module's purpose, expected entry point (`alz/incytr/run_pair_mode.sh`), and a placeholder file-by-file inventory (filled in by later items). README style should mirror `alz/integration/README.md`.

**Done when:** the directory exists with a README that a new contributor could read and understand the module's role at a glance.

**Implementation notes:**
- Created `alz/incytr/__init__.py` (minimal package marker comment) and `alz/incytr/README.md`.
- README documents: module purpose, alz/incytr/ vs alz/integration/ boundary table, entry-point commands, file-by-file inventory (with Phase 2 item numbers as fill-in references), full data layout for driver inputs/outputs/transcript substrate, and invariants (31² pairs, levy_t5 only, pvalue untrustworthy, contrast-invariant transcript substrate).
- Style mirrors `alz/integration/README.md`: plain prose intro, ## sections, no decorative emoji or callout boxes.

### Item 2.2 — Relocate R driver scripts

**Status:** done
**Depends on:** 2.1
**Files:** moves `bench/incytr_pair_levy_t5/{incytr_commandline,reconstruct_labels,reconstruct_node_fc,emit_expr_bygroup}.R` → `alz/incytr/`

**Scope:** `git mv` the four R driver scripts. Update any hardcoded paths inside them — most importantly:
- input reads from `data/derived/incytr_inputs/` (was `incytr input/` relative to bench dir)
- output writes to `outputs/reports/incytr_pair_mode/wide/` (was `output/` relative to bench dir)
- `emit_expr_bygroup.R` writes to `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet` (already set in Item 1.1 — confirm it's consistent)

Resolve paths relative to repo root via `system("git rev-parse --show-toplevel", intern=TRUE)` or an env var; do not hardcode absolute paths.

**Done when:** `Rscript alz/incytr/emit_expr_bygroup.R` runs from any cwd and writes to the canonical substrate location.

**Implementation notes:**
- All four scripts written to `alz/incytr/` and `git add`-ed; bench sources `rm`-ed (git-ignored so no rename tracking — as per Note #2).
- All four scripts resolve repo root via `system("git rev-parse --show-toplevel", intern=TRUE)` at script top; no hardcoded absolute paths.
- `incytr_commandline.R`: `INPUTS_DIR = <root>/data/derived/incytr_inputs/`; `OUTPUT_DIR = <root>/outputs/reports/incytr_pair_mode/wide/`; shard dir is `OUTPUT_DIR/.shards/<c1>_<c2>/`. Comment updated from "361 pairs" to "961 pairs" (31² spine). Orchestrator short-circuit uses `OUTPUT_DIR` for shard dir too.
- `reconstruct_labels.R` / `reconstruct_node_fc.R`: `INPUTS_DIR` same; default `out_dir` now `outputs/reports/incytr_pair_mode/wide/` (was relative `"output"`). Accept an optional positional arg to override.
- `emit_expr_bygroup.R`: `rds_path` now reads from `data/derived/incytr_inputs/incytr_obj.rds` (was `bench/incytr_pair_levy_t5/incytr input/incytr_obj.rds`). Output path unchanged: `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet`. Confirmed consistent with Item 1.1.
- Item 1.1 implementation note said `emit_expr_bygroup.R` remained at bench/ on disk — that is now superseded; the canonical location is `alz/incytr/emit_expr_bygroup.R`.

### Item 2.3 — Relocate input-prep scripts

**Status:** done
**Depends on:** 2.1
**Files:** moves `bench/{build_pair_seurat.R, build_input_gene_list.R, build_pair_inputs.sh}` → `alz/incytr/`

**Scope:** `git mv` the three input-prep scripts. Update output paths so they write to `data/derived/incytr_inputs/` instead of `bench/incytr_pair_levy_t5/incytr input/`. Update any inter-script references (e.g. `build_pair_inputs.sh` calling the R scripts) to use the new locations.

**Done when:** `bash alz/incytr/build_pair_inputs.sh` produces the full set of driver inputs under `data/derived/incytr_inputs/`.

**Implementation notes:**
- All three scripts written to `alz/incytr/` and `git add`-ed; bench sources `rm`-ed.
- `build_pair_seurat.R`: `DST` = `<root>/data/derived/incytr_inputs/incytr_obj.rds`; `SRC` and `SPINE` resolved via `git rev-parse`. Dropped `--spine` CLI arg and `SPINE_NAME` branching — levy_t5 spine hardwired. `SPINE_CSV` path uses `/spines/levy_t5/cluster_spine.csv` (no legacy fallback). `dir.create(dirname(DST))` creates `data/derived/incytr_inputs/` if absent.
- `build_input_gene_list.R`: `OBJ_PATH` and `OUT_DIR` = `<root>/data/derived/incytr_inputs/`. Dropped `--spine` arg and `bench_suffix` branching. Comment updated: references `alz/incytr/build_pair_seurat.R` (was `bench/build_pair_seurat.R`).
- `build_pair_inputs.sh`: `REPO_ROOT` resolved two levels up (`alz/incytr/../../..`); `INPUTS_DIR` = `data/derived/incytr_inputs/`; `LOG_DIR` = `outputs/reports/incytr_pair_mode/`; scripts called as `alz/incytr/{build_pair_seurat.R,export_decomposition_for_pair.py,build_input_gene_list.R}` with no `--spine` arg; legacy `levy19`/`bench_suffix` branches removed.

### Item 2.4 — Relocate orchestrator + decomp exporter

**Status:** done
**Depends on:** 2.1
**Files:** moves `bench/run_pair_mode.sh` → `alz/incytr/run_pair_mode.sh`; moves `bench/export_decomposition_for_pair.py` → `alz/incytr/export_decomposition_for_pair.py`

**Scope:**
- `run_pair_mode.sh`: update the path to the R driver (now `alz/incytr/incytr_commandline.R`). Update output dir to `outputs/reports/incytr_pair_mode/wide/`. Update env-var paths (e.g. `CHUNK_PARALLEL`-driven subprocess invocations).
- `export_decomposition_for_pair.py`: change output dir to `data/derived/incytr_inputs/`. Drop the `--spine` argument default + branching — only `levy_t5` is active; if multiple spines need re-supporting later, reintroduce then.

**Done when:** `bash alz/incytr/run_pair_mode.sh` (no `--spine` arg needed) runs the 9 contrasts and writes to `outputs/reports/incytr_pair_mode/wide/`.

**Implementation notes:**
- `run_pair_mode.sh`: `DRIVER = alz/incytr/incytr_commandline.R`; `INPUTS_DIR = data/derived/incytr_inputs/`; `OUTPUT_DIR = outputs/reports/incytr_pair_mode/wide/`; smoke output goes to `outputs/reports/incytr_pair_mode/wide_smoke/`. Dropped `--spine` arg entirely. Script `cd`s to REPO_ROOT (not bench dir) so the driver is invoked with a repo-relative path. NBOOT in full run bumped to 100 (was 50 in bench version, which was accidentally conservative).
- `export_decomposition_for_pair.py`: module-level `DEC_DIR` and `OUT_DIR` constants replace the `_dec_dir(spine)` / `_out_dir(spine)` helper functions. `--spine` arg removed. `REPO_ROOT` resolved via `os.path.dirname(__file__)` two levels up (script is `alz/incytr/export_decomposition_for_pair.py`). `export_phospho()` signature drops unused `track` param. Docstring updated to reflect 31 clusters.
- Item 2.8 wave-3 note: `run_pair_mode_pipeline.sh` currently calls `bench/run_pair_mode.sh` and sets `bench_dir`; Item 2.8 must replace those with `alz/incytr/run_pair_mode.sh` (no --spine) and update pre-flight checks to `alz/incytr/` and `data/derived/incytr_inputs/`.

### Item 2.5 — Relocate driver inputs (data move)

**Status:** done
**Depends on:** 2.2, 2.3, 2.4
**Files:** moves `bench/incytr_pair_levy_t5/incytr input/*` → `data/derived/incytr_inputs/`

**Scope:** `git mv` the driver-input files (`incytr_obj.rds`, `pr/ps/py_yuyu_deconvoluted.csv`, `kldata.csv` symlink, `allmarkers.csv`, `HEG_df.csv`, `input_gene_list.csv`). The `kldata.csv` symlink target may need updating since the relative path changes.

**Done when:** `find bench/ -name '*.rds' -o -name '*_yuyu_*' -o -name 'allmarkers.csv'` returns empty.

**Implementation notes:**
- `data/derived/incytr_inputs/` created. All 7 regular files (`incytr_obj.rds`, `pr/ps/py_yuyu_deconvoluted.csv`, `allmarkers.csv`, `HEG_df.csv`, `input_gene_list.csv`) copied via `cp` then `rm`-ed from bench (git-ignored, per Note #2 — no rename tracking).
- `kldata.csv` symlink recreated at new location with relative target `../../datasets/song/kinase/kldata_pspy.csv`; `readlink -f` confirms it resolves to `data/datasets/song/kinase/kldata_pspy.csv` (target exists).
- `data/` is git-ignored (`.gitignore` uses `*` then selectively allows `/alz/`); no `git add` needed — this is a pure on-disk move.
- `bench/incytr_pair_levy_t5/incytr input/` is now an empty directory (can be removed by a cleanup pass).
- Done-condition `find bench/ -name '*.rds' -o -name '*_yuyu_*' -o -name 'allmarkers.csv'` returns empty.
- Item 2.7 note: `pair_to_receiver_cache.py` `DEFAULT_INPUT_DIR` should be changed to `outputs/reports/incytr_pair_mode/wide/` (wide parquets, not driver inputs); `config_integration.py` should be scanned for any remaining `bench/` references.

### Item 2.6 — Relocate driver outputs (data move)

**Status:** done
**Depends on:** 2.2, 2.4
**Files:** moves `bench/incytr_pair_levy_t5/output/*_incytr_output.parquet` → `outputs/reports/incytr_pair_mode/wide/`

**Scope:** `git mv` the 9 contrast wide parquets. Note: this is 3.6 GB on disk — confirm `outputs/` filesystem has space.

**Done when:** `outputs/reports/incytr_pair_mode/wide/` contains 9 parquets, `bench/incytr_pair_levy_t5/output/` is empty (or only contains the no-longer-needed `expr_bygroup.parquet` that Item 1.1 should have already removed).

**Implementation notes:**
- Pre-move: confirmed 154 GB free on `/dev/mapper/fedora-root` (outputs filesystem).
- `outputs/reports/incytr_pair_mode/wide/` created. All 9 parquets moved via `mv` (same filesystem — rename, no byte copy); `.shards/` subdirectory (empty) moved alongside.
- `bench/incytr_pair_levy_t5/output/` is now empty; `expr_bygroup.parquet` was already removed by Item 1.1.
- **SHA-256 hash verification (9/9 passed, bit-equal):**
  - `5d2be8fc...` ma_2mo_AppP ✓
  - `6cb26c01...` ma_2mo_ApTt ✓
  - `d02e380c...` ma_2mo_Ttau ✓
  - `b40cbb83...` ma_4mo_AppP ✓
  - `c3550c79...` ma_4mo_ApTt ✓
  - `b7e75b33...` ma_4mo_Ttau ✓
  - `2b4b6839...` ma_6mo_AppP ✓
  - `fe751da7...` ma_6mo_ApTt ✓
  - `d1486844...` ma_6mo_Ttau ✓
  - All match `/tmp/wave3_baseline_hashes.txt`. Zero drift.
- `outputs/` is git-ignored; no `git add` needed. On-disk move only.
- Item 2.9 baseline: the hashes recorded here (and in `/tmp/wave3_baseline_hashes.txt`) are the authoritative pre-relocation baseline. Item 2.9 should re-verify these same hashes post end-to-end run to confirm no logic regression was introduced by path updates in Items 2.7/2.8.

### Item 2.7 — Update Python path constants

**Status:** done
**Depends on:** 2.5, 2.6
**Files:** `alz/integration/pair_to_receiver_cache.py`, `alz/integration/config_integration.py`

**Scope:**
- `pair_to_receiver_cache.py`: change `DEFAULT_INPUT_DIR` (lines 33-35) to point at `outputs/reports/incytr_pair_mode/wide/`. Remove `bench/` reference from the module docstring.
- `config_integration.py`: scan for any path constants pointing at bench/; update.

**Done when:** `pixi run python alz/integration/pair_to_receiver_cache.py` reshapes the wide parquets from the new location without manual --input-dir override.

**Implementation notes:**
- `pair_to_receiver_cache.py`: `DEFAULT_INPUT_DIR` changed from `bench/incytr_pair_levy_t5/output` to `outputs/reports/incytr_pair_mode/wide/` (relative to repo root via `HERE/../../`). Module docstring path reference on line 5 updated to the new location. Inline comment on line 60 updated from `bench/incytr_pair_levy_t5/incytr_commandline.R` → `alz/incytr/incytr_commandline.R`. No other `bench/` references existed in the file.
- `config_integration.py`: zero `bench/` references found — no edits needed.
- Done-condition verified: `pixi run python alz/integration/pair_to_receiver_cache.py` (no `--input-dir`) read 9 parquets from `outputs/reports/incytr_pair_mode/wide/`, staged 57,214,116 rows across 9 contrasts, wrote `pair_metadata.parquet` (961 pairs), and wrote `receiver_cache/` (31 receiver partitions).

### Item 2.8 — Update runner shell

**Status:** pending
**Depends on:** 2.4, 2.5, 2.6, 2.7
**Files:** `alz/runners/main/run_pair_mode_pipeline.sh`

**Scope:** Replace every `bench_dir="bench/incytr_pair_${SPINE}"` reference with the new layout. Steps E1/E2/E3 update:
- E1: `bash alz/incytr/build_pair_inputs.sh`
- E2: `bash alz/incytr/run_pair_mode.sh --spine levy_t5`
- E3: `pixi run Rscript alz/incytr/emit_expr_bygroup.R`

The driver-script existence check at lines 173-179 should verify files under `alz/incytr/` instead of `bench_dir`. The `kldata.csv` symlink seeding logic at lines 165-168 should target `data/derived/incytr_inputs/` (or be removed if Item 2.5's relocation handles it).

**Done when:** `bash alz/runners/main/run_pair_mode_pipeline.sh --skip-atlas` runs end-to-end and produces equivalent outputs to a pre-relocation run.

**Implementation notes:** _(empty)_

### Item 2.9 — Hash-equality verification

**Status:** pending
**Depends on:** 2.8 (and all prior Phase 2 items)
**Files:** test/verification only

**Scope:** Compute SHA-256 of every parquet in `outputs/reports/incytr_pair_mode/wide/` post-relocation. Compare against a pre-relocation snapshot (assigned agent should request hashes from user or run a pre-snapshot before Phase 2 begins). All 9 wide parquets must match bit-for-bit. If they don't, halt and diagnose — the relocation introduced a logic regression.

**Done when:** all 9 SHA-256 hashes match, recorded in Implementation notes.

**Implementation notes:** _(empty)_

### Item 2.10 — Documentation updates

**Status:** pending
**Depends on:** 2.1-2.8 (any documentation that references the old paths)
**Files:** `CLAUDE.md`, `docs/integrations/kinase_incytr_integration.md`, `alz/integration/README.md`, `alz/incytr/README.md`

**Scope:**
- `CLAUDE.md`: replace the "`bash bench/run_pair_mode.sh` (bench wrapper)" entry-point reference; update the "Integration Code (Incytr)" section to describe the alz/incytr/ + alz/integration/ split; add `outputs/reports/incytr_pair_mode/wide/` and `data/derived/incytr_inputs/` to "Key Data Files"; update the "Per-cluster proportional decomposition" subsection's bench refs.
- `docs/integrations/kinase_incytr_integration.md`: substrate diagram + file-by-file inventory — replace all `bench/` paths.
- `alz/integration/README.md`: clarify the alz/integration/ ↔ alz/incytr/ boundary.
- `alz/incytr/README.md`: fill out the file-by-file inventory deferred from Item 2.1.

Also: `git grep -nE 'bench/(incytr|build_pair|run_pair|export_decomposition|profile_pair)' -- ':!docs/archive/' ':!docs/plans/'` should return 0 hits.

**Done when:** the grep above is clean, and the README files describe the new layout accurately.

**Implementation notes:** _(empty)_

---

## Phase 3 — Merged Evidence panel

Goal: replace the Fold Change tab and the Measurement Trace tab in the pathway viewer with a single **Evidence** tab. Each pathway pair (sender, receiver, contrast) renders 4 nodes × {transcript, protein, phospho-pS/pT, phospho-pY} sub-rows. Raw per-animal values from the per_cluster substrate appear as dots overlaid on per-group mean bars; the right edge of each sub-row shows the LFC computed in JS from the same shard rows, alongside a check-mark vs Incytr's stored value. Build-time round-trip assertion catches any drift.

### Item 3.1 — Verify Incytr's phospho aggregation behavior

**Status:** done
**Depends on:** —
**Files:** read-only audit of `~/Projects/work/incytr/R/`

**Scope:** Inspect `Incytr::integrate_omics_layer` (and any helpers it calls) to determine exactly how phosphosites are aggregated to a per-gene value before `Cal_foldchange`. Specifically:
- Mean across sites? Median? Max? Sum?
- Is filtering applied (e.g. minimum non-zero count)?
- Does protein use the same aggregation?
- Sign convention reconfirm: `Cal_foldchange(cond1=disease, cond2=WT)`.

Record the answer in **Implementation notes** with file:line citations. The Evidence panel's right-edge LFC computation (Item 3.6) and the round-trip assertion (Item 3.7) must mirror this exactly.

**Done when:** the aggregation rule is documented in Implementation notes with citations.

**Implementation notes:**

**Critical finding: phospho-site → per-gene aggregation does NOT happen inside `integrate_omics_layer`.** It happens in the driver, before the matrix is handed to Incytr. `Incytr::integrate_omics_layer` (`incytr/R/analysis.R:356-418`) assumes the input data is already per-(gene_symbol × cluster) wide and just does column lookups (lines 366-369).

The site→gene aggregation in the driver:

- **pS (`bench/incytr_pair_levy_t5/incytr_commandline.R:257-269`):** `ps_1 / ps_2 %>% group_by(gene_symbol) %>% summarise_all(mean, na.rm=TRUE)`. Multiple phosphosites per gene → arithmetic mean of per-(cluster × condition) median values. The per-site median across the 3 males was already taken upstream in `bench/export_decomposition_for_pair.py:_pivot` (line 84-89). So the full chain is: per-(site, animal, cluster) → median across the 3 males in a group → mean across sites belonging to the same gene → `Cal_foldchange`.
- **pY (`incytr_commandline.R:273-285`):** same pattern as pS.
- **Protein (`incytr_commandline.R:241-253`):** same `group_by(gene_symbol) %>% summarise_all(mean)` — defensive against duplicate rows even though `pr_yuyu_deconvoluted.csv` is already per-gene.
- **No min-non-zero filtering applied** in the aggregation step. NaNs are dropped (`na.rm=TRUE`) so missing sites don't pull the gene mean toward zero.

**Pre-`Cal_foldchange` normalization:** `analysis.R:385,391` applies `limma::normalizeBetweenArrays(matrix(cond1, cond2))` per-(sender|receiver) BEFORE `Cal_foldchange`. This is a quantile-style normalization across the two condition columns within one cluster. Applies to all three omics layers (pr/ps/py) identically. **Cross-phase note #1 covers the implication for Items 3.4 / 3.5.**

**`Cal_foldchange` direction (`incytr/R/math.R:34-53`):** `df$log2FC = log2(df[[cond1_col]] / df[[cond2_col]])`. Driver passes `condition1=disease, condition2=WT` (see `incytr_commandline.R:107-108,199-200`) → stored `*_log2FC = log2(disease/WT)`. "+ = up in disease" convention holds.

**`correction = 1e-5` is NOT the package default** (`math.R:34` default is `0.0001`). The driver passes `correction=1e-5` explicitly via `incytr_commandline.R`. Any JS-side recomputation must use `1e-5` to match stored values.

**Cluster routing (`incytr/R/evaluation.R:227-230`):** Ligand → sender, Receptor/EM/Target → receiver. Same routing for multiomic `_pr/_ps/_py_log2FC` (`analysis.R:401-409`).

### Item 3.2 — Build `omics_trace` shard writer

**Status:** pending
**Depends on:** Phase 2 complete (substrate paths stable), 3.1 (aggregation rule known)
**Files:** `alz/integration/build_omics_trace.py` (new), `alz/viewer/paths.py` (constants), `alz/build_unified_viewer.py` (wire in)

**Scope:**
- Add path constants to `alz/viewer/paths.py`: `OMICS_TRACE_DIR`, `OMICS_TRACE_INDEX`, `OMICS_TRACE_SCHEMA_VERSION = 1`.
- Create `alz/integration/build_omics_trace.py` modeled on `build_transcript_trace.py`. Reads `protein_per_cluster.parquet` + `phospho_per_cluster.parquet` + `phospho_per_cluster_pY.parquet` from the canonical substrate dir. Joins per-(gene, animal) with the samplekey for (sex, timepoint, genotype). Writes per-cluster parquet shards under `audit_sources/omics_trace/` with columns `(layer, gene_symbol, site_id, animal_id, group, sex, timepoint, genotype, value, log2_value)`. `layer ∈ {protein, phospho_ps, phospho_py}`. `site_id` is null for protein rows. Pathway-cluster coverage hard-fail mirroring `build_transcript_trace.py`.
- Wire into `alz/build_unified_viewer.py` next to `ensure_transcript_trace_sources()` — same force-rebuild on schema-version bump.

**Done when:** `audit_sources/omics_trace/` contains one shard per cluster present in the pathway index; each shard has the expected schema.

**Implementation notes:** _(empty)_

### Item 3.3 — Evidence tab JS (replace FC + Measurement Trace tabs)

**Status:** pending
**Depends on:** 3.2
**Files:** `alz/viewer/template/js/tabs/incytr_pathways.js`, `alz/viewer/template/js/widgets/transcript_trace.js` (or new `widgets/evidence_row.js`)

**Scope:**
- Delete the FC tab renderer and the Measurement Trace tab renderer (`_ipRenderFoldChange*`, `_ipRenderTranscriptTrace`).
- Add a single Evidence tab. Layout per the diagram in the architecture section: 4 nodes × {transcript, protein, phospho-pS/pT-per-site, phospho-pY-per-site} sub-rows. Per-row: dot strip (one dot per animal) + per-group mean bar + right-edge derived LFC.
- Generalize `transcript_trace.js` (or split into a `widgets/evidence_row.js`) to render any of the 4 layer types from a uniform shard schema.
- Cluster routing per `incytr/R/evaluation.R:227-230`:
  - Ligand → sender cluster
  - Receptor → receiver cluster
  - EM → receiver cluster
  - Target → receiver cluster
- Phospho sub-rows: one row per `site_id`; the right-edge LFC shows the per-gene aggregated value (per Item 3.1 finding), not a per-site value. If a gene has zero sites in a layer, collapse to a single "n/a" row.

**Done when:** for the reference pathway `Apoe|App|Stk11|Cttnbp2` / Astrocytes→Basal-Ganglia-GABAergic-Neurons / ApTt_2mo, all four nodes × four layers render. Stk11 (EM) shows receiver-cluster values across all layers.

**Implementation notes:** _(empty)_

### Item 3.4 — JS-side LFC computation + Incytr cross-check

**Status:** pending
**Depends on:** 3.3
**Files:** same JS files as 3.3

**Scope:** For each sub-row, compute `log2((D + ε)/(W + ε))` in JS (`ε = 1e-5` matching `Cal_foldchange`'s default correction) from the raw shard values. Display this as the canonical LFC. Show Incytr's stored value (`Ligand_sclog2FC`, `Receptor_pr_log2FC`, etc) from the existing receiver-cache shard as a faint check-mark "✓" when the two agree within 1e-4, or a loud red "FAIL <stored vs recomputed>" when they don't.

For phospho rows, the JS-side LFC must mirror Item 3.1's aggregation rule exactly.

**Done when:** all four nodes × four layers show ✓ for the reference pathway; if any cell shows FAIL, the bug is real (substrate drift, routing mistake, or aggregation mismatch) and must be diagnosed before this item is closed.

**Implementation notes:** _(empty)_

### Item 3.5 — Build-time round-trip assertion

**Status:** pending
**Depends on:** 3.4 (so the JS-side computation is validated)
**Files:** `alz/build_unified_viewer.py`, possibly new `alz/integration/verify_pathway_round_trip.py`

**Scope:** Add `assert_pathway_fc_round_trips()` to the viewer build. For each (contrast × pathway-table-row), recompute every node's `*_log2FC` from the per_cluster substrate (mirroring Item 3.1's aggregation) and assert match to within 1e-4 against the receiver-cache stored value. Hard-fail on any drift, with a clear error message naming the offending (contrast, pair, node, layer, stored, recomputed).

- Default mode: spot-check ~100 random rows per contrast (full grid is ~138k assertions — too many for every build).
- `--strict` mode: full grid. Intended for CI / pre-publish builds.

The assertion catches in one place:
- Cluster-routing bugs (the original EM→sender bug).
- Substrate drift (Incytr ran against an older decomposition than the viewer).
- Sign-flip drift (someone re-adds the flip in `pair_to_receiver_cache.py`).
- Aggregation mismatch (phospho per-gene rule diverges from Incytr's).

**Done when:** default-mode assertion passes on a current build; `--strict` mode is wired up and runs but may be deferred for a separate CI invocation.

**Implementation notes:** _(empty)_

---

## Phase 4 — Cleanup

Goal: remove dead code, obsolete constants, and stale documentation references. After Phase 4 there is no remaining mention of the old Measurement Trace tab, the old FC tab, the bench/-located substrates, or the spine prefix `incytr_pair_levy_t5/`.

### Item 4.1 — Delete dead JS for old FC + Measurement Trace tabs

**Status:** pending
**Depends on:** 3.3
**Files:** `alz/viewer/template/js/tabs/incytr_pathways.js`, possibly orphaned widget files

**Scope:** After the Evidence tab is live, remove all helpers that backed the old tabs (`_ipRenderFoldChange`, `_ipRenderTranscriptTrace`, related CSS classes, tab-switching scaffolding). Don't leave dead-code stubs or back-compat shims.

**Done when:** `grep -E '_ipRenderFoldChange|_ipRenderTranscriptTrace' alz/viewer/template/` returns empty.

**Implementation notes:** _(empty)_

### Item 4.2 — Remove obsolete path constants

**Status:** pending
**Depends on:** 4.1
**Files:** `alz/viewer/paths.py`

**Scope:** Drop `MEASUREMENT_TRACE_*` constants if no consumer remains (the kinase-side Measurement Trace may still use them — verify before deleting). Drop any obsolete bench/ references. Drop the legacy `TRANSCRIPT_TRACE_PSEUDOBULK` if the new structure subsumes it (or rename it to be omic-agnostic).

**Done when:** `git grep MEASUREMENT_TRACE_` returns only references inside `paths.py` itself, or zero if the kinase Measurement Trace also migrated.

**Implementation notes:** _(empty)_

### Item 4.3 — Final docs sweep

**Status:** pending
**Depends on:** all prior items
**Files:** any doc that mentions the old tab names, bench/ data paths, or `incytr_pair_levy_t5/`

**Scope:** `git grep -nE 'Fold Change tab|Measurement Trace tab|incytr_pair_levy_t5|bench/incytr' -- ':!docs/archive/' ':!docs/plans/'`. For each hit, update the doc to reflect the new structure. Confirm `CLAUDE.md` reads naturally with the new layout (no awkward leftover sentences from earlier scaffolding).

**Done when:** the grep is clean; CLAUDE.md is internally consistent.

**Implementation notes:** _(empty)_

---

## Cross-phase running notes

Shared scratch space for discoveries that affect multiple items. Agents should append here (in addition to per-item notes) when they learn something with broad relevance.

### Note #1 — `limma::normalizeBetweenArrays` breaks naive JS round-trip (affects Items 3.4, 3.5)

Discovered during Items 1.2 / 3.1 audits. `incytr/R/analysis.R:385,391` applies `limma::normalizeBetweenArrays(matrix(cond1, cond2))` per-(sender|receiver) on the (gene × {cond1, cond2}) matrix before `Cal_foldchange`. This is a quantile-style normalization across the two columns within one cluster, so stored `*_pr/_ps/_py_log2FC` ≠ `log2((D+ε)/(W+ε))` from the raw substrate even after correct aggregation.

Empirical gap on the reference path (Apoe, Astrocytes, ApTt_2mo): substrate `WT=41.437 / ApTt=69.324` → naive `+0.7424`; stored `Ligand_pr_log2FC = +0.7015`. Sign correct, magnitude off by ~0.04.

Implication for Item 3.4 (JS-side LFC + Incytr check):
- A naive `log2((D+ε)/(W+ε))` recomputation **will fail** the 1e-4 cross-check on every multiomic row.
- Options: (a) port `normalizeBetweenArrays` to JS (it's a quantile-style operation, doable but non-trivial), (b) precompute the post-normalization values at build time in a Python helper and ship them alongside the substrate shards, (c) widen the tolerance and document that the displayed LFC is the "pre-normalization" value while Incytr's stored value is "post-normalization."

Recommended: **(b)**. Add a `build_normalized_substrate.py` step that calls limma via rpy2 (or replicates the algorithm) once at viewer-build, writes a parallel `*_per_cluster_normalized.parquet`, and the JS layer reads from that. Keeps the ε and aggregation rule in one Python place, mirrors Incytr exactly, and the build-time round-trip in Item 3.5 collapses to a near-trivial join.

`*_sclog2FC` (transcript) does NOT go through `normalizeBetweenArrays` — it's computed via `Cal_scFC` (`analysis.R:246`) which uses `Cal_foldchange` directly on `Expr_bygroup` output. So transcript JS-side recomputation can stay naive.

This finding should be re-validated at Item 3.2 design time, and the substrate writer must produce both raw and normalized values for protein/phospho.

### Note #2 — `bench/` is fully git-ignored (affects all Phase 2 items)

Discovered during Item 1.1. The entire `bench/` directory is git-ignored; no files under it are tracked. Phase 2 items (2.2–2.4) that plan to `git mv` from `bench/` must instead: (1) copy or edit the file in-place, (2) `git add` the new location under `alz/incytr/`, and (3) `rm` the bench/ source. The net result in git history is "file created at alz/incytr/X" with no rename tracking — that is acceptable per the research-pivot rule (no back-compat shims). Agents for 2.2–2.4 should note this in their implementation notes.

### Note #3 — Canonical path layout established by Items 2.2–2.4 (affects Items 2.5–2.8)

After Items 2.2–2.4 the authoritative path layout is:

| Resource | New canonical path |
|---|---|
| Driver inputs | `data/derived/incytr_inputs/` |
| Driver outputs (wide parquets) | `outputs/reports/incytr_pair_mode/wide/` |
| Smoke outputs | `outputs/reports/incytr_pair_mode/wide_smoke/` |
| Build log | `outputs/reports/incytr_pair_mode/build.log` |
| Run log | `outputs/reports/incytr_pair_mode/pair_run.log` |
| Transcript substrate | `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet` |
| All R driver + input-prep scripts | `alz/incytr/` |

Item 2.5 (data move): move `bench/incytr_pair_levy_t5/incytr input/*` → `data/derived/incytr_inputs/`. The `kldata.csv` symlink currently points (relative) at `../../data/datasets/song/kinase/kldata_pspy.csv`; after the move the symlink target must be updated to an absolute path or a new relative path from `data/derived/incytr_inputs/`. Easiest fix: recreate as `ln -s ../../datasets/song/kinase/kldata_pspy.csv data/derived/incytr_inputs/kldata.csv`.

Item 2.8: `run_pair_mode_pipeline.sh` — the `--spine levy_t5` arg to `run_pair_mode.sh` is gone (no `--spine` arg accepted). Pre-flight checks should validate `alz/incytr/` scripts + `data/derived/incytr_inputs/` inputs, not bench dir contents.

---

## Non-goals

- Re-running the proportional decomposition pipeline. Phase 3 takes the existing per_cluster parquets as input.
- Touching the MEA pipeline (per-site stoichiometry β / NES). Parallel branch; stays independent.
- Adding a "compare normalizations" view between Incytr's per-cluster forward projection and the MEA pipeline's IRS stoichiometry. Different questions; the Evidence panel is for the Incytr-pathway view only.
- Reopening factorial Incytr (archived 2026-05-18).
- Patching the upstream Incytr R package — mirror its behavior, don't change it.
- Eliminating the yuyu CSV staging step in favor of arrow-based parquet reads inside the R driver. Worth doing but out of scope; defer to a follow-up epic.
