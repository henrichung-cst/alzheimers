# incytr_pair / integration cleanup audit — 2026-06-08

Purpose: identify non-critical files in `alz/incytr_pair/` and `alz/integration/`
for a cleanup pass. Method: cross-referenced every file against pixi tasks,
runner scripts, code imports, and docs (scoped to `alz/ docs/ conf/ pixi.toml`).

## Verdict

- **`alz/integration/` is clean.** Every file is a live pipeline step, a pixi
  task target, or a documented one-time generator of a frozen prerequisite.
  Nothing to remove.
- **`alz/incytr_pair/` carries a ~2.2k-LOC layer of sce4-reproduction forensic
  probes** that are referenced *only* from the investigation log
  (`docs/plans/sce4_reproduction.md`) and the README forensics section. The
  sce4 reproduction is SOLVED (project memory + CLAUDE.md §pair-mode). These are
  the cleanup surface.

## alz/incytr_pair/ — classification

### Critical — keep (production pipeline + regression gates)

| File | Why |
|---|---|
| `incytr_commandline.R` | Core R driver (`Cal_pairwise_grid`). |
| `run_pair_mode.sh` | Mouse AD 9-contrast driver. |
| `run_pair_mode_tcells.sh` | T-cell driver (pixi `tcells-incytr`). |
| `filter_significant_paths.py` | Downstream significance filter. |
| `build_pair_inputs.sh` | Mouse input orchestrator. |
| `build_pair_seurat.R` | Builds `incytr_obj.rds`. |
| `build_input_gene_list.R` | `allmarkers.csv` (mouse). |
| `build_tcells_seurat.R` | pixi `tcells-build-incytr-seurat`. |
| `build_tcells_input_gene_list.R` | pixi `tcells-build-input-gene-list`. |
| `export_decomposition_for_pair.py` | Provenance deconvolution → `{pr,ps,py}_yuyu_deconvoluted.csv`. |
| `pair_to_receiver_cache.py` | Reshapes wide → `receiver_cache/`; exports `_sanitize_celltype` to 4 integration modules. |
| `emit_expr_bygroup.R` | Transcript substrate; wired into pipeline + `build_transcript_trace.py` + `viewer/paths.py`. |
| `extract_sce4_geneuse.R` | **AD per-pair frozen gene.use** (CLAUDE.md §1b); consumed by `run_pair_mode.sh`, `incytr_commandline.R`, `verify_incytr_sce4.sh`, `verify_sce4_parity.py`. |
| `verify_incytr_sce4.sh` | pixi `verify-incytr-sce4`. |
| `verify_sce4_parity.py` | Called by `verify_incytr_sce4.sh`. |
| `verify_sce4_full.R` | pixi `verify-incytr-sce4-full`. |
| `audit_incytr_input_provenance.py` | Provenance guardrail run *before* production runs (documented in README). Utility, not a probe — keep. |
| `__init__.py` | Package marker. |

### Non-critical — cleanup candidates

**Tier A — sce4 forensic probes (investigation SOLVED; only the log references them).**
One-off scripts written during the sce4 reproduction; each only referenced by
`docs/plans/sce4_reproduction.md` (and three by the README forensics block).

| File | LOC | External refs |
|---|---|---|
| `forensic_sce4_afc.R` | 292 | sce4_reproduction.md |
| `audit_ppds_provenance.R` | 288 | sce4_reproduction.md |
| `audit_pds_score_influence.R` | 275 | README + sce4_reproduction.md |
| `audit_extra_universe.R` | 254 | sce4_reproduction.md |
| `audit_sce4_mismatches.R` | 241 | sce4_reproduction.md |
| `audit_pds_provenance.R` | 240 | sce4_reproduction.md |
| `audit_transgene_excluded_reproduction.R` | 232 | README + sce4_reproduction.md |
| `audit_shared_score_residuals.R` | 183 | sce4_reproduction.md |
| `audit_phospho_engine_trace.R` | 167 | README + sce4_reproduction.md |
| `run_sce4_full_unfiltered.sh` | 56 | sce4_reproduction.md |

**Tier B — orphaned / redundant (no live references at all).**

| File | LOC | Note |
|---|---|---|
| `launch_pair_mode.sh` | 68 | Self-service systemd launcher wrapping `run_pair_mode.sh`. Referenced nowhere; superseded by the canonical `systemd-run` invocation documented in `README.md` §"Running on this box". Redundant. |

**Tier C — standalone QC diagnostic (recent, keep-or-cut judgment call).**

| File | LOC | Note |
|---|---|---|
| `audit_cell_count_pathway_relationship.py` | 161 | Interpretability QC (cell-count sparsity vs pathway burden → CSV+scatter). Added 2026-06-04, referenced nowhere. Not sce4 forensics. Keep if the QC is still wanted; cut if it was a one-look. |

## Removal coupling (anti-shim: update in the same pass)

Removing Tier A/B requires editing in the same commit:
- `alz/incytr_pair/README.md` — drop the inventory rows for
  `audit_pds_score_influence.R`, `audit_phospho_engine_trace.R`,
  `audit_transgene_excluded_reproduction.R` (the only forensics it lists).
- `docs/plans/sce4_reproduction.md` — this is the *investigation log*; its
  references are historical narrative. Leave the prose, but any "run this
  script" instructions pointing at deleted files should be marked as archived.

## Recommendation

Remove **Tier A (10 files, ~2.2k LOC) + Tier B (1 file)**. These are dead weight
from a closed investigation. Decide Tier C separately. `alz/integration/`
untouched.

## Outcome (2026-06-08)

Tier A + B (11 files) moved via `git mv` to
`archive/sce4_reproduction_2026-06-08/` (with a README cataloguing each). Tier C
(`audit_cell_count_pathway_relationship.py`) kept in place — it is a live QC
diagnostic, not sce4 reproduction. `alz/incytr_pair/README.md` inventory updated
(3 forensics rows dropped, archive pointer added). No code references broke —
only `docs/plans/sce4_reproduction.md` (the historical investigation log) still
names the moved scripts. `alz/integration/` untouched.
