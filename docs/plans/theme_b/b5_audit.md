# B5 Audit — Backbone / pathway reduction

Recon for the reduction+ranking layer over the gated Incytr pair-mode rows, producing the R-EM-T backbone table B2's sankey consumes. Contract: `_contracts.md §B5`. Plan: `b5_plan.md`. Consolidation: `docs/plans/backbone_fold_into_build_2026-06-28.md`.

## Input — gated pathway rows
`outputs/reports/incytr_pair_mode/wide/` — 9 parquets, one per contrast. The driver emits all paths (unfiltered); the canonical floor (`SigProb > 0.1` either condition AND `|PDS| >= 0.2`) is re-applied in-query. **Must be read DuckDB-streamed.**

Schema (DuckDB DESCRIBE): `Sender.group, Receiver.group, Ligand, Receptor, EM, Target` (the 6-tuple identity), `PDS`, two contrast-specific `SigProb_<cond>` cols (e.g. `SigProb_ma_2mo_AppP`, `SigProb_ma_2mo_WTyp`), plus per-channel `sclog2FC`/`pr`/`ps`/`py` columns. **No `contrast` column** — it is filename-encoded. Distinct gated R-EM-T spines = **2,782,293** (the output row count).

## Contrast → (condition, timepoint)
Filename `ma_<age>mo_<Geno>_ma_<age>mo_WTyp_incytr_output.parquet` → regex `ma_(\d+)mo_([A-Za-z]+)_ma_\1mo_WTyp` → normalize geno (`AppP→App, Ttau→Tau, ApTt→ApTt`) → contrast `f"{geno}_{age}mo"`. **condition** = geno ∈ `[App,Tau,ApTt]`; **timepoint** = `2mo|4mo|6mo`. `backbone_reduction.py:_parse_filename` injects `(contrast, condition, timepoint)` per file when unioning.

## Output — R-EM-T backbone table
`outputs/reports/incytr_pair_mode/backbone/backbone_rem_t.parquet`. Schema in `b5_plan.md` / the module docstring. The reducer carries **no** per-spine specificity or cell-count annotation columns (the `snrna`-derived annotation stage was not built into the reducer — see `_contracts.md §B5` open divergence). The `snrna` step is still run at the wave gate to restore the kinase tab's specificity, but it is not a B5 input.

## Consumer — B2 (greenfield)
No sankey/chord code exists yet. B2 ranks/filters off `backbone_rank` + the recurrence columns and must not re-implement its own reduction.

## Module placement
`alz/incytr_pair/backbone_reduction.py` — `reduce()` is a library function called from the B4 bridge's song branch. **No standalone pixi task and no separate runner step** (folded into `kinase-incytr-bridge`). The complete-path view stays in the existing Incytr-pathways tab.
