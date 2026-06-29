# Theme B5 — Backbone / pathway reduction

**Contract:** `_contracts.md §B5`. **Consumer:** B2 (sankey). **Code:** `alz/incytr_pair/backbone_reduction.py` (the `reduce()` library function), called from the B4 bridge (`alz/cross_reference/kinase_incytr_bridge.py`, song branch). **Consolidation rationale:** `docs/plans/backbone_fold_into_build_2026-06-28.md`.

The backbone reduction is **not a standalone pipeline step**. `#Backbones` is a per-kinase count that requires B4's kinase↔path join, and the R-EM-T recurrence aggregation is a cheap (~10s) cross-contrast `GROUP BY` over the same `wide/` shards — so both run inside the B4 build. B4 emits the per-kinase participation counts (`kinase_participation.csv`); the song branch additionally calls `backbone_reduction.reduce()` to materialize the one table B2 needs.

## Grouping — R-EM-T only

A path's backbone identity is the **Receptor-EM-Target** spine:

    GROUP BY (Sender.group, Receiver.group, Receptor, EM, Target)

This is the only grouping any consumer needs. The complete-path 6-tuple is not a backbone grouping — Target already fans a Receptor-EM spine ~547× (B4's `recep_em_fan.csv` settled this), so keying on the full tuple re-commits the widest-enumeration conflation the reduction exists to undo. `Ligand` is therefore absent from the output. `Receptor`/`EM` are always present; `Target` is in the key but may be NULL where a path has no target (joined NULL-safe via `IS NOT DISTINCT FROM`).

## Reduction semantics

Over each R-EM-T spine's gated occurrences across the 9 contrasts (3 conditions × 3 timepoints):
- **`n_conditions_present`** = distinct conditions (genotypes) the spine appears in (1–3).
- **`n_timepoints_present`** = **max over conditions** of (distinct timepoints in that condition) (1–3). "Backbone" = within-genotype temporal persistence: a spine at 2/4/6mo of App scores 3; a spine scattered one-timepoint-each across genotypes scores `n_timepoints_present=1, n_conditions_present=3`. Matches the contract's "common across all timepoints first, then conditions."
- **representative `PDS`** = the spine's max-`|PDS|` occurrence (signed value retained); tiebreak + B2 coloring.
- **`backbone_rank`** = dense rank over `n_timepoints_present` desc → `n_conditions_present` desc → `|PDS|` desc. Not a composite score (explainable for publication, per contract).
- **`conditions_present`** / **`contrasts_present`** = comma-joined sorted labels, so B2 colors flows without re-reading `wide/`.

**Cholinergic-Neurons anchor:** spines with `Receiver.group == "Cholinergic-Neurons"` are never filtered or reordered out — they carry an `is_cholinergic_target` boolean B2 can pin/highlight. (Anchoring is a B2 display concern; B5 only flags.)

## Output schema (`outputs/reports/incytr_pair_mode/backbone/backbone_rem_t.parquet`)

`Sender.group, Receiver.group, Receptor, EM, Target` + `PDS` (representative, signed) + `n_timepoints_present, n_conditions_present, backbone_rank, is_cholinergic_target, conditions_present, contrasts_present`. One row per R-EM-T key-tuple. The module docstring is the authoritative column reference.

## Canonical floor (idempotent re-application)

`(SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1) AND |PDS| >= 0.2`, re-applied in-query so the reduction is correct whether or not `filter_significant_paths.py` has pre-gated the shards. **Raising the cutoffs is forbidden** (CLAUDE.md). DuckDB-streamed throughout (spill to `DUCKDB_TEMP_DIR`); no whole-file pandas read of `wide/`.

## Verification

`reduce(..., verify=True)` / `verify()`:
- `n_timepoints_present`/`n_conditions_present` ∈ [1,3]; `backbone_rank` dense + gap-free.
- rows ≤ distinct R-EM-T key-tuples in gated `wide/`.
- `is_cholinergic_target` consistent, and Cholinergic-Neurons spines PRESENT (not dropped) — the annotation-not-filter invariant.

Standalone verified: 2,782,293 R-EM-T backbones, all checks pass.

## Out of scope

The sankey itself (B2), any hard specificity/cell-count filter, aggexp, re-gating beyond the canonical floor, viewer/builder edits. Per-spine specificity / cell-count annotations are **not** produced by the reducer (see `_contracts.md §B5` — open divergence flagged there); add them in B2 if the sankey needs them.
