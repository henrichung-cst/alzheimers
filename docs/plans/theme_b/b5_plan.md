# Theme B5 — Backbone / pathway reduction

**Contract:** `_contracts.md §B5`. **Audit:** `b5_audit.md`. **Wave:** 1 (contract producer; disjoint backend — parallel-safe). **Prereq:** the `snrna` step (regenerates specificity + cell counts; h5ad on box). **Consumer:** B2 (sankey, greenfield, written after). **Collision class:** disjoint — new module + new output dir; no viewer/builder edits.

## Decisions (locked, P3 grill 2026-06-25)
- **Specificity = the canonical snrna file** `song_expression_specificity.csv` (same as kinase tab); NO aggexp, no second method. Direct `(gene, cell_type)` join — its `cell_type` is the Levy-t5 spine == `Sender.group`/`Receiver.group`.
- **`min_cell_count`** from `pseudobulk_cell_counts.csv` (spine labels, males_only).
- **`snrna` is a B5 prerequisite**, run at the gate (also restores the kinase tab's currently-NaN specificity).
- **DuckDB-streamed throughout** (4.48M gated rows; no whole-file pandas).
- Greenfield module `alz/incytr_pair/backbone_reduction.py` → `outputs/reports/incytr_pair_mode/backbone/backbone_table.parquet`.

## Reduction semantics — DEFINED HERE (contract was ambiguous; adjust on review)

> **RESOLVED (W1 gate, 2026-06-26) — multi-grouping, not a single key.** The single `full` 6-tuple key is **wrong** and is replaced. B4's `recep_em_fan.csv` settled it with data: across Song's 769 Receptor–EM spines, **Target fans to a mean of 547 (max 586 ≈ all) targets** and Ligand to 4.68 — so the `full` 4-tuple explodes one spine into a mean **2,696×** distinct rows (worst: `Sort1→Ctnnb1`, 9,945), exactly the "widest enumeration" conflation B5 exists to undo. Target carries ~zero discriminating value; it must not be in any identity. **Decision:** emit **three groupings** — `R-EM` (769), `L-R-EM` (~3.6k), `R-EM-T` (~440k) — as a `grouping` dimension (one stacked table, `grouping` column), with `backbone_rank` computed *within* each grouping. The **complete-path** view is NOT a backbone grouping (2.07M rows/cohort — not a browser payload); it stays in the existing Incytr-pathways tab. The viewer **grouping selector** is tail work (B2/B4.2 consume the table); this plan makes the data carry `grouping` so the tail can filter. B4's `n_backbones` reports per-grouping counts accordingly.

Path identity = the active **grouping key** (`R-EM` | `L-R-EM` | `R-EM-T`), carried in a `grouping` column. For each (grouping, key-tuple) group, over its gated occurrences across the 9 contrasts (= 3 conditions × 3 timepoints):
- **`n_conditions_present`** = number of distinct conditions (genotypes) the path appears in (1–3).
- **`n_timepoints_present`** = **max over conditions** of (distinct timepoints present in that condition) (1–3). Rationale: "backbone" = *within-genotype temporal persistence* — a path at 2/4/6mo of App is a stable App backbone (→3); this is not gameable by a path scattered one-timepoint-each across genotypes (that scores `n_timepoints_present=1, n_conditions_present=3`). Matches the contract's "common across all timepoints **first**, then conditions."
- **representative `PDS`** = the path's max-`|PDS|` occurrence (signed value retained); used for the tiebreak and for B2 coloring.

**`backbone_rank`** = dense rank over lexicographic sort: `n_timepoints_present` desc → `n_conditions_present` desc → `|PDS|` desc. Not a composite score (explainable for publication, per contract).

**Cholinergic.Neurons anchor:** paths with `Receiver.group == "Cholinergic.Neurons"` are not filtered or reordered out — they surface via an `is_cholinergic_target` boolean column B2 can pin/highlight. (Anchoring is a B2 display concern; B5 only flags.)

## Annotations — NEVER drop rows
- **`mean_gene_specificity`** = mean of `specificity_score` over the 4 position genes present (NaN positions — EM/Target can be empty — ignored): Ligand looked up in `Sender.group`; Receptor/EM/Target in `Receiver.group`. Join key `(gene_symbol upper, cell_type)`.
- **`min_cell_count`** = min over {`Sender.group`, `Receiver.group`} of that cluster's n_cells (min across males `ma_*` samples — worst-case sparsity, surfaces the Cholinergic 1-cell case honestly).

Both inform `backbone_rank` softly via B2's interactive filters; **neither gates the table.**

## Output schema (`backbone_table.parquet`, documented in module docstring)
`grouping` (`R-EM`|`L-R-EM`|`R-EM-T`) + the grouping's key genes (`Sender.group, Receiver.group` always; `Ligand`/`Target` present only for groupings that include them) + `PDS` (representative, signed) + `n_timepoints_present, n_conditions_present, backbone_rank` (ranked *within* `grouping`)`, mean_gene_specificity, min_cell_count, is_cholinergic_target`. One row per (`grouping`, key-tuple). Also carry `conditions_present` (e.g. `"App,Tau"`) and `contrasts_present` (comma list) so B2 can color flows by which genotype/timepoint cells fired without re-reading `wide/`.

---

## Stages
**Stage 0 — Prereq gate (run, don't code):** `pixi run snrna` (or the snrna step) under `systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0`. Verify `song_expression_specificity.csv` + `pseudobulk_cell_counts.csv` exist and are non-empty; spine `cell_type` values ⊇ the `Sender.group`/`Receiver.group` set in `wide/`.

**Stage 1 — Reduction query (`backbone_reduction.py`, DuckDB):**
- `read_parquet('outputs/reports/incytr_pair_mode/wide/*.parquet', union_by_name=true, filename=true)`; derive `contrast`/`condition`/`timepoint` from `filename` via the audit regex (a DuckDB `regexp_extract` or a per-file UNION injecting the literal, mirroring `song.py:904`).
- Re-apply the canonical floor in-query (idempotent; per-file `SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1` AND `abs(PDS) >= 0.2`) so B5 is correct whether or not `wide/` is pre-gated. Do NOT raise cutoffs (CLAUDE.md).
- `GROUP BY` the 6-tuple → `count(distinct condition)`, the per-condition `count(distinct timepoint)` → max, `max(abs(PDS))` (carry its signed PDS), `string_agg(distinct condition)`, `string_agg(distinct contrast)`, `bool_or(Receiver.group='Cholinergic.Neurons')`.
- Compute `backbone_rank` (window `row_number`/`dense_rank` over the lexicographic order).

**Stage 2 — Annotations (bounded reads):**
- Extract the distinct position-gene set from the reduced table; join `mean_gene_specificity` from `song_expression_specificity.csv` (small long-form — read fully is fine) keyed `(gene_upper, cell_type)`; mean over the ≤4 present genes.
- Join `min_cell_count` from `pseudobulk_cell_counts.csv` (males only); min over the two clusters.
- Write `backbone_table.parquet`.

**Stage 3 — Runner + task:** add the step to `run_pair_mode_pipeline.sh` (after the 9-contrast run, before viewer rebuild) and a `pixi` task `incytr-backbone`.

## Verification
- Row-count sanity: `backbone_table` rows ≤ distinct 6-tuples in `wide/`; `n_timepoints_present`/`n_conditions_present` ∈ [1,3]; `backbone_rank` dense and gap-free.
- Spot-check one known recurrent path: appears in the expected contrasts, `n_*` match, `mean_gene_specificity` equals the kinase-tab `song_specificity` for the same (gene, cluster) (proves single-method consistency).
- Annotation honesty: confirm a low-specificity / low-`min_cell_count` Cholinergic path is PRESENT (not dropped) and merely flagged — the whole point of annotation-not-filter.
- Memory: the DuckDB step runs under the cap; peak RSS reported; no pandas whole-file read of `wide/` or aggexp.

## Out of scope
The sankey itself (B2), any hard specificity/cell-count filter, aggexp, re-gating beyond the canonical floor, viewer/builder edits.
