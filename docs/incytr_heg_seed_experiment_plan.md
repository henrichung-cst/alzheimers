# Incytr seed-list HEG augmentation experiment

## Goal

Add an **HEG (highly expressed genes)** step to `alz/integration/compute_seed_lists.R`
so the Incytr candidate-gene pool matches the upstream paper's design
(`HEG ∪ DEG ∪ DEP`) rather than our current `DEG ∪ prG`-only restriction. Then
rerun a small slice of the factorial pipeline to quantify the effect on the
pathway universe and PDS distribution before committing to a full rerun.

## Why

Our current seed-list recipe omits HEGs entirely. The paper
(`docs/incytr_paper.pdf`, Methods + Figure 2D legend) treats `HEG ∪ DEG ∪ DEP`
as the standard input-gene pool. The shipped upstream fixture
(`data/incytr/v1_8clusters/incytr input/input_gene_list.csv`) carries
1,278–4,104 genes per cluster (avg ~2,800). Our DEG-only pool is **138–527
genes per cluster** — 5–10× smaller — so we silently exclude pathways routed
through genes that are highly expressed but not differentially expressed.
Those are exactly the genes that sit in the Hill-function saturation regime
and contribute strongly to signaling probability.

Reference investigations: the ApTt-late TPDS-collapse triage in
session-context (May 2026); seed-list label design notes in
`docs/integrations/seed_list_labels.md`.

**Not expected to fix**: the ApTt_4mo / ApTt_6mo TPDS collapse (that is a
contrast-geometry problem at n=1 per ApTt cell, not a seed-list problem).
**Expected**: more pathways in the universe, a different DEG/HEG/prG label
mix, and possibly new high-|PDS| pathways routing through stably-expressed
signaling proteins.

## Inputs

All paths relative to repo root:
- `data/incytr_factorial_inputs/pseudobulk_counts.mtx` — pseudobulk counts
  (genes × pseudosamples)
- `data/incytr_factorial_inputs/pseudobulk_metadata.csv` — pseudosample → (celltype, animal, genotype, timepoint, n_cells)
- `data/incytr_factorial_inputs/pseudobulk_genes.csv` / `pseudobulk_pseudosamples.csv`
- `data/incytr_factorial_inputs/expression_genes.csv` — vocabulary (must be a superset of every gene we emit)
- `data/incytr_factorial_inputs/deg_lists.json` — existing per-celltype DEG sets (do not overwrite)
- `data/incytr_factorial_inputs/prg_list.csv` — existing prG set (do not overwrite)

Pixi env is auto-activated by direnv on `cd`. R deps (`DESeq2`, `limma`,
`Matrix`, `jsonlite`) are already pinned for `compute_seed_lists.R`.

## Step 1 — Add an HEG helper to `alz/integration/compute_seed_lists.R`

Insert a new function `compute_heg_lists()` and wire it into `main()` between
the DESeq2 call and the limma call. Recipe:

1. For each cell type `ct`:
   1. Subset `pb$meta` to that cell type, drop pseudosamples with
      `n_cells < min_cells` (reuse the existing `--min-cells` arg, default 10).
   2. Sum counts across the surviving pseudosamples per gene → cluster-level
      count vector.
   3. Normalize to CP10K and apply `log1p` → cluster-level log-expression
      vector.
   4. Rank genes by descending log-expression; take the **top
      `--heg-top-n`** (default **1500**, see "Knob choices" below).
2. Intersect with `vocab_set` (read from `expression_genes.csv`) — drop any
   gene not in the expression vocabulary.
3. Return `heg_lists` as a named list keyed by cell type, same shape as
   `deg_lists`.

Pass-through arguments to add to `parse_args()`:
- `--heg-top-n <int>` (default 1500)
- `--skip-heg` (bool; default FALSE) so the current behavior is recoverable
  for A/B comparisons

## Step 2 — Emit a new artifact and update MANIFEST

Write `data/incytr_factorial_inputs/heg_lists.json` (same JSON-of-named-lists
shape as `deg_lists.json`). **Do not modify `deg_lists.json` or
`prg_list.csv`.**

Append a `heg_lists` block to the `MANIFEST.json` seed_lists section with:
```json
{
  "heg_method": "top-N by cluster-level log1p(CP10K) pseudobulk expression",
  "heg_top_n": 1500,
  "heg_min_cells_per_animal_celltype": 10,
  "heg_partition_note": "Label precedence: DEG > prG > HEG. Applied upstream at label-assignment time."
}
```

## Step 3 — Update `alz/integration/load.R` to expose HEGs

In `load_seed_lists()`:
1. Read `heg_lists.json` when present (mirror the existing `deg_lists.json` handling).
2. Add `heg_lists = heg_lists` to the returned list.
3. Leave loader silent (no `stop()`) when the file is absent, so old fixtures keep working.

## Step 4 — Wire label precedence into the upstream call

`construct_factorial_paths()` in `~/Projects/work/incytr/R/factorial.R` is the
label-assignment site. Confirm with the upstream code path used in
`alz/integration/factorial.R`. Required label precedence: **DEG > prG > HEG**
(i.e. a gene that is both DEG and HEG is labeled DEG; a gene that is HEG but
not DEG or prG is labeled HEG). This must be applied at label-assignment time
in the upstream package — not in our exporter.

If upstream `construct_factorial_paths` does not yet accept an `heg_lists`
argument, file a one-line task to the upstream author (Henri/Anthony) before
proceeding — do not silently fall through. Until upstream supports HEGs, the
new `heg_lists.json` file should be present but unused, and our smoke-test
run will exercise only the loader plumbing.

## Step 5 — Smoke-test rerun (one receiver)

Pick **`Astrocytes`** as the smoke-test receiver (well-populated, not on the
ApTt-collapse pathology). From repo root:

```bash
pixi run compute-seed-lists -- --heg-top-n 1500
pixi run export-factorial-inputs            # regenerate MANIFEST + load HEG-aware fixture
# Targeted single-receiver run (use existing receiver-sweep machinery):
bash alz/runners/main/run_phase4_receiver_sweep.sh --receivers Astrocytes \
     --output-dir outputs/reports/incytr_factorial_heg_smoke
```

If the runner does not accept a `--receivers` flag yet, run
`alz/integration/factorial.R` directly on the existing inputs, restricted to
the Astrocytes receiver (set `RECEIVER=Astrocytes` env var or whatever
filter the script supports).

## Step 6 — Compare against the current run

Diagnostic script (drop into `alz/integration/diagnostics/compare_heg_smoke.py`,
or run as a one-off pixi-python snippet):

```python
import duckdb, pandas as pd
con = duckdb.connect()
con.execute("""CREATE VIEW base AS SELECT * FROM
  read_parquet('outputs/reports/incytr_factorial_5xfad_kldata/receiver_cache/receiver=Astrocytes/*.parquet',
               hive_partitioning=true)""")
con.execute("""CREATE VIEW heg AS SELECT * FROM
  read_parquet('outputs/reports/incytr_factorial_heg_smoke/receiver_cache/receiver=Astrocytes/*.parquet',
               hive_partitioning=true)""")

for label, view in [('base', 'base'), ('heg', 'heg')]:
    print(f"\n=== {label} ===")
    print(con.execute(f"""
      SELECT contrast, COUNT(*) AS n_paths,
             COUNT(*) FILTER (WHERE abs(PDS)>=0.5) AS n_high_pds,
             round(avg(abs(PDS)),3) AS mean_abs_pds,
             round(avg(abs(TPDS)),3) AS mean_abs_tpds
      FROM {view} GROUP BY contrast ORDER BY contrast""").df().to_string(index=False))

    print(con.execute(f"""
      SELECT "Ligand.label", "Receptor.label", "EM.label", "Target.label", COUNT(*) AS n
      FROM {view} WHERE contrast='ApTt_2mo'
      GROUP BY 1,2,3,4 ORDER BY n DESC LIMIT 10""").df().to_string(index=False))
```

## Step 7 — Decision criteria

Proceed to a full rerun (all 19 receivers, both Incytr-factorial and dependent
downstream stages) if **all three** of these hold for the Astrocytes smoke
test:

1. **Pathway universe grew meaningfully**: HEG run has ≥ 1.5× the `n_paths`
   of the base run for Astrocytes.
2. **High-|PDS| pathway count is at least non-decreasing** (≥ 95% of base) for
   non-ApTt-late contrasts. If HEG halves the high-|PDS| count, something is
   wrong with the label precedence and we should not proceed.
3. **Label distribution looks plausible**: HEG-labeled pathways appear as a
   meaningful minority (≥ 10% but ≤ 60% of the universe). Both 0% and 90%+
   indicate plumbing problems.

If any criterion fails, stop and post the diagnostic table for review before
spending compute on a full rerun.

## Knob choices to leave default-but-tunable

- `--heg-top-n` default **1500** — sits in the middle of the upstream
  fixture's per-cluster range (1,278–4,104). Lower bound stays in the
  cluster-specificity regime; higher would pull in too much noise.
- HEG cell-type filter — reuse `--min-cells 10` from the existing recipe.
- Pseudobulk normalization for HEG ranking — CP10K + log1p, sum across all
  surviving pseudosamples in the cluster (no per-condition splitting). HEGs
  are condition-agnostic by design; condition information enters via DEG and
  prG.

## Out of scope

- Changing DEG / prG construction (`raw_p<0.05 AND |LFC|>=0.5`, marginalized
  over timepoint). Keep as-is.
- Changing label precedence beyond `DEG > prG > HEG`.
- Re-running anything beyond Astrocytes smoke until Step 7 criteria pass.
- Investigating the ApTt-late TPDS collapse — that is tracked separately and
  is not a seed-list problem.

## Acceptance

A reviewer reading this plan should be able to:
1. Implement `compute_heg_lists()` in `compute_seed_lists.R` from the
   recipe in Step 1 without further questions.
2. Confirm before running that upstream `construct_factorial_paths` accepts
   an HEG argument (Step 4 gate).
3. Run the smoke test (Step 5), compute the comparison (Step 6), and
   decide go/no-go using the Step 7 criteria — all without needing further
   context from the original investigators.
