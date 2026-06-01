# Pair-mode gate switch (drop p_adj → sce4 Top300) + viewer rebuild — 2026-05-30

## What changed and why

The nboot=100 production run finished; `outputs/reports/incytr_pair_mode/wide/`
holds the **unfiltered** superset (3.4 GB, 4.8M–9.7M rows/contrast, max ~127k
rows/pair). Before rebuilding the viewer we replaced the significance gate.

**Decisive finding (2026-05-30):** sce4 never ran the permutation test. None of
its artifacts for the 2mo AppP contrast carry a `p_value`/`p_adj`/`fdr`/`qval`
column — checked across all five artifact types:

| sce4 artifact | shape | inferential column? |
|---|---|---|
| 420 per-pair CSVs | 80 cols | none |
| `Allpathway_table.csv` | 80 cols | none |
| `Top300_table.csv` | 80 cols | none |
| `Pairwise_pathway_table.rds` | 418 pairs × 79 cols | none |
| `Pairwise_kinase_table.rds` | 418 pairs × 9 cols | none |

So the `p_adj < 0.05` "paper gate" had nothing in the reference to match and was
dropping paths sce4 kept (cell-sparse pairs get NA permutation p → the
Microglia→Cholinergic benchmark vanished entirely).

**The faithful sce4 gate (verified byte-for-byte against sce4's own output):**

    SigProb > 0.1 (either condition)  AND  |PDS| >= 0.2
      THEN per (Sender.group, Receiver.group) pair:
        top-300 rows by PDS desc  ∪  top-300 rows by PDS asc

- The two floors reproduce sce4's `Allpathway` exactly (0 rows `|PDS|<0.2`,
  0 rows both SigProb≤0.1).
- The per-pair top-300-up ∪ top-300-down cap reconstructs sce4's `Top300`
  exactly: 65,750 rows / 418 pairs, **0-row symmetric set difference** vs the
  reference `Top300_table`.
- The nboot=100 `p_value_*` columns stay in `wide/` as **informational only**
  (no longer gating). The pair pvalue remains untrustworthy → rank on `|PDS|`.

The cap also keeps the viewer small: floors-only ≈ 300k rows/contrast (sce4) /
≈ up to 438k for our 784-pair spine; with the cap ≈ 66k (sce4) / ~438k→capped
per pair for ours. Probe pair verified: 30,067 → 600 (300 up ∪ 300 down).

## Code already changed (this session)

- `alz/incytr_pair/filter_significant_paths.py` — rewritten: dropped the p_adj /
  BH machinery (`_bh_cutoff`, p_value detection); new gate = two floors + per-pair
  top-300 PDS up ∪ down (`TOP_N=300`). Idempotent, atomic, DuckDB-streamed.
- `alz/incytr_pair/run_pair_mode.sh` — filter call-site comment updated (sce4 gate).
- `alz/incytr_pair/verify_incytr_sce4.sh` — header comment updated.
- `CLAUDE.md` — override #5 rewritten (no p_adj arm; top-300 cap; pvalue informational).

The sce4 parity gate (`verify_incytr_sce4.sh`) is **unchanged in behavior**: it
regenerates the two reference pairs unfiltered (nboot=0) and verifies the engine,
independent of the filter.

## Rebuild steps (NOT yet run — awaiting go)

1. **Parity gate first.** `pixi run verify-incytr-sce4` — must PASS (573/600
   Micro→Cholin, 599/600 Ndnf×Ndnf, max |Δ sclog2FC|=0) before touching `wide/`.
2. **(Decision) snapshot the unfiltered superset?** The filter is in-place on
   `wide/`. Re-tuning `TOP_N` later without a snapshot means a ~5 h re-burn.
   Options: (a) accept loss — sce4 locks TOP_N=300, unlikely to change; (b) copy
   `wide/` → `wide_unfiltered/` first (+3.4 GB on the box). **Recommend (a)** —
   the gate is sce4-locked.
3. **Filter:** `pixi run python alz/incytr_pair/filter_significant_paths.py --dir
   outputs/reports/incytr_pair_mode/wide` (9 files, in place, ~6% kept).
4. **Reshape + rebuild:** `bash alz/runners/main/run_pair_mode_viewer_build.sh`
   — resets `receiver_cache/`, reshapes filtered `wide/` → `receiver_cache/`,
   runs `pixi run viewer`. (`PAIR_MODE_STRICT=1` to enforce all 9 contrasts.)
5. **Verify viewer:** hard-refresh (Ctrl+Shift+R); confirm
   `PAYLOAD.meta.generated_at` is fresh and the Incytr heatmap renders the
   benchmark pairs.

## Cleanup (per docs/plans/outputs_reports_cleanup_2026-05-29.md — now partly stale)

- `wide_nboot0/` is already deleted. The cleanup doc's Tier D ("pvalue
  untrustworthy → wide_nboot0 is production") is **superseded**: production is
  nboot=100 `wide/` with the pvalue kept informational, gated on PDS/top-300.
- After the rebuild verifies, delete `_parity_probe/`, `_sce4_parity/` (regen on
  demand) and any remaining Tier A/B dirs from the cleanup doc.

## Commit (after rebuild verifies)

`refactor(incytr_pair): drop p_adj gate, adopt sce4 Top300 (per-pair top-300 PDS
up ∪ down)` — `filter_significant_paths.py`, `run_pair_mode.sh`,
`verify_incytr_sce4.sh`, `CLAUDE.md`, this plan. (`outputs/` is gitignored.)
