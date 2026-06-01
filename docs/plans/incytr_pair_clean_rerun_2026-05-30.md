# Pair-mode Incytr — clean full re-run to close the coverage gap (2026-05-30)

> **STATUS: PLAN — awaiting approval. No code/run started.**

## Goal

Produce a complete-coverage engine run over the full 31-cluster levy_t5 spine so the
diff against sce4's 6 Top300 references (`docs/plans/sce4_reproduction.md` §2)
has an honest, full denominator — and eliminate the two defects that made the current
`wide/` under-cover: a stale skipped contrast and a silent sparse-cluster drop.

## Root cause (diagnosed 2026-05-30, on-box — CORRECTED after Phase 0)

The current `wide/` (dated 05-30 02:xx) covers only 28–29 of 31 clusters per contrast.
**Phase 0 executed and FALSIFIED the original "nboot=100 permutation-crash" hypothesis.**
The real picture is one stale-skip + one downstream gate:

1. **`ma_2mo_AppP` is stale.** `pair_run.log` line 2:
   `SKIP ma_2mo_AppP vs ma_2mo_WTyp (exists, 2.8G)`. The resumable runner skips a
   contrast whose final parquet exists, with **no check that the inputs are unchanged**.
   The current production inputs (`data/derived/incytr_inputs/incytr_obj.rds`,
   `input_gene_list.csv`) are dated **05-20 and carry all 31 clusters**; the skipped
   `ma_2mo_AppP` parquet predates them. So the benchmark contrast was diffed against a
   pre-rebuild artifact. (Still valid; the provenance-stamp guard fixes this at the source.)

2. **The sparse-cluster drop is a downstream `p_adj` gate, NOT a permutation crash.**
   The driver enumerated and scored **all 961 pairs** (`pair_run.log` line 2821:
   `pair loop done in 27.8 min (961 shards)`; line 2826 wrote the `ma_4mo_AppP` parquet at
   **2467 MB**, with Cholinergic pairs scored at hundreds of thousands of rows each, e.g.
   `pair 939/961 Microglia -> Cholinergic-Neurons rows=129218`). The drop happened *after*
   the driver, in the **`significance filter` step** (`pair_run.log` line 30968).

   The committed `filter_significant_paths.py` that ran at 02:08 was the OLD **"paper
   gate"**: `(SigProb>0.1 either) AND (p_adj<0.05 either — BH on the permutation p_value)
   AND |PDS| >= 0.2`, **with no top-300 cap**. This matches the on-disk state of all 9
   files exactly: `min|PDS| = 0.2000` (floor present), `79k–164k rows/pair` (no cap), and
   the 2–3 sparsest clusters absent (Cholinergic-Neurons, GABAergic-VIP-positive,
   Cortical-2-4-pyramidal) → clean 28²/29².

   **Why those clusters vanish:** at nboot=100 a single-digit-cell pair yields a
   *degenerate* permutation → `p_value` is **NA** (not an error, not an empty shard). The
   old gate's `p_adj<0.05` arm then rejects every NA-p row, removing the whole pair; the
   sparsest clusters are NA on *every* pair, so they drop symmetrically.

   This is precisely the failure CLAUDE.md warns of: "a p_adj gate is foreign to the
   reference and drops paths sce4 kept — cell-sparse pairs get NA permutation p → the
   Microglia→Cholinergic benchmark vanishes."

   **The fix is already partly in the working tree.** `filter_significant_paths.py` has
   since been rewritten to the sce4 gate (`SigProb>0.1 AND |PDS|>=0.2`, then per-pair
   top-300 PDS up ∪ down — **no `p_adj` arm**); it simply was never re-run on `wide/`. So
   the coverage gap is a *stale-filtered-artifact* problem, not an engine problem. nboot is
   irrelevant to scoring (the four `*_sclog2FC`, SigProb, PDS are identical at nboot=0 and
   100 — Phase 0 verified); it only ever fed the now-deleted gate.

## Approach

### Phase 0 — pin the drop mechanism — DONE 2026-05-30 (`bench/perf/phase0_pin_drop.sh`)
Ran `PAIR_SUBSET=Microglia:Cholinergic-Neurons` for `ma_4mo_AppP` at both NBOOT=0 and
NBOOT=100. **Result falsified the hypothesis:**
- **Both** runs produced a non-empty parquet: **99,211 rows**, identical at nboot=0 and 100
  (the engine does not drop the pair; nboot does not gate scoring).
- All four `*_sclog2FC`, SigProb, PDS valid; 44,067 rows pass the sce4 floor
  (`SigProb>0.1 AND |PDS|>=0.2`); `|PDS|` ranges 0→2.08 (unfiltered, no 0.2 floor — proves
  the driver does NOT floor).
- **100% of permutation `p_value` is NA** (both conditions) → degenerate permutation on a
  1–9-cell pair, not a crash, not an empty shard.
- Cross-checked all 9 on-disk `wide/` files: every one has `min|PDS|=0.2000`, no top-300
  cap (79k–164k rows/pair), and 0 Cholinergic rows — the signature of the OLD `p_adj`
  "paper gate", not the working-tree sce4 gate.

**Conclusion:** the drop is the now-deleted `p_adj<0.05` filter arm applied to the sparse,
NA-p pairs — confirmed by code archaeology (`git diff HEAD -- filter_significant_paths.py`
shows the p_adj arm was removed in the working tree). No permutation guard is needed; the
clean re-run + the already-fixed filter recovers every cluster. (Phase-0 option (b),
guarding the permutation, is moot.)

### Phase 1 — clean re-run (no resume)
- **Clear** `outputs/reports/incytr_pair_mode/wide/` and its `.shards/` entirely — no
  stale parquet, no stale shard. (Do NOT rely on the resumable skip.)
- Regenerate all 9 contrasts from the current 05-20/05-28 inputs.
- **nboot = 0** (the recommended setting for this goal): sufficient for parity (the four
  `*_sclog2FC`, SigProb, PDS the diff reads are nboot-independent — Phase 0 verified
  byte-identical at 0 vs 100), and far lighter on memory/time than nboot=100 (~minutes vs
  ~30–44 min per contrast; sidesteps the OOM-prone permutation path documented in
  `run_pair_mode.sh`). Cluster coverage is *independent* of nboot — the sparse clusters are
  scored at either setting; they only vanished via the deleted `p_adj` gate. nboot=0 also
  drops the `p_value_*` columns entirely, which the new gate does not read anyway.

### Phase 2 — honesty guard (no silent drops; provenance stamp)
- The per-contrast concat must emit a **coverage manifest**: pairs scored vs pairs
  dropped, with reason (`0-cells-both-conditions` = genuinely uncoverable vs
  `permutation-degenerate`/error). Surface degeneracy numerically; never let a cluster
  vanish silently (project anti-shim / honesty rule).
- Stamp `wide/` with an **input provenance hash** (mtime/md5 of `incytr_obj.rds` +
  `input_gene_list.csv` + the deconvoluted CSVs). The runner's "skip if parquet exists"
  must re-run when the stamp mismatches — fixes the stale-`ma_2mo_AppP` class of bug at
  the source.

### Phase 3 — validate
- `pixi run verify-incytr-sce4` (benchmark gate must still PASS: Micro→Cholin 572/600,
  Ndnf×Ndnf 599/600, R/E/T `max|Δ|=0`).
- Re-run `bench/perf/diff_engine_vs_sce4.py`. Acceptance:
  - all 6 canonical contrasts cover the full set of clusters that have cells (in-spine
    denominator complete; only genuinely 0-cell-both-conditions pairs excluded, listed);
  - Receptor/Target `max|Δ sclog2FC|` ≈ 0 on non-transgene matched paths;
  - residual divergence is only (a) App/Psen1/Mapt transgene positions and (b) the
    documented per-cluster `gene.use` candidacy gap (never-enumerated) — no new class.

## Risks / open questions
- **Memory**: nboot=0 removes the permutation allocations that drove the prior OOM
  tuning; the existing chunking (`N_CHUNK_MULT`, subprocess-per-chunk) stays as headroom.
- **nboot choice is a pivot, not a toggle** (anti-shim): if we move production `wide/` to
  nboot=0, the `p_value_*` columns go away and downstream consumers must already rank on
  `|PDS|` (they do). Do not keep both paths behind a flag — pick one for `wide/`.
- **Sparse-pair statistics**: a pair with 1 cell/condition yields a one-sided saturated
  fold; its paths are real but low-confidence. The coverage manifest documents this so the
  viewer/diff don't over-read single-cell clusters.

## Rollback
The re-run only rewrites `outputs/reports/incytr_pair_mode/wide/` (a derived artifact);
the prior `wide/` is reproducible and not an input to anything frozen. Phase-2 code
changes (manifest + provenance stamp) are additive to `run_pair_mode.sh` /
`incytr_commandline.R`; revert those files to undo.

## Decisions (locked 2026-05-30)
1. **nboot = 0.** `wide/` moves to nboot=0; the `p_value_*` columns are dropped (anti-shim
   — not kept behind a flag). Downstream already ranks on `|PDS|`. Phase-0 guard (b) is
   therefore not needed for production; Phase 0 stays only as the cheap mechanism check.
2. **All 9 contrasts** re-run from scratch (full production `wide/`).
3. **Wire the input-provenance stamp guard now** — Phase 2's stamp + skip-on-mismatch
   ships in this pass.

## Consequences of the locked choices
- Because nboot=0, the significance filter / viewer paths that referenced `p_value_*`
  must be updated in the same pass (anti-shim: remove, don't disable). Audit
  `filter_significant_paths.py`, the viewer build, and any `p_value`-reading code before
  the run; the sce4 gate is already p_adj-free so the Top300 path is unaffected.
- Expected coverage after re-run: every cluster with ≥1 cell in either condition appears
  (sparse pairs scored one-sided); only true 0-cells-both pairs excluded (none seen so
  far). Pair counts should rise from 784/841 toward 961 minus genuine empties.
