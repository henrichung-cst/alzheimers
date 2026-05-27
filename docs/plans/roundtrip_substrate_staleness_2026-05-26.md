# Round-trip verifier failures after the wide_nboot0 rebuild — audit + fix plan

**Date:** 2026-05-26
**Trigger:** viewer rebuild from filtered `wide_nboot0` aborts in
`assert_pathway_fc_round_trips()` with **7420 round-trip failures** at
`ROUNDTRIP_TOL=1e-4`.
**Status:** fixed (Phases A + B complete); final verifier + viewer rebuild in flight.

## TL;DR

The OOM that killed the 34-min build is fixed (verifier now peaks at 4.3 GB,
was 21.9 GB global-OOM). With it fixed, the verifier does its job and surfaces
a **real, pre-existing substrate problem** — the `audit_sources/` measurement
trace does **not** reproduce the stored `*_log2FC` values. The **shards are
correct** (faithful pass-through of sce4-parity-validated `wide_nboot0`); the
**substrate is stale and, for transcript, computed with the wrong aggregation.**

This was never caught before because the substrate matched the *old* data; the
`wide_nboot0` rebuild (new decomposition + nboot=0 filter) moved the data out
from under a substrate that was never regenerated.

## What is correct

- **Shards** (`edge_slices/incytr_pathways/*.parquet`, built today 11:14) carry
  the `wide_nboot0` FC values verbatim. Verified: `Syt1` Ligand_sclog2FC =
  `0.002182` in both the shard and the source parquet; large-|FC| values match
  `log2(D/W)` (disease/WT) sign, i.e. the "+ = up in disease" convention holds.
- **The verifier** is structurally right to fail — it is detecting genuine
  divergence, not float16 noise (it casts recomputed→float16 before comparing).

## Two independent root causes

### A. Omics substrate is stale (pr/ps/py layers)

- `omics_trace_normalized/` built **2026-05-21**.
- Its protein source `data/derived/incytr_inputs/{pr,ps,py}_yuyu_deconvoluted.csv`
  is **2026-05-25** (post the 2026-05-24 decomposition reconciliation).
- `build_normalized_substrate.py:74` points `WIDE_DIR` at the **old**
  `incytr_pair_mode/wide/` (2026-05-20, nboot=100) — not `wide_nboot0`.
- Result: pr/ps/py layers diverge wholesale — median |Δ| 0.6–2.1, max ~18.

### B. Transcript substrate uses the wrong aggregation (sclog2FC layer)

- `sclog2FC` is **trimean**-based: `grid.R:187` calls
  `Expr_bygroup(mean_method = NULL)` (trimean = 0.25·Q1+0.5·Q2+0.25·Q3) **once**;
  `grid.R:209-210` shows `Cal_SigProb` **and** `Cal_scFC` both read that one
  `expr.bygroup` slot (`analysis.R:266`). Driver passes `mean_method = NULL`
  (`incytr_commandline.R:335`).
- `emit_expr_bygroup.R:64` emits the **arithmetic mean** (`mat %*% indic`),
  and its comment (lines 10-12) wrongly asserts "Cal_scFC bypasses that slot and
  aggregates the raw matrix directly with arithmetic mean." It does not — there
  is a single trimean fill feeding both.
- Empirical proof: `Lama2`/Glutamatergic stored `-7.43` requires a disease-arm
  value ≈0 (trimean of a sparse gene), but the arithmetic mean is 0.268, which
  gives only `-2.43`. The stored value matches trimean, not mean.
- Magnitude of divergence: transcript median |Δ| 0.18, max 5.6.

## Fix plan + status

**Phase A — omics (DONE; was NOT mechanical — two real logic gaps):**
1. ✅ Repoint `build_normalized_substrate.py` `WIDE_DIR` → `INCYTR_PAIR_MODE_INPUT_DIR`
   (default `wide_nboot0`; the env var the viewer build reads). `wide/` superseded.
2. ✅ **Protein floor** — `pmax(.,1)` on raw protein before quantile-normalization,
   mirroring `floor_pr` (pr only; ps/py unfloored). Fixed the ~15-log2 one-sided
   protein outliers.
3. ✅ **Conditional correction** — the substrate's recompute (and the verifier's)
   used a uniform `+ε`; `Cal_foldchange` (math.R) only adds ε when the normalized
   column-pair contains an exact zero, else `log2(c1/c2)` raw. Fixed in
   `_roundtrip_sample` and `verify_pathway_round_trip._recompute_omics_lfc`.
4. ✅ Regenerated `omics_trace_normalized/` — builder self-check **passed** (31
   shards, peak 7.4 GB). The earlier "irreducible R-vs-Python precision" worry was
   wrong — the spurious uniform ε caused the sub-1e-3 failures too.

**Phase B — transcript (in progress):**
5. ✅ `emit_expr_bygroup.R` now emits the **trimean** via Incytr's own kernel
   `grouped_weighted_quartile` (bitwise parity, gene-chunked to bound memory).
6. ✅ Corrected the wrong "arithmetic mean" comments in `emit_expr_bygroup.R` and
   `build_transcript_trace.py`.
7. ✅ Regenerated `transcript_per_cluster.parquet` (trimean) → `transcript_trace/`
   shards. Full verifier: 7420 → **31** failures, all `Ligand.sclog2FC`, all tiny
   (Δ ≤ 2.2e-3). The trimean fixed the systematic divergence.
8. ✅ Resolved the 31: confirmed they are `Cal_scFC`'s zero-conditional ε —
   `log2(d/w)` (no correction) matches stored to ~6e-6, the uniform `+ε` is off by
   exactly Δ. `Cal_foldchange` adds ε only when the pair's sender/receiver gene set
   has a zero-trimean gene; that set is pair-specific and the significance filter
   can drop genes from the shard, so the verifier cannot reconstruct which branch
   fired. Fix: `_recompute_sc_lfc` returns BOTH branches and `_check_shard` accepts
   a match against either (the branches differ by ≤ a few e-3; a real
   routing/sign/drift bug misses both by >>tol). Verified the two known cases
   (Apoe, Ptn) now pass.

**Phase C — rebuild:**
9. Rebuild viewer from `wide_nboot0`; round-trip passes.

**JS:** no change needed — `evidence_row.js` displays the *stored* FC value
(`_lfcChip(stored)`), not a substrate-recomputed one; the dot-bar shows raw
per-animal values for context. The round-trip verifier is the integrity check.

## Why not skip the round-trip

`--skip-roundtrip` only guards the post-build call (line 3048); the gating call
at line 2582 is unconditional. More importantly, the measurement-trace panel
would otherwise show per-group values that do not reproduce the FC tab — a
"honesty over polish" violation. The substrate must be fixed, not bypassed.

## Already done this session (verifier memory fix)

`verify_pathway_round_trip.py` default mode rewritten from "read every shard,
`pd.concat` all 181M rows, then sample" (→21.9 GB global-OOM) to a pyarrow
per-shard bounded hash-reservoir (≤100 rows/contrast). Peak RSS 4.3 GB. (DuckDB
was a dead end — it cannot read the shards' float16 FLBA+BYTE_STREAM_SPLIT
columns; pyarrow can.)
