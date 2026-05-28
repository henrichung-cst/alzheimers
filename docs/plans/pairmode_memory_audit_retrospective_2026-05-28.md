# Pair-mode memory audit — retrospective

**Date:** 2026-05-28
**Audit doc:** `docs/plans/pairmode_memory_audit_2026-05-27.md`
**Branch:** `perf/pairmode-memory-audit` (both repos)
**Commits:** incytr `50558d0`, `4d1d0f7`, `b8bc9eb`, `52c44f1`; alz `a30427b`, `cd62f9a`

## Question being answered

The audit set out to *"identify memory improvements"* in the pair-mode driver.
Did we find any?

## Honest answer

**Net cumulative effect on `memory.peak`: essentially zero.** Measured at W=2
(direct comparison) under a 14 GB cgroup cap on the same 5-pair
`PAIR_SUBSET`:

| state                            | memory.peak |
|----------------------------------|-------------|
| pre-Step-1                       | ~11.6 GB    |
| Step 1 (in-fork `gc()` removed)  | 11.21 GB    |
| Step 2/4 (fusion + tidy-ups)     | 11.66 GB    |

The audit's *stated expected outcome* — *"a flatter memory profile that
admits a higher fork count, with the wall-time improvement following from
the parallelism the lower footprint unlocks"* — did **not** occur:

- Flatter profile → no (peak essentially flat).
- Higher fork count → no (W=3 was already the production default; the
  Experiment B sweep confirmed it but did not unlock W=4 — a different
  ceiling, the single-heaviest-pair wall, kicks in).
- Wall improvement from unlocked parallelism → no (W=4 doesn't move wall;
  T>1 hurts on this footprint).

## Why each step's memory effect did or didn't survive

| step | predicted mechanism | observed effect |
|---|---|---|
| **1** in-fork `gc()` removal | audit hypothesis: gc() dirtied base pages, multiplying COW per worker by 4–5 GB | -3.6% (~400 MB). Real but small. The audit's COW-multiplication mechanism was disconfirmed — `Permutation_test`'s cached path already nulls `cond_mats` before the boot loop, so gc() had little to scan. |
| **2** fuse enumeration + SigProb cutoff | drop the 1.56M-row materialization, save the transient RSS spike | The 1.56M *Path-string + sort* materialization did drop. But `Cal_foldchange` still has to run on the full 1.56M-element SigProb vectors **pre-cutoff**, because its aFC formula uses `quantile(c(c1,c2), 0.75)` over the population — running it post-cutoff (188k) gives a different quantile, different `th`, different aFC → fidelity break. A *different* full-vector allocation reappeared at the same point in the pipeline, ~450 MB worth. Net peak: +4%. |
| **4** rowsum + paste suffix + qualify `set()` | "~1 s combined, low effort, mostly a tidy-up" — audit was explicit | Below the noise floor on both wall and memory. |

The aFC quantile coupling is the key constraint the audit's design didn't
model. Once Step 2 was held to the strict `max_abs_diff = 0` fidelity gate,
the only design that preserved aFC bit-identical was the one that allocates
the same full-vector transient — just at a different point in the pipeline.
**The memory shape changed; the memory peak did not.**

## What we *did* get out of the audit (be honest about what's real)

1. **A correctness bug found and fixed (`Cal_PDS` row misalignment).**
   `prep_kinase_invariants` hoisted upfront built `@kl.pathways` at 1.56M
   rows; the per-pair tail then filtered `@pathways` to 188k post-cutoff but
   `@kl.pathways` wasn't reduced until inside `Cal_PDS` itself. The
   intervening `apply_condition_direction` used `ifelse` with `base_score`
   (188k) indexing `s4_1`/`s4_2` (1.56M) by position → R silently recycled
   the SiK vectors, mixing unrelated paths' kinase scores into PDS.
   Discovered only because Step 2 forced the kinase-prep lifecycle to be
   examined. Measured downstream impact: 45,224 / 888,805 rows (5.1%)
   PDS-corrected; mean |Δ|=0.04; +59 net paths at the `|PDS|>=0.2`
   significance gate. The audit didn't predict it and didn't aim for it.
   **This is the most important thing we shipped.**

2. **Empirical confirmation that the current parallelism default is right.**
   Experiment B sweep at W ∈ {2,3,4}, T ∈ {1,2}: W=3 T=1 is the frontier,
   matching the driver's existing `NPAIR_WORKERS=3` default. Before, the
   default was set by reasoning; now it's backed by measurement. Same
   setting, more confidence.

3. **A characterized memory wall, with a named bottleneck.** The next
   memory pivot has to happen at the aFC quantile in `Cal_foldchange` —
   `th = quantile(c(c1, c2), 0.75)` over the pre-cutoff population is what
   forces the full-vector allocation Step 2 couldn't eliminate. Decoupling
   it is behavior-changing (aFC values would shift in a defined way), so
   it sits outside the strict-fidelity gate this audit operated under. It
   needs to be reframed as *"we are explicitly choosing different aFC
   semantics, here's why"*, not as a refactor.

## The framing lesson

The audit's framing — *"we'll find a memory win that unlocks more
parallelism"* — assumed memory wins could be found under a strict
`max_abs_diff = 0` fidelity gate. That was wrong. The largest remaining
transient at the audit's start was structurally required by the current
aFC formula. **You can't get the memory without changing the math.**

This is not a failure of execution; it's a finding about the code path.
The right next move is either:

- accept the current memory profile as the floor under bit-identical
  fidelity, or
- frame and validate a behavior-changing proposal for `Cal_foldchange`'s
  aFC semantics.

## Was the audit worth doing?

If "worth" means "did it produce a memory win" — **no, and we should be
plain about that** rather than dressing up the wash as a win.

If "worth" means "did it produce something" — yes:
- the PDS bug fix alone (correctness, not performance) justifies the audit;
- we now know the next memory pivot requires a behavior decision rather
  than another fidelity-preserving refactor;
- we confirmed the parallelism default empirically.

But anyone reading the NEWS or commit log should not come away thinking
the audit reduced the memory footprint of the pair-mode driver. **It did
not.** It changed the shape of where allocations happen, fixed a
correctness bug along the way, and surfaced the next bottleneck for a
separate, behavior-aware decision.
