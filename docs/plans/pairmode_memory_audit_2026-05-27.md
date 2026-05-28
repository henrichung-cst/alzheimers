# Pair-mode memory audit: memory-per-worker as the parallelization gate (2026-05-27)

## Thesis

After the 2026-05 performance epic, **compute is near its ceiling and memory is the
binding constraint.** The permutation kernel is native C++/OpenMP and
bandwidth-bound; the per-pair scoring tail is vectorized R over already-C-backed
libraries. There is no single hotspot left to crush. What remains is a *memory*
problem wearing a parallelism costume: the full 9-contrast run is gated not by how
fast one pair scores, but by **how many pairs can run concurrently before the 30 GB
box OOMs.** That ceiling is set by per-worker resident memory, so the parallelization
question is downstream of the memory question — they are one investigation, not two.

This is an **audit** (per workflow: audit → approve → implement). It enumerates the
memory sources with what is measured vs. what needs an A/B, defines the two
experiments that resolve the open questions, and proposes a fidelity-gated
remediation sequence. **No driver/package code is changed until this is approved.**

> Precondition note: the 2026-05-27 `bench/` reorg moved the profiling scripts into
> `bench/perf/` but left their `REPO_ROOT="$(dirname …)/.."` pointing at `bench/`
> instead of the repo root. Five scripts (`profile_pair_one.sh`, `phase1_run.sh`,
> `parallel_sweep.sh`, `run_nboot0_w3.sh`, `launch_nboot0_w3.sh`) were repointed to
> `git rev-parse --show-toplevel` (the convention the gate scripts already use). The
> measurements below were taken with the fixed scripts.

## What the profiling established (grounding, measured 2026-05-27)

Box: **16 cores, 30 GB RAM** (~15 GB available at profile time). Profiles taken with
`bench/perf/profile_pair_one.sh` (cgroup-capped, single 375 MB object load, one pair),
on the heaviest pair class **Foxp2-Excitatory → Glutamatergic-excitatory** in the
`ma_2mo_AppP` vs `ma_2mo_WTyp` contrast.

| Fact | Value |
|---|---|
| Paths enumerated (heavy pair) | **1,558,050** |
| Paths surviving `cutoff_SigProb = 0` | **187,946** (12% — 88% discarded) |
| Per-pair wall, serial perm | 10.88 s |
| Per-pair wall, 8 perm threads | 10.32 s |
| RSS after `io_load` (base) | ~4.1 GB |
| Per-pair peak HWM over base (single process) | ~1.3 GB |

Per-pair stage breakdown (production config: `expr_bygroup` injected, so
`Expr_bygroup` ≈ 0):

| stage | serial | % | nature |
|---|---|---|---|
| Permutation_test (nb=100) | 4.15 s | 38% | native C++/OpenMP, **bandwidth-bound** |
| pathway_inference | 2.34 s | 22% | data.table join, materializes all 1.56M rows |
| Export_results | 1.51 s | 14% | wide cbind + `paste()` ID cols over 188k rows |
| Integr_kinasedata | 1.03 s | 9% | `tapply` sum over paths |
| Cal_SigProb | 0.77 s | 7% | SigProb over all 1.56M, then cutoff → 188k |
| rest | ~0.3 s | 3% | vectorized, negligible |

**OpenMP is genuinely active** (`libgomp.so.1` linked; `SHLIB_OPENMP_CXXFLAGS` in
`src/Makevars`). 8 threads moved permutation 4.15 s → 3.22 s — **1.3×, not linear.**
The kernel saturates memory bandwidth (188k pathways × 4 random gene-value gathers per
bootstrap) long before it saturates cores. **Speed-only parallelism of the permutation
is a weak lever.** Its value is that OpenMP threads share the fork's address space —
unlike pair-forks, which each pay a COW tax.

Carried forward from `pairmode_perf_oom_2026-05-25.md` (still valid):
- Expression matrix `object@data`: 30,567 × 29,542 sparse, **0.74 GB**.
- `rm(Data.input)` before the loop already lands; base is ~3 GB post-rm.
- W=3 pair-forks peak **13–15 GB**, ~2.77× speedup, zero OOM under a 24 GB cap.
- W=4 worst-case heavy-pair alignment projects past the 0.7×24 GB headroom → not run.

## Memory taxonomy: necessary vs. design vs. unknown

Per-worker resident memory under `mclapply` decomposes into three sources. They are
**not equal**, and two of three are already classifiable:

### 1. COW-inherited base (~3 GB) — *necessary IF kept read-only*
`template@data` (0.74 GB sparse) + `expr_substrate` (gene_union × clusters × 2
conditions) + the sliced omics frames (`pr/ps/py_{1,2}`) + `dg_by_cluster` + R base.
Under copy-on-write this is **shared, not multiplied** — three forks should cost ~3 GB
total for the base, not 9 GB. It only becomes per-worker cost if something dirties its
pages (see source 2).

### 2. Fork-`gc()` breaking COW — **design (recoverable), highest-leverage**
`process_pair()` calls `gc()`; `Permutation_test()` calls `gc()` twice. R's garbage
collector writes to **every object header** on a full pass, which dirties the
COW-shared base page-by-page and forces a private copy into each child. The driver
itself documents the consequence: forks "carry ~4–5 GB private." So a base that *could*
be shared at ~3 GB total becomes ~3–4 GB **per worker** — this is the single biggest
multiplier on the fork ceiling, and it is a deliberate-but-unverified tradeoff
(reclaim transient allocations vs. preserve COW). **Decidable by A/B (Experiment A).**

### 3. Per-pair transient — *mixed*
- **Design (recoverable):** enumeration materializes the full **1.56M-row** table
  (with a `paste()` Path string for all 1.56M) when only 188k survive the very next
  step. This is both a time cost (≈3 s across `pathway_inference` + `Cal_SigProb`) and
  a transient RSS spike. Fusing enumeration + SigProb + cutoff so zero-SigProb paths
  drop *before* materialization removes it. SigProb needs only the four nodes' trimean
  expression, already in the injected substrate.
- **Necessary:** the dense submatrix the permutation kernel extracts
  (`geneuse × cells_in_condition`). The C++ quartile kernel cannot run on the sparse
  `dgCMatrix`; this densification is real and bounded by the contrast's cell count
  (2mo = 4,858 of 29,542 cells, 16.4%).
- **Design:** `Export_results` builds the full wide table (~40 cols × 188k rows) plus
  two `paste()` ID columns; the per-pair parquet write buffers it again.

**Conclusion of the taxonomy:** the "necessary or poor design?" question is *not*
undetermined. Source 1 is necessary-if-shared; source 2 is the dominant recoverable
design cost and is A/B-decidable; source 3 is half recoverable (enumeration, export)
and half necessary (dense kernel input). The keystone is **source 2**: if dropping
fork-`gc()` preserves COW, per-worker footprint falls and the fork ceiling rises with
no algorithmic change.

## The two open questions → two experiments

### Experiment A — fork-`gc()` A/B (resolves "necessary vs. design")
**Question:** does in-fork `gc()` net-reduce or net-increase per-worker peak RSS?

**Method:** on the two sce4 benchmark pairs + 3 representative heavy pairs, run the
driver at fixed W=2 under a cgroup cap, measuring per-worker peak RSS and total
`cgroup memory.peak`, in two arms:
- **arm 1 (current):** `gc()` in `process_pair` and `Permutation_test` as-is.
- **arm 2:** in-fork `gc()` removed (rely on fork teardown to reclaim; the parent
  still `gc()`s between scheduling waves).

**Decision rule:** if arm 2's `memory.peak` is lower (COW preserved beats transient
reclamation), remove in-fork `gc()` and re-measure the fork ceiling. If arm 1 wins
(transients dominate), keep `gc()` but scope it tighter (only the largest transients,
not a full collection). Either way the answer is a measured number, not a guess.

**Cost:** ~30 min of capped runs. No code shipped until the number is in.

### Experiment B — the forks × threads frontier (resolves "what level is efficient")
**Question:** the efficient operating point is not a scalar `W` — it is a 2-D
allocation `(pair_forks × perm_threads)` bounded by **RAM on one axis and
cores/bandwidth on the other.**

**Method:** sweep with `bench/perf/parallel_sweep.sh` + `assert_core_budget()` over a
small grid on a fixed representative pair-set, recording wall time and `memory.peak`
per cell:

| | threads=1 | threads=2 | threads=4 |
|---|---|---|---|
| **forks=1** | baseline | | |
| **forks=2** | | | |
| **forks=3** | current | | |
| **forks=4** | (OOM-risk) | | |

Constraints from the grounding: forks are memory-expensive (each ~3–5 GB depending on
Experiment A); threads are memory-cheap but yield only ~1.3× (bandwidth). On 16 cores
the core budget is generous; **RAM is the wall.** Run every cell under the cgroup cap;
a cell that breaches is OOM-killed inside its scope only.

**Decision rule:** pick the `(forks, threads)` with lowest wall time whose
`memory.peak` stays under 24 GB with margin. Expect the frontier to favor *more forks*
if Experiment A frees COW, and *more threads* if it does not.

**Coupling (the key point):** Experiments A and B are sequential, not parallel.
Whatever A shaves off per-worker memory directly raises the optimal fork count in B.
**Do A first; B's grid is meaningless until the footprint is fixed.**

## Proposed remediation sequence (post-experiment, fidelity-gated)

Each step ships only after `bench/fidelity/compare_pair_outputs.R` confirms
`max_abs_diff = 0` against the frozen reference. Ordering is by leverage:

1. **Fix source 2 per Experiment A** (drop or tighten fork-`gc()`). Pure memory win;
   raises the fork ceiling. Lowest risk — `gc()` placement cannot change a computed
   value.
2. **Fuse enumeration + SigProb + cutoff** (source 3, design half). Stops materializing
   the 1.56M-row table; cuts ~3 s/heavy-pair *and* the transient RSS spike. Moderate
   effort, stays in data.table/C++.
3. **Re-run Experiment B** to set the production `(forks, threads)` config now that the
   footprint is smaller.
4. **Trim Export_results + kinase** (`paste()` ID cols, `tapply` → data.table keyed
   sum). ~1 s combined, low effort, mostly a tidy-up.

Expected outcome: a **flatter memory profile that admits a higher fork count**, with
the wall-time improvement following from the parallelism the lower footprint unlocks —
not from chasing the bandwidth-bound kernel.

## Explicitly out of scope (recorded so it is not re-litigated)

- **Rust rewrite.** The hot kernel is already native and at a hardware-bandwidth
  ceiling; Rust hits the same wall and would forfeit the bitwise sce4 parity the
  `bench/` gate exists to protect. Not the next step.
- **DuckDB-ifying the deterministic scoring path.** A genuine high-ceiling
  architectural move (whole-grid scoring in the columnar engine already on the
  dependency list, eliminating the 961×/contrast re-enumeration and the fork model
  entirely). Deferred: it is a redesign, not a memory fix, and should be evaluated only
  after the cheap memory wins above are measured.

## OOM safety protocol (applies to every experiment)

1. Every run launches inside `systemd-run --user --scope --slice=alz-incytr.slice
   -p MemoryMax=<cap> -p MemorySwapMax=0` (the `profile_pair_one.sh` / README pattern).
   The cap is the wall; a breach kills inside the scope only — it cannot take the host
   or Claude down.
2. Box is 30 GB. **Never cap above 24 GB** (6 GB reserved for host + Claude).
3. Scale parallelism incrementally and empirically: read `memory.peak` /
   `memory.events` after each cell before escalating.
4. One 375 MB object load per process; never load per-pair (per
   `feedback_memory_safety_shared_box`).

## Fidelity gate (non-negotiable)

Every code change is verified with `bench/fidelity/validate_derived_parity.sh` (or the
cheaper 2-pair `compare_pair_outputs.R` against
`bench/fidelity/parity_frozen_out/`) for `max_abs_diff = 0` across all columns,
p-values included. A change that perturbs any output is rejected regardless of its
speed/memory benefit. See `bench/bench.md` §2 for the six locked sce4-parity overrides.

## Outcome (2026-05-28)

Steps 1, 2, and 4 landed (incytr commits `50558d0`, `4d1d0f7`, `b8bc9eb`; alz
commit `a30427b`). Re-measurement (Task #11) and Experiment B (Task #12) ran
on a 30 GB shared box under `systemd-run --user --scope` with
`MemorySwapMax=0`, using the same 5-pair `PAIR_SUBSET` (2 sce4 benchmark pairs
+ 3 heaviest by surviving rows), NBOOT=100, `bench/perf/post_step24_remeasure.sh`
and `bench/perf/exp_b_cell.sh`.

**Honest memory accounting.** Cumulative cgroup `memory.peak` (W=2 reference
cell) is essentially unchanged: pre-Step-1 ~11.6 GB → Step 1 (gc off) 11.21 GB
→ Step 2/4 11.66 GB. The Step 1 gc-off win (~3.6%) was consumed by Step 2's
full-vector `Cal_foldchange` (required pre-cutoff to preserve the aFC
quantile). The real Step 2/4 deliverables are the **Cal_PDS row-alignment bug
fix** (5.1% of PDS rows corrected; +59 paths at the `|PDS|>=0.2` gate) and the
`@pathways` lifecycle cleanup. NEWS.md was amended (incytr commit `52c44f1`)
to reflect that the originally-claimed ~2× per-pair wall speedup was a
NBOOT=2 fidelity-mode measurement; at NBOOT=100 permutations dominate and the
wall is within noise.

**Experiment B frontier (5-pair, NBOOT=100, current code state):**

|       | T=1                 | T=2                 |
|-------|---------------------|---------------------|
| **W=2** | 11.66 GB / 24 s   | 11.33 GB / 24 s     |
| **W=3** | **13.87 GB / 12 s** (selected) | — |
| **W=4** | 14.12 GB / 12 s   | —                   |

Decision rule (lowest wall with memory.peak margin under 24 GB): **W=3, T=1**.
That matches the driver's current `NPAIR_WORKERS=3` default — the audit
confirms the default sits at the frontier, no driver change required.
**Threads hurt on this footprint** (nested forking + memory-bandwidth
contention; heavy pairs got ~30% slower at T=2). **Forks help up to W=3,
then plateau** — the single heaviest pair (376k rows, 6.3 s) is the wall.
Safe cgroup cap at this setting: 18 GB (4 GB headroom).

**Caveat for long-list workloads (sce4 ~thousands of contrasts).** With many
more pairs the heaviest-pair ceiling no longer binds — W=4 might pay off
there. Re-sweep on a long pair list before changing the default for that
workload.

**Memory.events across all cells:** oom=0, oom_kill=0, low=0, high=0, max=0.
The OOM safety protocol (systemd-run scope + MemorySwapMax=0) held: zero
cells breached, no host impact.

## Out of scope, but recorded for the next memory pivot

- **Cal_foldchange aFC quantile decoupling.** The pre-cutoff full-vector
  allocation is the largest remaining transient. Replacing the population-
  dependent quantile with a fixed reference value (or a streaming quantile
  digest) would let `Cal_foldchange` run post-cutoff on 188k rows instead
  of 1.56M, restoring the Step 1 gc-off memory win. Behavior change requires
  re-validating sce4 parity, so this is a separate fidelity-gated proposal,
  not a follow-up to this audit.

- **Re-sweep on a long pair list.** Per the W=4 caveat above. Would resolve
  whether the production default should stay at 3 or move to 4 for sce4-scale
  jobs.
