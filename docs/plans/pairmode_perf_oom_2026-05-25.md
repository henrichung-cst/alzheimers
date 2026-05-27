# Pair-mode performance + OOM-safe parallelism plan (2026-05-25)

## Goal

Cut the full 9-contrast pair-mode wall time (currently estimated ~7 days, serial,
single perm-worker) **without** OOMing the shared 30 GB box. An OOM here kills not
just the run but Claude and the desktop session, so every experiment runs under a
hard cgroup cap — the cgroup OOM-kills *inside its scope only*.

This plan is test-first: validate each lever on the two sce4 benchmark pairs and a
few representative pairs **before** touching the production driver
(`alz/incytr_pair/incytr_commandline.R`).

## What the profiling established (grounding)

Measured with `bench/profile_pair_one.R` (+ `.sh` cgroup wrapper) and
`bench/probe_matrix_footprint.R`:

| Fact | Value |
|---|---|
| Per-pair cost dominated by | `Permutation_test` (60–84%) |
| `Permutation_test` output | only the **untrusted** pvalue (we rank on `|PDS|`) |
| Expression matrix `object@data` | 30,567 genes × 29,542 cells, sparse, **0.74 GB** |
| Resident set after `io_load` | ~4.1 GB — i.e. ~3.3 GB is the redundant full Seurat object `Data.input` (counts+data+meta) + omics frames + R base |
| `Data.input` used only in | setup (template build + gene-set construction), never in the per-pair loop |
| Per-pair private memory (peak_hwm − base) | ~1.0–1.2 GB at nboot=100 |
| 2mo contrast cells | 4,858 of 29,542 (16.4%) |
| duckdb engine | wrong phase (optimizes enumeration, 9–23%), net **+18% slower** at per-pair gene.use scale — **not in this plan** |

**Two separable problems:** wall time = permutation; OOM-under-parallelism = forks
COW-inheriting the resident base (R's gc copies it page-by-page). Shrinking the base
is what makes parallelism safe.

## OOM safety protocol (applies to EVERY phase)

1. **Every** run launches inside `systemd-run --user --scope --slice=alz-incytr.slice
   -p MemoryMax=<cap> -p MemorySwapMax=0` (the existing `profile_pair_one.sh` /
   README pattern). The cap is the wall; breach kills inside the scope only.
2. Box is 30 GB. **Never cap above 24 GB** (6 GB reserved for host + Claude).
3. Parallelism scales **incrementally and empirically**: W=1 → read peak
   `memory.current` and `memory.events` → compute per-worker marginal → step W up one
   at a time → stop while `base + W×marginal` projected ≤ 0.7×cap.
4. Abort any scale-up the instant `memory.events` shows `oom > 0` (or `high > 0` if a
   soft limit is set). Record the W where it broke; never push past it.
5. No uncapped runs, ever. No parallel experiment in Claude's own process.
6. nboot is parametrized throughout; parity is nboot-independent (the gate checks
   SigProb recall + sclog2FC, not pvalue), so parity tests run at production
   nboot=100 to stay conservative.

## Phase 0 — Baseline (done; recorded here)

Benchmark-pair per-pair cost at nboot=100, NPERM=1, serial, base ~4 GB:
- Microglia→Cholinergic.Neurons: 22.6 s/pair, peak hwm 5,233 MB, 30,067 output rows.
- glutamatergic×glutamatergic: 8.2 s/pair, peak hwm 5,118 MB, 5,566 output rows.

Insight already banked: permutation cost tracks *surviving* path count (output rows),
not gene.use breadth — so "heavy" ≠ excitatory.

## Phase 1 — Free the redundant Seurat object; prove parity holds — DONE (2026-05-25)

**Result:** `rm(Data.input); gc()` added before the pair loop in
`incytr_commandline.R` (unconditional; no flag survives). A/B on the two
benchmark pairs (ma_2mo_AppP vs WTyp, capped 12 G):
- Output **byte-identical** to baseline — max |Δ| = 0 across all 41 numeric
  columns, every string/label column identical, same 65,979 rows.
- `verify_sce4_parity --all-known-pairs`: **PASS** (Micro→Cholin 573/600 App-only
  residual, max|Δ|=0 R/E/T; Ndnf×Ndnf 599/600, R/T exact, 7 App-EM residual).
- Resident drop smaller than hoped: ~4.25 → ~3.05 GB at the rm point (~1.2 GB
  freed), per-pair RSS ~4.8 → ~3.1 GB. The remaining ~3 GB base is the matrix
  (0.74 GB) + the resident pr/ps/py tibbles & their condition slices + kldata +
  R/arrow/data.table base — all live, all needed by the loop. So the COW base
  for Phase 2 forks is ~3.1 GB, not the ~1.5 GB target.

Harness: `bench/phase1_run.sh`.

### (original plan below)

**Change (test harness first, then driver):** after `create_Incytr(template)` and
`dg_by_cluster` construction, insert `rm(Data.input); gc()`. The per-pair loop only
reads `template`, `dg_by_cluster`, `pr/ps/py_*`, `kldata` — so this cannot alter any
computed value.

**Steps**
1. Add `rm(Data.input); gc()` to a *copy* of the driver path (or a guarded
   `FREE_SEURAT=1` env in a bench harness) right after gene-set construction.
2. Regenerate the two benchmark pairs into a scratch output dir
   (`OUTPUT_DIR_OVERRIDE`), nboot=100, under a 12 GB cap.
3. Run `pixi run python alz/incytr_pair/verify_sce4_parity.py --all-known-pairs
   --wide-dir <scratch>`.
4. Record resident base before/after `rm` (VmRSS + VmHWM).

**Acceptance**
- `verify_sce4_parity: PASS` on both pairs (Micro→Cholin 573/600 App-only residual;
  Ndnf×Ndnf 599/600; max |Δ sclog2FC| = 0 on R/T). **Byte-identical output to
  baseline** is the bar — `rm` must change nothing but memory.
- Resident base drops measurably (target: ~4 GB → ~1.5 GB).

**If parity breaks:** `rm` was placed before a real use of `Data.input` — move it
later; do not proceed.

## Phase 2 — Parallelism perf test — DONE (2026-05-25)

Harness `bench/parallel_pair_probe.R` (outer-loop `mclapply`, each fork runs one
single-pair `Cal_pairwise_grid`, `perm.n.cores=1`), driven by
`bench/parallel_sweep.sh` (one capped scope per W, self-reports cgroup
`memory.peak`/`memory.events`). Fixed 6-pair weight-spanning pool, box has 16
cores so memory — not CPU — is the binding constraint.

| nboot | W | wall/pair | speedup | cgroup peak | oom |
|---|---|---|---|---|---|
| 100 | 1 | 10.86 s | 1.00× | 5.61 GB | 0 |
| 100 | 2 |  6.39 s | 1.96× | 10.29 GB | 0 |
| 100 | 3 |  4.66 s | 2.77× | 13.14 GB | 0 |
| 0   | 1 |  3.97 s | 1.00× | 5.79 GB | 0 |
| 0   | 3 |  1.65 s | 2.72× | 11.92 GB | 0 |

Findings:
- **Speedup is near-linear to W=3** (~2.77×). Marginal RSS per added worker
  ~3–5 GB: fork-`gc()` breaks COW, so each worker carries most of the ~3 GB base
  privately. nboot does **not** change per-worker peak (peak is enumeration/heap-
  bound at `perm.n.cores=1`); nboot only changes *time* (~2.7× faster at 0).
- **W=3 is the safe ceiling under the 24 G cap.** 0.7×24 = 16.8 GB headroom;
  a heavier 18-pair sample (incl. a 374 K-row pair) peaked **15.3 GB** at W=3.
  W=4 worst-case (heavy alignment in a real 961-pair run) projects ~17–20 GB,
  past the 0.7× margin — not recommended. Production W=3 needs the full 24 G cap
  (not 20 G) because the heaviest pairs alone push past 15 GB at W=3.
- Zero oom/high events in every reported configuration.

### (original plan below)

## Phase 2 — Parallelism perf test (OOM-guarded, incremental)

Architecture note: each pair has its own per-cluster `gene.use`, so pairs cannot share
one `Cal_pairwise_grid` enumeration. Parallelism = the **outer** loop runs N
independent single-pair `Cal_pairwise_grid` calls concurrently (each enumerates +
scores + permutes its own pair). Workers share the COW base (template `@data` +
omics frames); private = enumerated paths + dense submatrices + perm cache (≤500 MB).

**Steps (bench harness `bench/parallel_pair_probe.R`, NOT the driver yet)**
1. Load inputs once, build template + `dg_by_cluster`, `rm(Data.input)` (Phase 1).
2. `parallel::mclapply` over a fixed list of W pairs (mix of one heavy + light),
   each running the single-pair body. nboot=100 first.
3. Wrapper escalates W = 1, 2, 3, … each in its own capped scope; after each W, read
   peak `memory.current` and `memory.events` from the cgroup and wall time.
4. Stop at the W that violates the 0.7×cap projection or shows `oom>0`.
5. Repeat the sweep at nboot=0 (forks vanish — expect a much higher safe W).

**Acceptance**
- A table of (W, wall/pair, cgroup peak, oom events) at nboot ∈ {100, 0}.
- Identify max safe W under 24 GB for each nboot, and the realized speedup vs Phase-0
  serial. Zero oom events in any *reported* configuration.

**Guardrail specifics:** start at MemoryMax=12 G for W≤2; only raise toward 24 G once
the per-worker marginal is measured. Heavy pair in the batch must be a high-output-row
pair (e.g. Microglia→Cholinergic at 30 K rows), not a big-gene.use one.

## Phase 3 — Representative per-pair cost on 2–3 other pairs

Per-pair cost varies with surviving-path count, which we cannot predict from cluster
size. Sample the distribution.

**Steps**
1. Profile 2–3 pairs spanning the space, e.g.:
   - a glia→neuron cross (Astrocytes→Excitatory-Pyramidal),
   - a large inhibitory self-pair (Basal-Ganglia-GABAergic × itself),
   - one more chosen from Phase-2 observations (suspected high output rows).
2. Same harness, nboot=100, capped; record time, output rows, peak hwm per pair.

**Acceptance**
- 5+ pairs total characterized (2 benchmark + 3 here). A per-pair time vs output-rows
  relationship good enough to bound a full-run mean. Note min/median/max.

## Phase 4 — Full 9-contrast cost estimate — DONE (2026-05-25)

Unbiased 60-pair random sample (seed 1) of the 961-grid, nboot=100, W=1 serial,
24 G cap: **mean 18.78 s/pair**, single-process peak 7.5 GB, zero oom. Heavy tail
tops at ~442 K rows / ~43 s (the README's ~1.3 M-row / 4–5 min "monsters" do not
occur with current inputs). nboot=0/nboot=100 per-pair ratio ≈ 2.74× (6-pool);
W=3 speedup 2.77× (nboot=100) / 2.72× (nboot=0). +~15 % driver overhead
(per-pair gc×2 + Export + parquet write) folded in below.

Full run = 8,649 pairs (961 × 9 contrasts):

| config | per-pair serial | full-run serial (W=1) | **full-run W=3** | projected peak |
|---|---|---|---|---|
| nboot=100 | ~19 s | ~45 h (~1.9 d) | **~17–19 h** | ~13–15 GB |
| **nboot=0** | ~7 s | ~17 h | **~6–7 h** | ~13–15 GB |

The README's stale ~7-day figure assumed ~75 s/pair; current inputs run ~19 s/pair
at nboot=100. **Recommendation: nboot=0 + W=3 ≈ 6–7 h**, within the 24 G cap.
Parity is nboot-independent (gate checks SigProb recall + sclog2FC, not pvalue),
so nboot=0 output still validates against sce4.

## Phase 4 — original plan

**Steps**
1. From the sampled pairs, take a defensible mean per-pair seconds (weight toward the
   heavier tail; flag the uncertainty).
2. Estimate: `8,649 pairs × mean_s / W_safe / 3600` hours, computed at:
   - (nboot=100, W_safe@100), and
   - (nboot=0, W_safe@0).
3. Produce a short estimate table (config → projected wall, projected peak memory)
   and a recommendation.

**Acceptance**
- A single table the user can decide from, with the memory headroom for each config
  stated explicitly.

## Decision points (need user input, do not assume)

- **nboot policy.** nboot=0 is the largest single lever (removes 60–84% of time *and*
  the fork-OOM driver) but empties the viewer's opt-in pvalue gate. nboot=25 keeps a
  coarse pvalue. This plan *measures* both; the choice is the user's.
- **Cell-subset of `@data` to the contrast's 2 conditions** (0.74 GB → ~0.12 GB) is a
  further parallelism enabler but *changes the permutation null* (currently shuffles in
  all 12 groups' cells). Out of scope unless the user opts in after seeing Phase-2/4
  numbers.

## Anti-shim / rollback

- Phase 1's `rm(Data.input)` is a straight replacement once validated — no
  `FREE_SEURAT` flag survives into the driver; the env toggle exists only in the bench
  harness during testing.
- If parallelism lands, the serial path is replaced, not kept behind a default. The
  cgroup invocation in `alz/incytr_pair/README.md` is updated in the same pass.
- Bench harnesses (`parallel_pair_probe.R`, sweep wrapper) stay in `bench/` as
  artifacts; they are not production code.

## Sequencing

Phase 1 → 2 → 3 → 4, strictly. Do not start Phase 2 until parity holds in Phase 1.
Report results after each phase before proceeding.
