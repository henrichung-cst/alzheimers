> **Archived 2026-05-09.** This document covers the legacy shadow-fork integration code at `alz/integration/{wrappers,adapters,sidecar,tests}/` + orchestrator shells, all relocated to `~/Projects/work/incytr_integration_archive/` on 2026-05-08. Forward-looking guidance lives in `docs/incytr_remediation_plan.md`.

# Incytr layer inventory & cleanup plan

**Purpose.** A single forward-looking source of truth for every optional / parked
layer in our Incytr fork. Each row captures (1) what the layer does, (2) its
current status, (3) the cleanup verdict, and (4) a revival pointer (commit SHA,
ledger row) for layers we delete so they can be re-implemented from history if
ever needed.

**End-state goal.** A clean walkthrough of the codebase as three layers:

- **Section A** — factorial OLS extension (production, always on).
- **Section B** — DuckDB / vectorized performance rewrites (production, always on, bitwise-equivalent to native).
- **Section C** — opt-in extension packs, off by default. Each pack is a coherent unit of related functionality, gated by a single `INCYTR_LAYER_*` env var, with no per-feature flags. Off-by-default means the production pipeline behaves exactly as if Section C did not exist.

Bias on triage: **delete the truly dead, park the deferred as named packs.**
Anything that has a future use case is parked under a single uniform invocation
scheme (`INCYTR_LAYER_*`); anything else is deleted and logged here for revival.

**Evidence basis.** Every disposition cites the audit ledger row from
[`incytr_audit_ledger.md`](incytr_audit_ledger.md), so the audit's per-row
verdict is the substrate; this document is the *next step* (what to do about it),
not a rehash.

---

## Section A — retained as production (no triage)

Factorial OLS extension. Verified by Sprint 1 anchor (`run_degenerate_2cond.sh`):
on 2-condition input, our SigProb / FC slots are bitwise-identical to upstream
`93b9881`.

| Rows | Scope |
|---|---|
| `INC-1` … `INC-12`, `INC-14` … `INC-24`, `INC-31` … `INC-40` | Factorial scaffolding, per-animal SigProb, `Contrast_SigProb`, factorial PDS dispatch, S4 slots, N-condition generalizations of `Cal_SigProb` / `Cal_scFC` / `Pathway_evaluation` / `Cal_PDS`, plus chores. |

These do not get a flag and do not get deleted. They *are* the production
pipeline.

---

## Section B — retained as production (no triage)

Performance rewrites with bitwise-equivalent output. Verified by Sprint 2
anchors (`run_duckdb_enumeration_equiv.sh`, `run_verify_phase2.sh`).

| Rows | Scope |
|---|---|
| `INC-35`, `INC-36`, `INC-37` (perf hunks), `INC-38` | Code reorganization, dplyr → base R + data.table, perf hunks, source split. |
| `ALZ-13`, `ALZ-15`, `ALZ-16`, `ALZ-17`, `ALZ-18` | Cross-pair aggregation, vectorized receiver scoring, DuckDB pathway enumeration. |
| `INC-37.b` | Correctness fixes split out from `c1b6fb9`; numerically neutral. Will be upstreamed separately. |

Bitwise-equivalent. No flag. Stays as-is.

---

## Section C — triage

Bias: delete unless there is a concrete production or sensitivity-analysis use
case we will actually run within the next year.

Legend: ✂ delete · ✓ keep · ❓ decide

| Row | Layer | Current state | Verdict | What gets removed | Justification |
|---|---|---|---|---|---|
| `INC-25` | EM promiscuity weight in `Cal_SigProb` | Default-off (package); wrapper-side `em_w_vec` gated behind `ENABLE_EM_PROMISCUITY_WEIGHT=1`. | ✂ | `em_promiscuity_weight` parameter, `em_weight_log`, `em_w_vec` block in wrapper, `ENABLE_EM_PROMISCUITY_WEIGHT` env var. `Cal_SigProb` becomes single-path no-EM. | Audit ledger: "no empirical / theoretical justification was recorded for the original default-on choice." Nobody runs it on. |
| `INC-28` | Augmented kinase channel: `kl.evidence` / `kl.activity` / `kl.pathways` slots, `Cal_activity_score` family, `Integr_kinasedata` slot writes, `Cal_PDS` branches that read those slots. | Default-inert; wrapper makes zero calls to `Integr_kinasedata`. Owns the SiK_score_* (~0.18) and residual PDS (~0.089) drift on the synthetic golden. | ✂ | All three new slots (revert to native single `@kl`), `Cal_activity_score` family, slot-writing branches in `Integr_kinasedata`, slot-reading branches in `Cal_PDS`. `Cal_PDS` becomes single-path. | Dead in production. Removing it eliminates both golden drifts and reduces `Cal_PDS` to one code path. Native SiK (INC-DESIGN-6) will be a fresh implementation, not a re-enabling of this. |
| `INC-13` | N-condition `Cal_EI` (interface unchanged from native; vectorized over N conditions); plus `reverse_sik_weight` parameter on `Integr_kinasedata`. | Inert when `kldata=NULL`. | ✂ (partial) | `reverse_sik_weight` parameter (only meaningful for the deleted INC-28 channel). `Cal_EI` itself: keep — interface matches native, collapses to native on 2 conditions, no scoring change, no flag. | Once INC-28 is gone, the SiK-specific bits of `Integr_kinasedata` go with it; `Cal_EI` survives as a clean N-condition expression-index helper. |
| `INC-27` | `Permutation_test(type="condition")` + `permutation_test_condition` + `run_permutation_loop` private helpers. | Opt-in; default dispatch matches native. Wrapper never invokes. | ✂ | `type` argument on `Permutation_test`, both private helpers. `Permutation_test` becomes single-path (cell-identity shuffle, native behavior). | Never called from production. |
| `INC-29` | `as_kldata` / `as_kl_evidence` adapters for external kinase-library inputs. | Adapter; only called from `Integr_kinasedata` (deleted with INC-28) and end-user setup. | ✂ | Both adapter functions. | Loses its only caller when INC-28 goes. |
| `INC-30` | PTM scope expansion: `Ack_FC` / `KGG_FC` / `Rme1_FC` slots and parallel scoring path. | Additive; "data not found … skipped" when PTM omics absent. | ✓ | Nothing. | **Verified 2026-05-06**: `Ack_FC`/`KGG_FC`/`Rme1_FC` are handled identically to native `ps_FC`/`py_FC` at slot declaration (`Incytr_class.R:71-75`), `score_omics_layer` dispatch (`evaluation.R:67-71`), export `cbind` (`evaluation.R:358-375`), and iteration (`utils.R:162`). Pure parallel extension to new PTM types — no scoring divergence. Stays in. |
| `INC-30.b` | `cutoff_SigProb` filter applied uniformly across all omics slots. | Default-inert (`cutoff_SigProb = NULL` on the package default code path; wrapper sets `0.01`). | ✓ | Nothing — filter is structurally clean and matches native default behavior when not set. | The cutoff *value* is the `INCYTR_CUTOFF_SIGPROB` knob below. The filter machinery itself is fine. |
| `ALZ-19` | Kinase-imputed gene expansion adapter + soft-rescue weighting in factorial wrapper. | Shell-gated behind `ENABLE_KINASE_IMPUTATION=1` (off by default). | 📦 park (kinase pack) | Existing per-feature env var collapses into `INCYTR_LAYER_KINASE_PACK`. Code path stays; rescue block at `run_incytr_factorial_all_pairs.R:471–495` and `EXPR_IMPUTATION_FLOOR` constant remain, gated by the unified flag. | Future Section C re-activation. Bundled with ALZ-20 as a coherent kinase extension pack so they're toggled together (they're conceptually the same feature: kinase-aware extensions). |
| `ALZ-20` | `compute_kinase_support_factorial.py` sidecar (IDF, sender-attribution discount, median aggregation). Reads `recv_*.parquet`, writes `kinase_support_scores.csv` + optional `kinase_routes.parquet` per pair. | Sidecar; not invoked from `run_factorial_all_pairs.sh`. | 📦 park (kinase pack) | Move under `alz/integration/sidecar/kinase_pack/` to make sidecar status obvious in the path. Invocation gated by `INCYTR_LAYER_KINASE_PACK=1`. | Future Section C re-activation alongside ALZ-19. |
| `ALZ-9` / `ALZ-11` / `ALZ-12` | Backbone permutations (within-receiver shuffle "enrichment null" + "wiring null"); separate runner `run_factorial_permutations.sh` writing `backbone_permutation_pvalues_by_contrast.csv`. | Sidecar; never gates native PDS. | 📦 park (backbone perms) | Move runner + helpers to `alz/integration/sidecar/backbone_perms/`. Invocation gated by `INCYTR_LAYER_BACKBONE_PERMS=1`. | Park pending design revisit: the distinction between "backbone" and "pathway" needs to be re-examined before this is reactivated. The null-distribution machinery itself is sound; the conceptual carving is what needs work. |

---

## Standardized invocation scheme

All optional behavior lives behind a single uniform namespace: `INCYTR_*`. One
prefix, defaults declared in one place, no per-feature env vars scattered across
the wrapper / shell / package.

| Knob | Default | Effect |
|---|---|---|
| `INCYTR_LAYER_KINASE_PACK` | `0` | When `1`: enables ALZ-19 kinase-imputed gene expansion (rescue block in wrapper) **and** ALZ-20 kinase support score sidecar (`compute_kinase_support_factorial.py`). The two are bundled as one coherent kinase-aware extension pack. |
| `INCYTR_LAYER_BACKBONE_PERMS` | `0` | When `1`: enables backbone permutation runner (within-receiver shuffle nulls). Off pending the backbone-vs-pathway design revisit. |
| `INCYTR_CUTOFF_SIGPROB` | `0.0` | Pre-prune cutoff for DuckDB pathway enumeration. `0.0` matches native (no pre-prune); a future Section C decision may raise this for production runtime if benchmarks justify. |

### Single registry

The defaults and their consumption sites are defined in **one** place:
`alz/integration/wrappers/incytr_runtime.R` (or equivalent module). The
wrapper, shell runner, and any sidecar invocation all read from this registry.
No package-level defaults override the registry; the registry is the contract.

### Migration from existing flags

| Old flag | New flag | Disposition |
|---|---|---|
| `ENABLE_EM_PROMISCUITY_WEIGHT` | (none) | Deleted with INC-25. |
| `ENABLE_KINASE_IMPUTATION` | `INCYTR_LAYER_KINASE_PACK` | Renamed; same default (off); now also gates ALZ-20. |
| `DUCKDB_CUTOFF_SIGPROB` | `INCYTR_CUTOFF_SIGPROB` | Renamed; default lowered from `0.01` to `0.0` to match native. |

### What this buys

- One namespace, one default location, one source of truth.
- "Is the production pipeline native-equivalent right now?" answered by reading
  three env vars, not by grepping for absent function calls.
- New optional features added later use the same `INCYTR_LAYER_*` pattern. No
  ad-hoc neutralization mechanisms.
- "What does Section C look like today?" is a one-liner: list of layer keys
  whose flags are off.

---

## Decisions resolved (2026-05-06)

1. **PTM (INC-30)**: ✓ keep — verified parallel-extension to native PTM tracks.
2. **Kinase imputation (ALZ-19)**: 📦 park — bundled into kinase pack for future Section C.
3. **Kinase support score (ALZ-20)**: 📦 park — bundled into kinase pack for future Section C.
4. **Backbone permutations**: 📦 park — pending backbone-vs-pathway design revisit.
5. **Cutoff**: keep as option; default `0.0` (native). Production-runtime tuning is a future Section C decision.

Convention: `📦 park` means the layer stays in the codebase but is moved under
`alz/integration/sidecar/<pack>/` and gated by a single `INCYTR_LAYER_*` env
var; off by default.

---

## Native SiK reinstatement (INC-DESIGN-6) — separate track

Not in this cleanup. INC-DESIGN-6 is *future feature work* — wiring native
`Integr_kinasedata` (single `@kl` slot, native `Cal_PDS` branch) into the
factorial wrapper. After this cleanup, native SiK will be a clean addition on
top of a clean baseline, not a re-enabling of dormant code.

Suggested follow-up: when we add native SiK, it goes in as one well-tested PR
(touches `Integr_kinasedata`, `Cal_PDS`, the wrapper), with no flag (always on).

---

## Execution order

Smallest / safest first; each step is one PR, gated by 593/593 testthat +
`run_degenerate_2cond.sh` + `run_duckdb_enumeration_equiv.sh`.

### Phase 1 — delete dead Section C code

1. **INC-27**: delete `Permutation_test(type=)` dispatch + 2 private helpers. Trivial.
2. **INC-25**: delete `em_promiscuity_weight` parameter, `em_weight_log`, wrapper `em_w_vec` block, `ENABLE_EM_PROMISCUITY_WEIGHT` env var. `Cal_SigProb` becomes single-path no-EM.
3. **INC-28 + INC-29**: delete `kl.evidence` / `kl.activity` slots, `Cal_activity_score` family, slot-writing in `Integr_kinasedata`, slot-reading branches in `Cal_PDS`. Revert `kl.pathways` to native `@kl`. Delete `as_kldata` / `as_kl_evidence` adapters (their only callers are deleted in this same PR). Re-run synthetic golden — both ~0.18 SiK_score_* and ~0.089 PDS drifts should go to zero. **This is the real cleanliness milestone.**
4. **INC-13 trim**: drop `reverse_sik_weight` parameter from `Integr_kinasedata`. Keep `Cal_EI` N-condition generalization.

### Phase 2 — standardize invocation

5. **Create `alz/integration/wrappers/incytr_runtime.R`**: single registry of `INCYTR_LAYER_KINASE_PACK`, `INCYTR_LAYER_BACKBONE_PERMS`, `INCYTR_CUTOFF_SIGPROB`. All defaults declared here.
6. **Migrate `ENABLE_KINASE_IMPUTATION` → `INCYTR_LAYER_KINASE_PACK`**: rename, point reads at the registry, no behavior change.
7. **Migrate `DUCKDB_CUTOFF_SIGPROB` → `INCYTR_CUTOFF_SIGPROB`**: rename, default `0.01 → 0.0`, point reads at the registry. Run a one-shot benchmark on a real all-pairs run to record the actual runtime / output-size cost of `0.0` for future reference.

### Phase 3 — relocate parked sidecars

8. **Move kinase pack**: `compute_kinase_support_factorial.py` → `alz/integration/sidecar/kinase_pack/`. `export_kinase_imputed_genes_factorial.py` and the wrapper rescue block stay where they are but the rescue block reads `INCYTR_LAYER_KINASE_PACK` instead of `ENABLE_KINASE_IMPUTATION`. Add a small README at `alz/integration/sidecar/kinase_pack/README.md` explaining the pack and the activation flag.
9. **Move backbone perms**: `run_factorial_permutations.sh` → `alz/integration/sidecar/backbone_perms/`. Helper functions stay in `aggregate_factorial.py` (deletion would require a larger refactor) but are gated behind the registry flag at their call sites. README notes the design-revisit reason for the park.

### Phase 4 — done

After Phase 3:

- Production wrapper (defaults) = pure Section A + Section B + INC-30 PTM (data-conditional). No Section C active.
- All three optional knobs read from one registry, one prefix.
- Sidecar status legible from path (`alz/integration/sidecar/<pack>/`).
- Audit's must-match gates (`run_degenerate_2cond.sh`, `run_duckdb_enumeration_equiv.sh`) still pass; testthat 593/593 still green.

---

## Revival pointers

For each deleted layer, the revival recipe is:

- **Row identifier** (e.g., `INC-28`) → look up in [`incytr_audit_ledger.md`](incytr_audit_ledger.md)
  for the originating commit SHA, files touched, and intended scope.
- **Pre-cleanup state** lives at the commit immediately before the deletion PR;
  the deletion PR description should record `Reverts: <SHA-range>` and
  `Restores: git checkout <SHA> -- <file paths>` as the recipe.
- **This document** stays the index. When a layer is deleted, append the
  deletion-PR SHA to its row.

---

## Status

- 2026-05-06: triage written and resolved.
- 2026-05-06: **Phase 1 complete** — Section C dead code deleted across
  3 commits in `incytr` (INC-27, INC-25, INC-28+INC-29) and 1 commit in
  `alzheimers` (INC-25 wrapper wiring). Net ~2,500 lines removed; testthat
  470/470; `run_degenerate_2cond.sh` + `run_duckdb_enumeration_equiv.sh`
  passing.
- 2026-05-06: **Phase 2 complete** — `INCYTR_*` namespace standardized.
  Single registry at `alz/integration/incytr_runtime.sh` (shell) +
  `alz/integration/wrappers/incytr_runtime.R` (R). `ENABLE_KINASE_IMPUTATION`
  → `INCYTR_LAYER_KINASE_PACK` (default flipped on→off). `DUCKDB_CUTOFF_SIGPROB`
  → `INCYTR_CUTOFF_SIGPROB` (default `0.01` → `0.0`, native-equivalent).
  Anchor gates re-verified.
- 2026-05-06: **Phase 3 complete** — parked sidecars relocated.
  `compute_kinase_support_factorial.py` → `alz/integration/sidecar/kinase_pack/`;
  `run_factorial_permutations.sh` → `alz/integration/sidecar/backbone_perms/`.
  READMEs added at each path documenting activation flag and revival pointers.
  Backbone-perms call site in `aggregate_factorial.py --permutations` now
  hard-gated by `INCYTR_LAYER_BACKBONE_PERMS=1`. Production wrapper defaults
  unchanged: pure Section A + Section B + INC-30 PTM.
