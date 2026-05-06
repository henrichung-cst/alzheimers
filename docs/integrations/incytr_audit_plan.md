# Incytr Audit Plan

Status: **Draft, awaiting sign-off on Sprint 0 scope.**
Owner: Henri Chung
Started: 2026-05-05

## 1. Why this exists

We extended Incytr from a two-condition pairwise method into a factorial (9-contrast) AD pipeline and added several scoring layers. The work proceeded too quickly. We need to revisit every change against upstream, prove that the factorial version is a faithful logical extension of native Incytr, that performance rewrites are equivalence-preserving, and that every remaining change has independent justification or is reverted.

This document is the **working plan** for that audit. It will be edited in place as sprints complete; the per-change ledger in §6 is the authoritative record.

## 2. Guiding principles

Every change must end the audit in exactly one of three buckets:

- **A. Factorial extension.** The change is logically entailed by lifting two-condition Incytr to a 9-contrast factorial design. The native formula is preserved; only the input becomes per-contrast.
- **B. Performance.** The change does not alter scientific output relative to native Incytr on a matched two-condition input. Equivalence must be demonstrated, not asserted.
- **C. Discretionary.** The change is not entailed by either A or B. It must be justified independently. If it cannot be justified, the code is reverted from baseline; the implementation is preserved in the parking branch (§4) for possible later reinstatement.

If a change spans two buckets (e.g., a function that both extends to factorial AND adds a new filter), it is split during the audit so each piece sits in exactly one bucket.

## 3. Reference points

| Anchor | Repo | Commit | Meaning |
|---|---|---|---|
| Upstream HEAD at fork | `incytr` | `93b9881` (2025-10-12, changhanhe@gmail.com) | "Native Incytr." Ground truth for native behavior. |
| Our incytr HEAD | `incytr` | current `main` | All package-level changes since fork. |
| Our alzheimers HEAD | `alzheimers` | current `main` | All wrapper code under `code/integration/`. |
| Audit working branch | `alzheimers` | `incytr-audit` (to create) | Where audit-driven reverts and refactors land. |
| Audit working branch | `incytr` | `incytr-audit` (to create) | Same, package side. |
| Parking branch | `incytr` | `parking/incytr-discretionary` (to create) | Carries reverted-from-baseline discretionary code as cherry-picks; navigable via tags. |

Upstream evolution past `93b9881` will be checked once at Sprint 0 start (in case `ChanghanGitHub/incytr` has moved). If it has, those commits are treated as native, not "our changes."

## 4. Mechanism

### 4.1 In-place audit, not rebuild-from-scratch

We do not branch off `93b9881` and re-cherry-pick from current HEAD. Instead, the audit branch starts at current HEAD, and each change is examined in place. Reverts are explicit `git revert` (or surgical edit) commits on the audit branch with a reference to the originating commit and a one-line audit verdict.

### 4.2 Parking mechanism

When a section-C change cannot justify itself:

1. The introducing commit(s) are tagged: `parking/<short-name>` (e.g., `parking/em-promiscuity-weight`).
2. The commits are cherry-picked onto `parking/incytr-discretionary` so they accumulate there independently of `main`.
3. The audit branch reverts the change (or refactors it out).
4. The ledger entry records the parking tag, so re-introduction is a cherry-pick away.

This applies to both repos.

### 4.3 Equivalence test (Section A and B sign-off)

The audit cannot conclude without numerical evidence. The test:

- **Fixture**: `incytr/examples/5xad_data/` (already exists). Two-condition (WT vs 5xAD), single sender-receiver pair.
- **Native run**: Native Incytr at `93b9881` end-to-end on the fixture, all default parameters. Output: `TPDS, PPDS, PhPDS_ps, PhPDS_py, multimodel_score, PDS, SiK_score` per pathway. This is the **golden output**.
- **Wrapper run, two-condition mode**: Run the alzheimers wrapper with a degenerate input (one timepoint, one disease genotype, no factorial). The 9-contrast OLS collapses to a single contrast that should reproduce native semantics.
- **Comparison**: Per-pathway, per-column, bitwise. Where bitwise fails but values are within ~1e-6, defer to ad-hoc human approval (Henri).

**Tolerance policy**: bitwise default; numerical near-match acceptable only with documented sign-off in the ledger. Stochastic components (permutation tests) are seeded; if seeded reproducibility cannot be achieved, the test for that component is "distribution match" (KS or summary statistics) instead of bitwise.

The existing `tests/testthat/` suite is leveraged and extended; `test-golden_output.R` already exists and is the natural home for this.

### 4.4 Per-change ledger

Single source of truth for audit verdicts. Lives at `docs/integrations/incytr_audit_ledger.md` (created Sprint 0). Schema:

| Column | Description |
|---|---|
| `id` | Stable short ID, e.g., `EM-DEGREE-WEIGHT` |
| `repo` | `incytr` or `alzheimers` |
| `location` | File and function/line range |
| `introducing_commits` | SHAs in chronological order |
| `description` | One-paragraph summary of the change |
| `bucket` | A / B / C (assigned at audit time) |
| `native_counterpart` | Function or formula in upstream `93b9881`, if any |
| `equivalence_test` | Test name(s) covering this change |
| `justification` | Filled at audit time; for C-bucket items, the case for keeping |
| `disposition` | `keep` / `keep-flagged` / `refactor` / `revert-park` |
| `parking_tag` | If reverted, the `parking/...` tag |
| `verdict_date` | When the row was finalized |
| `signoff` | Henri's confirmation (Y/N or "deferred") |

### 4.5 Working doc lifecycle

This plan doc is edited as sprints complete. The ledger is append-only within a sprint, immutable after sprint sign-off. A sprint is "done" when:
- All ledger rows for that sprint have a verdict and signoff.
- All keep/refactor/revert actions are merged to the audit branch.
- The equivalence test for that sprint passes (or the deviation is documented and signed off).

## 5. Sprint plan

Five sprints. Each ends with a signed-off ledger snapshot and a passing test gate.

### Sprint 0 — Scaffolding (no code semantics change)

**Goal**: make the audit possible.

Deliverables:
1. Confirm upstream `93b9881` is still upstream HEAD; if not, advance the reference and note it in §3.
2. Enumerate every commit `1e64f41 → HEAD` in `incytr` and every commit touching `code/integration/` in `alzheimers`. Produce a raw commit list (input to the ledger) at `docs/integrations/incytr_audit_commit_list.md`.
3. Create branches: `incytr-audit` (both repos), `parking/incytr-discretionary` (incytr only).
4. Stand up the equivalence-test harness: a script that runs native Incytr at `93b9881` on `examples/5xad_data/` and saves golden output to `tests/golden/native_93b9881/`. Run once, commit golden artifacts.
5. Stand up the corresponding wrapper-side runner: a degenerate two-condition invocation of the alzheimers factorial pipeline that emits a comparable artifact.
6. Stub the per-change ledger doc.

Sign-off gate: harness runs end-to-end, golden output exists, ledger schema is reviewed.

### Sprint 1 — Section A: Factorial extension (incytr repo)

**Goal**: prove the package-level factorial code is a faithful logical extension.

Scope: every change in the `incytr` repo since fork that touches per-animal SigProb, OLS contrasts, factorial Pathway_evaluation, or factorial PDS. The testthat suite already covers most of these (`test-Cal_SigProb_animal.R`, `test-Pathway_evaluation_factorial.R`, `test-Cal_PDS_factorial.R`, `test-factorial_integration.R`).

For each change:
1. Locate the native counterpart (function in `93b9881`).
2. Verify the formula is preserved per contrast (e.g., `TPDS = logi(aFC, k=2)` becomes `TPDS_{c} = logi(aFC_{c}, k=2)` per contrast `c`, with `aFC_{c}` from the OLS).
3. Verify default weights match native: `score.weight = c(0.5, 0.5, 0.5)`, `KPDS.weight = 0.5`, `k_logi = 2`.
4. Run the equivalence test with the wrapper in degenerate two-condition mode.

Specific items already known to need attention:
- Per-animal SigProb design choice (audit decision: "necessary, alternative considered" — see plan §7).
- 10-parameter design matrix and 9 contrast vectors.
- Per-contrast columns in `Pathway_evaluation_factorial`.
- Default scoring weights — verify untouched.

Sign-off gate: equivalence test passes; ledger A-bucket rows complete.

### Sprint 2 — Section B: Performance (both repos)

**Goal**: prove performance rewrites are equivalence-preserving.

Scope (known items):
- `duckdb_enumeration.R` replacing `pathway_inference()`'s data.table joins (alzheimers wrapper). Verify by disabling its internal pre-prune cutoffs and matching native pathway-set output bitwise.
- `receiver_scoring.R` vectorized scoring. `verify_phase2.R` exists for this purpose but is manual; promote it into the testthat suite or a runner-invoked check.
- Any package-level performance commits in `incytr` (e.g., `c1b6fb9 "Optimize performance, fix correctness issues, improve code organization"` may bundle both — these have to be split).

Critical sub-task: any commit that bundles "performance + correctness fix" is **split** in the ledger. Performance pieces go to bucket B; correctness fixes are evaluated separately and are likely bucket A (if they fix a bug native Incytr also has, that's a pre-existing-bug discussion to have with upstream; if they fix a bug introduced by our refactor, it's a wash).

Pre-prune cutoffs (`Hill < 0.01`, `SigProb >= 0.01 OR`) are **bucket C** even though they live inside DuckDB enumeration — they are filters, not performance. Audited in Sprint 4, not here.

Sign-off gate: equivalence test passes with all bucket-B changes in place; ledger B-bucket rows complete.

### Sprint 3 — Section C, group 1: Scoring formula changes

**Goal**: justify or revert items that modify the SigProb / TPDS / PDS formulas themselves.

Items (known):
- **EM promiscuity weight** `1/log2(1+degree)` modifying SigProb. Default disposition: revert + park, unless an empirical or theoretical justification is recorded.
- **`logi()` k parameter, scoring weights** if any have drifted from native defaults — revert.
- **Find_highexp_gene replacement** with 10% detection threshold (vs native `cutoff_percentile = 0.5`). Audit whether the wrapper actually uses native `Find_highexp_gene` or replaces it with a different gene-admission rule.

Sign-off gate: each item has a verdict row; reverted code lives in `parking/incytr-discretionary` with a tag.

### Sprint 4 — Section C, group 2: Filtering and pathway universe changes

**Goal**: justify or revert items that change which pathways enter scoring.

Items (known):
- DuckDB pre-prune at `Hill < 0.01`.
- DuckDB SigProb pre-filter at `SigProb >= 0.01 OR`. Native default is `cutoff_SigProb = NULL`.
- **Kinase-imputed receiver-gene expansion** (`export_kinase_imputed_genes_factorial.py`, R-side floor `EXPR_IMPUTATION_FLOOR = 0.05`, soft rescue logic). Native has no kinase-substrate-driven gene admission. Default disposition: revert from baseline; the audit doc already gates it behind `ENABLE_KINASE_AUGMENTATION`, so this may be partial cleanup rather than full revert.
- WMB ↔ SEA-AD subclass remapping via `ENABLE_CELLTYPE_MAPPING`. Audit whether it changes results when disabled.

Sign-off gate: same as Sprint 3.

### Sprint 5 — Section C, group 3: Added analysis layers

**Goal**: justify or revert layers that produce new outputs alongside native ones.

Items (known):
- **External `kinase_support_score` + λ reranking** (`compute_kinase_support_factorial.py`, IDF, sender attribution discount, median aggregation). Strong default disposition: revert from baseline output, keep as a **separate downstream** consumer of the baseline Parquet — never overwriting native PDS.
- **Backbone dual-null permutation tests** (`run_factorial_permutations.sh`, Storey q-values). Strong default: keep as separate analysis stage; never let its q-values gate native PDS-based selection.
- **`examine_factorial.py`** interpretive layer (additivity, temporal, cell-type centrality figures). Justification is "downstream interpretation" — keep, no native equivalent claimed, no overlap with native scoring.

Sign-off gate: each layer is either separated cleanly from baseline or reverted. The factorial baseline output (the Parquet schema) contains only native-equivalent columns; non-native columns live in adjunct files or downstream stages.

### Reinstatement of native SiK channel

Cross-cutting concern, addressed during Sprint 5. The native SiK channel (kinases that are themselves pathway nodes phosphorylating other nodes) is part of native baseline Incytr — `Cal_EI()`, `Integr_kinasedata()`, contribution to `PDS` via `Cal_PDS()`. The audit doc currently treats SiK as optional alongside the external kinase support score under one `ENABLE_KINASE_AUGMENTATION` flag. **Native SiK should run by default**; only the external kinase_support_score should be opt-in. The flag is split during Sprint 5.

## 6. Per-change ledger

See `docs/integrations/incytr_audit_ledger.md` (created in Sprint 0).

## 7. Open design decisions to record at audit time

These need explicit verdicts even though they are A-bucket "necessary" extensions. They are not mistakes, but they are choices, and the rationale should live in the ledger.

1. **Per-animal SigProb vs per-condition mean SigProb.** Native Incytr computes SigProb on a condition-mean expression vector; our wrapper computes SigProb per animal then OLS. These are not algebraically equal because Hill is nonlinear. The current choice (a) is defensible — it gives SE and p-values per contrast — but it is a divergence from the natural "TPDS = logi(aFC of mean SigProb)" reading of native semantics. Record both options, document why (a) was chosen, get explicit signoff.

2. **`Cal_foldchange` correction term in factorial mode.** Native uses an additive correction (`1e-4`) before `log2`. In factorial OLS on per-animal SigProb, the correction is implicit in the linear model. Verify this is handled consistently and does not silently introduce a new constant.

3. **`aFC` vs `log2FC` in evaluation.** Native default is `aFC` (75th-percentile-capped). Our factorial OLS estimates contrasts on per-animal SigProb — does the wrapper compute the per-contrast `aFC` analog using the same Hill-cap rule, or has it switched to plain `log2FC`? If the latter, that's a Section A divergence to flag.

4. **Sample filtering scope.** Native Incytr operates on whatever cells the user passes in; our wrapper hard-codes `males-only`. This is an upstream filtering choice (correct for the AD analysis), not a Incytr modification. Ledger entry: "wrapper-level filter, not a wrapper modification of Incytr."

## 8. Risks and known limitations

- **Stochastic reproducibility.** DuckDB row ordering, parallel reductions, permutation seeds — bitwise reproducibility may require disabling parallelism for the equivalence run. Acceptable; tested separately.
- **Bundled commits.** Several "Refactor, fix bugs, ..." commits bundle multiple buckets. The audit must split them; each piece gets its own ledger row even if they share a SHA.
- **Upstream may have moved past `93b9881`.** Sprint 0 step 1 catches this.
- **Test fixture coverage.** The 5xAD fixture is two-condition; it does not exercise the factorial code paths directly. The factorial pieces are covered by the testthat factorial-named tests, but those compare wrapper-to-wrapper, not wrapper-to-native (because native has no factorial). The audit relies on the **degenerate-two-condition collapse** being exercised end-to-end; if that path is untested today, Sprint 0 must add it.

## 9. Out of scope

- Re-running Incytr against the WMB-class taxonomy (currently mapped via `SEA_AD_SUBCLASSES`). Tracked separately; vocabulary choice is wrapper-side, not Incytr-side.
- Changes to the bulk kinase pipeline (`code/kinase_attribution.py` etc.). The audit treats `unified_attribution.csv` and `mea_stoichiometry.csv` as input.
- Upstream contribution. If the audit identifies fixes that upstream Incytr should have, that's a follow-up — not blocking.

## 10. Status checklist

- [x] Sprint 0: scaffolding _(2026-05-05; branches + golden harness + native/current goldens + pre-audit diff + ledger seeded — see `incytr_audit_pre_diff.md`)_
- [x] Sprint 1: factorial extension (Section A) _(2026-05-05; 28 A-bucket signed off, 4 design decisions recorded, PDS drift attributed to INC-25 → Sprint 3, SiK_score drift attributed to INC-28 + co-suspect INC-13 → Sprint 5; degenerate 2-cond runner at `code/integration/tests/run_degenerate_2cond.sh`; 593/593 testthat passing)_
- [x] Sprint 2: performance (Section B) _(2026-05-05; 9 B-bucket rows signed off, INC-37 split into perf + correctness (INC-37.b, fix-forward, no Sprint 3/4 re-route), DuckDB enumerator bitwise-equivalent to native `pathway_inference` with pre-prune cutoffs disabled, `verify_phase2.R` promoted to `code/integration/tests/run_verify_phase2.sh`; 593/593 testthat passing)_
- [x] Sprint 3: scoring formula changes (Section C, group 1) _(2026-05-05; INC-25 EM promiscuity weight reverted to default-off in package + env-gated in wrapper, parking-in-place per ENABLE_KINASE_AUGMENTATION precedent; ALZ-22 wrapper gene-admission divergence signed off as snRNA-seq config; INC-DESIGN-5 `logi()` k parameter audit confirmed no drift; Sprint 1 attribution amended — synthetic fixture has `em_degree=1` making INC-25 a no-op there, residual PDS drift now owned by Sprint 5 INC-28; 593/593 testthat passing)_
- [x] Sprint 4: filtering and pathway universe (Section C, group 2) _(2026-05-05; 6 C-bucket filter rows resolved — ALZ-18.b DuckDB pre-prune kept-flagged with `DUCKDB_CUTOFF_SIGPROB` env-var opt-out (mathematically lossless for any reasonable top-K cut), ALZ-19 + ALZ-23 confirmed parked behind default-off `ENABLE_KINASE_AUGMENTATION`/`ENABLE_CELLTYPE_MAPPING`, INC-30/INC-30.b/ALZ-2 signed off as PTM scope expansion / default-inert filter; wrapper edit limited to 1 line in `run_incytr_factorial_all_pairs.R:86` plus env-var forwarding in `run_factorial_all_pairs.sh:85`; 593/593 testthat passing; DuckDB equivalence preserved on `cutoff_SigProb = 0` path)_
- [x] Sprint 5: added analysis layers + native SiK reinstatement (Section C, group 3) _(2026-05-05; 9 C-bucket added-layer rows resolved — INC-13 N-condition kinase scoring signed off as legacy-collapse-preserved when `kldata=NULL`; INC-26 docs-only; INC-27 `Permutation_test(type=...)` additive with native default; INC-28 augmented kinase channel signed off as default-inert (drift attribution finalized: SiK_score_* ~0.18 + residual PDS ~0.089 are the expected signatures of the augmented channel on the synthetic golden, not regressions); INC-29 adapters not on default code path; ALZ-9/11/12 backbone permutations confirmed sidecar (q-values never gate native PDS); ALZ-20 external kinase support score confirmed sidecar (no λ-style PDS rerank exists — λ at `aggregate_factorial.py:499` is Storey's pi0 q-value parameter); INC-DESIGN-6 records that native SiK is *absent* from the factorial wrapper, not gated off, and defers the wiring to a follow-up PR (audit-plan's `ENABLE_KINASE_AUGMENTATION` flag does not exist in code — actual flag is `ENABLE_KINASE_IMPUTATION`); 593/593 testthat passing; doc-only sprint, no incytr or alzheimers code edits)_

**Audit closed 2026-05-05.** All five sprints signed off. Remaining follow-ups (tracked outside the audit): native SiK wiring (per INC-DESIGN-6), upstream PR for INC-37.b correctness fixes.

## 11. Related documents

- `docs/integrations/incytr_invocation_audit.md` — Prior column-by-column audit; will be revised once Sprint 1 produces real numerical comparisons.
- `docs/integrations/kinase_incytr_integration.md` — Source of truth for current integration design; will be amended sprint-by-sprint to reflect the audited baseline.
- Upstream reference: `../incytr` at commit `93b9881`.
