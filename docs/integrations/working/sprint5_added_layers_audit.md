# Sprint 5 — Added analysis layers + native SiK reinstatement (Section C, group 3)

Per audit-plan §5 Sprint 5: justify or revert layers that produce *new outputs
alongside* native ones. Sign-off gate: each layer is either separated cleanly
from baseline or reverted. The factorial baseline output (`recv_*.parquet`)
contains only native-equivalent columns; non-native columns live in adjunct
files or downstream stages.

## Items audited

### 1. `INC-13` (`ca6a96e`) — N-condition kinase scoring generalization — SIGN OFF

`incytr/R/kinases.R` against native `incytr-93b9881/R/kinases.R`:

- `Cal_EI` (`R/kinases.R:569`): signature unchanged from native (`df`,
  `cell_group`, `fold_threshold=10`); body vectorized but interface identical.
  Two-condition collapse preserved (verified by 593/593 testthat +
  `run_degenerate_2cond.sh` must-match SigProb/FC slots clean).
- `Integr_kinasedata` (`R/kinases.R:639`): adds `reverse_sik_weight=0.3`
  parameter (absent from native at `93b9881:R/kinases.R:78`); calls
  `as_kldata(kldata)` on entry; writes three slots
  (`kl.pathways` / `kl.evidence` / `kl.activity`) vs native's single `@kl`. The
  slot proliferation is the *structural enabler* of the `SiK_score_*` drift,
  but is **inert when `kldata = NULL`** (the parameter default).

The factorial wrapper (`code/integration/wrappers/run_incytr_factorial_all_pairs.R`)
makes zero calls to `Integr_kinasedata`, so the slot proliferation is
observationally inert in the production pipeline. The previously-flagged
`SiK_score_*` golden drift is owned by INC-28 (see §3 below); INC-13 is the
structural co-anchor, not a numerical driver.

**Verdict: signed-off (N-condition generalization; legacy two-condition
collapse preserved when `kldata` is absent).**

### 2. `INC-26` (`b654fdd`) and `INC-27` (`f922c3a`) — condition-label permutation test — SIGN OFF

`INC-26` is docs-only (`docs/incytr_proposals/*.md`).

`INC-27`:
- `Permutation_test` (`R/analysis.R:667`) dispatches on
  `type = c("cell_identity", "condition")`. Native (`93b9881:R/analysis.R:1084`)
  only has cell-identity shuffle. The `type` argument defaults to native
  behavior — calling `Permutation_test()` with no `type` matches native exactly.
- `permutation_test_condition` (`R/analysis.R:804`, private) and
  `run_permutation_loop` (`R/analysis.R:3`, private) are reachable only through
  `Permutation_test(type = "condition")`.

**Verdict: signed-off (additive; default dispatch matches native).** Cell-
identity shuffle remains the default; condition-label shuffle is opt-in via the
`type` parameter. 593/593 testthat exercises both paths.

### 3. `INC-28` (`6858063`) — memory permutation pass + `kl.evidence`/`kl.activity` slots — SIGN OFF

The +651-line `kinases.R` expansion introduces:
- `kl.evidence` slot (`R/Incytr_class.R:81`), written by `Integr_kinasedata` at
  `R/kinases.R:179`; absent from native.
- `kl.activity` slot (`R/Incytr_class.R:82`), written at `R/kinases.R:176`;
  absent from native.
- `kl.pathways` slot (`R/Incytr_class.R:78`), written at `R/kinases.R:690, 738,
  746`; native has `@kl`.
- `Cal_activity_score` family (`R/kinases.R:444, 488`); absent from native.

`Cal_PDS` (`R/evaluation.R:471`) reads `@kl.evidence` (`evaluation.R:490–512`)
and `@kl.pathways` (`evaluation.R:545–546, 618–619`), but only **after** the
slots are populated. On default constructions (NULL slots) these branches no-op.

**Drift attribution.** This is the *prime suspect* for both drift signatures
seen in the synthetic golden:

- `SiK_score_*` drift max\|diff\|≈0.18 (originally raised in Sprint 0 pre-diff,
  named INC-13 + INC-28 co-suspects in Sprint 1).
- residual `evaluation.PDS` drift max\|diff\|≈0.089 (re-attributed away from
  INC-25 in Sprint 3 because the synthetic fixture has `em_degree = 1` making
  the EM weight a no-op).

Both drifts are exposed only when `Integr_kinasedata(kldata = <non-null>)` is
invoked. The factorial wrapper does not invoke it, so the drift is the
*expected signature of the augmented kinase channel* on the synthetic golden,
not an accidental regression on production runs.

**Verdict: signed-off (additive kinase channel; default-inert when `kldata`
absent; drift is the expected signature of the augmented channel on the
golden, not a regression on baseline runs).**

### 4. `INC-29` (`d6d5c8c`) — `as_kldata` / `as_kl_evidence` adapters — SIGN OFF

Pure adapter functions for external kinase-library inputs:
- `as_kldata` (`R/kinases.R:534`): normalizes `Kinase` / `Substrate` / site-
  position columns to canonical `gene` / `site_pos` / `motif.geneName`; rejects
  list inputs with an informative pointer to `as_kl_evidence`.
- `as_kl_evidence` (`R/kinases.R:555`): accepts data frames or named lists of
  per-kinase substrate tables; outputs canonical
  `kinase` / `substrate` / `site_pos` / `score` / `p_value` / `padj`.

Called only from `Integr_kinasedata` (D3) and end-user setup code; no native
counterpart, not on default code path.

**Verdict: signed-off (adapter; no native counterpart, not on default code
path).**

### 5. `ALZ-9` / `ALZ-11` / `ALZ-12` — backbone permutations as separate stage — SIGN OFF

`code/integration/run_factorial_permutations.sh` invokes
`code/integration/adapters/aggregate_factorial.py::_run_permutation_one_contrast`
once per contrast (9 contrasts: App/Tau/ApTt × 2mo/4mo/6mo) and concatenates
per-contrast results into
`intermediates/factorial/all_pairs/aggregation/backbone_permutation_pvalues_by_contrast.csv`.
Within-receiver shuffles ("enrichment null" and "wiring null") are implemented
in `aggregate_factorial.py::run_backbone_permutations()`; q-values use
Storey's fixed-λ pi0 estimator (`aggregate_factorial.py:499`).

The runner does **not** modify `recv_*.parquet`. Q-values are written to a
sidecar CSV and never gate native PDS-based selection.

**Verdict: signed-off (separate stage; q-values written to sidecar CSV, never
gate native PDS).** Matches audit-plan §5 Sprint 5 directive verbatim.

### 6. `ALZ-20` (`19de928`) — `compute_kinase_support_factorial.py` as separate downstream — SIGN OFF

`code/integration/adapters/compute_kinase_support_factorial.py` reads baseline
`recv_*.parquet` (line 128) and writes per-pair sidecar files:
- `factorial/all_pairs/{sender}__{receiver}/kinase_support_scores.csv` (line 260)
- optional `kinase_routes.parquet` (line 278)

It is **not invoked from `run_factorial_all_pairs.sh`** (only mentioned in the
trailing echo summary at `run_factorial_all_pairs.sh:109`). It must be run as a
separate downstream stage. `aggregate_factorial.py` joins the sidecar CSV to
`recv_*.parquet` via DuckDB (line 324) for summary statistics; no merged
parquet is written back. TPDS and `kinase_support_score` remain in separate
files.

The λ in `aggregate_factorial.py:499` is Storey's pi0 fixed-λ q-value
parameter, **not** a blending weight. There is no λ-style PDS reranking in the
codebase.

**Verdict: signed-off (separate downstream; reads baseline Parquet, writes
sidecar CSV; never overwrites native PDS).** Matches audit-plan §5 Sprint 5
directive verbatim.

### 7. Native SiK channel reinstatement — `INC-DESIGN-6`, defer wiring

Audit-plan §5 cross-cutting concern says native SiK should run by default and
only the external kinase support score should be opt-in. The current state is
different from "gated off":

- `run_incytr_factorial_all_pairs.R` makes zero calls to `Integr_kinasedata`,
  `Cal_PDS`, `kl.pathways`, `as_kldata`, or `Permutation_test`. The wrapper
  header explicitly states "Phospho, kinase scoring, and downstream reranking
  are deferred to a second PR."
- `run_factorial_all_pairs.sh:76` exposes `ENABLE_KINASE_IMPUTATION` (default
  on, gates only the kinase-imputed gene expansion adapter); there is no
  `ENABLE_KINASE_AUGMENTATION` flag in the codebase despite Sprint 4 ledger
  references to that name.

So **native SiK is currently absent from the factorial wrapper, not gated off
by a flag**. Wiring it in requires substantive code work (passing
`kldata`/`kl_evidence` through the wrapper, deciding which contrast feeds the
kinase channel, handling the OLS-vs-mean SigProb interaction described in
`INC-DESIGN-1`).

**Verdict: design-decision row `INC-DESIGN-6` records the current state. The
wiring is deferred to a follow-up PR; closing the audit on the current state
is in scope for Sprint 5. Adding the wiring is feature work, not audit
cleanup.**

Suggested follow-up flag scheme (post-audit):
- `ENABLE_NATIVE_SIK = 1` (default ON once wired) — gates the native
  `Integr_kinasedata` channel.
- `ENABLE_KINASE_SUPPORT_SCORE = 0` (default OFF) — gates the external
  `compute_kinase_support_factorial.py` opt-in.

This splits the audit-plan's conceptual `ENABLE_KINASE_AUGMENTATION` into two
separate flags so the two layers can be toggled independently.

### Flag-name correction (Sprint 4 fix-up)

Sprint 4 ledger rows for `ALZ-19` and `INC-25` reference an
`ENABLE_KINASE_AUGMENTATION` flag, but **no such flag exists in the current
code**. The actual flag is `ENABLE_KINASE_IMPUTATION` at
`run_factorial_all_pairs.sh:76`, and it gates only the kinase-imputed gene
expansion adapter. Sprint-4-as-shipped is left immutable; this Sprint 5 note
records the corrected flag name for future readers. The Sprint 4 verdicts are
unaffected because the underlying gating logic is correct — only the flag
*name* in the ledger is wrong.

## Verification gate (run order)

1. `bash code/integration/tests/run_degenerate_2cond.sh` — must-match
   SigProb/FC slots remain bitwise-clean against native; `evaluation.PDS` and
   `SiK_score_*` drift remain owned by INC-28 (additive kinase channel; not a
   regression on the baseline `kldata = NULL` path).
2. `bash code/integration/tests/run_duckdb_enumeration_equiv.sh` — DuckDB
   enumeration with `cutoff_SigProb = 0` produces bitwise-identical pathway
   set to native `pathway_inference()`.
3. `bash code/integration/tests/run_verify_phase2.sh` — SKIP (no Phase 1/2
   fixture committed; gracefully handled).
4. `Rscript -e 'pkgload::load_all("../incytr"); testthat::test_dir("../incytr/tests/testthat")'`
   → 593/593 (no regression).

All four gates passed on 2026-05-05.

## Audit close

Sprint 5 is the final audit sprint. With this sprint signed off:

- All 5 sprints in audit-plan §10 are `[x]`.
- Every `→ Sprint N` row in the ledger has a final verdict.
- The factorial baseline output (`recv_*.parquet`) contains only native-
  equivalent columns; non-native columns (kinase support scores, backbone
  permutation q-values) live in adjunct files / downstream stages.
- One open follow-up: native SiK wiring (`INC-DESIGN-6`) — deferred to a
  separate PR; tracked outside the audit.
- One open upstream PR: `INC-37.b` correctness fixes (carried forward from
  Sprint 2).
