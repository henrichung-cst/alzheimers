# Incytr Pipeline Remediation Plan

> **Heads-up to the implementing engineer (2026-05-08):** the legacy `wrappers/`, `adapters/`, `sidecar/`, and `tests/` trees described below have been **physically relocated** out of the alzheimers repo to `~/Projects/work/incytr_integration_archive/` as part of a separate cleanup sprint. The Phase 1 stubs (`factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh`) and `config_integration.py` remain at `alz/integration/`. Nothing was deleted — `mv` files back if you need them in-tree to finish the rewrite. See `alz/integration/MOVED.txt` for the manifest.

**Status:** Required, not optional. This is a structural error, not a refactor opportunity.
**Audience:** Senior engineer assigned to implement.
**Authorization:** No production dependencies. No backwards-compatibility shims. No legacy code retention. Delete with prejudice.

**Implementation update, 2026-05-08:** Phase 1/2 remediation has started but is not complete, and the first implementation attempt exposed a major flaw in this plan. The sibling `incytr` worktree previously had a `run_factorial_all_pairs()` scaffold, `Export_results(..., format = c("long", "wide"))`, docs, and focused tests for long export/all-pairs orchestration. That scaffold was the wrong abstraction. A managed-environment 1x1 smoke attempt (`PAIR_FILTER="Microglia-PVM:L5 IT"`) proved that looping the package's single-pair path is unsafe: it performs unrestricted pathway enumeration and can exhaust system memory even for a single sender/receiver pair. The repo entry point must require a production package API before loading data; it must not bind the architecture to a temporary API name.

**Planning correction, 2026-05-08:** Guard-only APIs, stop-before-crash scaffolds, "native all-pairs" loops, and blind ports of wrapper phases are not remediation work. They are distraction unless they are immediately replaced by the production engine. The plan below supersedes those constructs. It defines required package capabilities and forbidden implementation shapes; it does not require preserving the wrapper's internal decomposition, temporary APIs, or compatibility shims.

**Immediate corrective action:** The interim `run_factorial_receivers()` guard/scaffold has been removed from the accepted implementation path. Do not reintroduce tests, docs, wrappers, or plan milestones that make that scaffold look like a valid intermediate deliverable. The only valid next package work is the real production engine.

---

## 1. Diagnosis

The factorial Incytr pipeline in this repository is architecturally broken. The breakage has been hidden by misleading naming for long enough that participants in the codebase reason about the system incorrectly. Before any work begins, the implementing engineer must internalize the actual state.

### 1.1 The wrapper does not wrap

`alz/integration/wrappers/run_incytr_factorial_all_pairs.R` calls `library(Incytr)` at line 140 and **calls zero exported incytr functions afterward**. Verified by exhaustive grep for `Cal_SigProb`, `Cal_PDS`, `Cal_foldchange`, `Cal_scFC`, `Pathway_evaluation`, `pathway_inference`, `Export_results`, `run_factorial`, `Kinase_exploration`, `Find_highexp_gene`, `Integr_kinasedata`, `Merge_results`, `Permutation_test`, `create_Incytr`, `Infer_5step_pathways`, `Generate_indicator`, `Expr_bygroup`, `Contrast_SigProb`, `object_update` across the file. The four matches that exist are all comments saying "Mirror of …", "matching …", "parity per …".

The package is loaded for kinase database tables and nothing else. Every numerical step — sigprob, scFC, factorial OLS, evaluation, kinase scoring — is **reimplemented** in:

- `alz/integration/wrappers/receiver_scoring.R::compute_sigprob_vectorized`
- `alz/integration/wrappers/receiver_scoring.R::compute_scfc_vectorized`
- `alz/integration/wrappers/run_incytr_factorial_all_pairs.R:649` (the local `Pathway_evaluation` mirror)
- `alz/integration/wrappers/duckdb_enumeration.R` (replaces `Infer_5step_pathways`)

This is a shadow fork. Comments saying "matching upstream" are aspirational, not enforced. There is no test that pins the wrapper's output against `incytr::run_factorial()`.

### 1.2 The output schema is also forked

The wrapper assembles its 503-column wide-by-contrast output schema by hand at `run_incytr_factorial_all_pairs.R:1280–1300`, reproducing the column-suffix convention that `Export_results()` factorial branch (`incytr/R/evaluation.R:251–298`) would have produced. Two implementations of the same schema, neither tied to the other.

### 1.3 The "adapter" layer materializes derivations as primary outputs

`alz/integration/adapters/aggregate_factorial.py` produces seven CSV files:

- `backbone_provenance.csv`
- `contrast_comparison.csv`
- `temporal_dynamics.csv`
- `hub_matrix_by_contrast.csv`
- `kinase_tpds_integration.csv`
- `backbone_recurrence_by_contrast.csv`
- `target_convergence_by_contrast.csv`

Every one is a `GROUP BY` rollup over the per-receiver parquet (with one optional join). Three of them — `contrast_comparison`, `temporal_dynamics`, `hub_matrix_by_contrast` — share the same key (`sender`, `receiver`, `contrast`) and overlapping metric columns. They are partial duplicates of each other.

`backbone_provenance.csv` lives under `aggregation/` (suggesting cross-pair rollup) but its schema contains no sender column — it is per-receiver data, regenerated 22 times per receiver as the pair loop iterates.

`reranking_summary.json` is written **once per pair**, producing 462 single-record JSON files at full scope.

### 1.4 Net effect

A pipeline named "incytr integration" where:

- Incytr is a kinase database dependency, not a computation.
- The "wrapper" is a 1500-line fork.
- The "adapters" materialize derivations of the fork's output as if they were primary data.
- Cross-file consistency is enforced by hand-coordinated comments saying "matching upstream."

This cannot be patched. It must be torn out.

---

## 2. Goal

**Eliminate shadow forks and the materialized-derivations layer.**

When the pipeline runs incytr, it must call incytr. The wrapper must be a wrapper — thin, AD-specific, with no math. Every primary output must come from a function in the incytr package. Every derived output must be a SQL view, not a materialized file. Every file in `alz/integration/` must serve exactly one of: data loading, persistence, or orchestration.

At completion:

- `git grep -E "Mirror of|matching upstream|parity per|reimplemented from" alz/integration/` returns zero results.
- `git grep "library(Incytr)" alz/integration/` returns exactly one R file.
- No file in `alz/integration/` reimplements math defined in `incytr/R/`.
- No CSV is written by any pipeline step. Parquet only.
- No JSON-per-pair file is written. Runtime metadata is one parquet appended per pair.
- Aggregation tables are SQL views in one file, queried on demand.

---

## 3. Non-goals

- **Backwards compatibility.** No deprecation flags. No legacy entrypoints. No "_v2" suffixes. The new code replaces the old code in the same PR.
- **Snapshot preservation.** Pinned baselines under `outputs/reports/incytr_baseline_AB/` and intermediates under `alz/integration/intermediates/factorial/all_pairs/` are deleted. Recover from git history if needed.
- **Performance retuning.** Performance work happens upstream where the math lives. The wrapper does not optimize.
- **Scope expansion.** Cell-type vocabulary, contrast design, AD data loaders are unchanged in semantics.

---

## 4. Architecture target

There is one boundary:

| Concern | Owner |
|---|---|
| Math, statistical models, schema of returned data, package-level orchestration | `incytr` package (`/home/hchung/Projects/work/incytr/`) |
| AD-specific data loading, persistence layout, cross-pair derivations as views, run-level orchestration | this repo (`alz/integration/`) |

After remediation, `alz/integration/` looks like this and contains nothing else:

```
alz/integration/
├── README.md                       # describes the new architecture, no historical notes
├── config_integration.py           # paths and thresholds (kept, possibly trimmed)
├── factorial.R                     # ~200-line entry point — calls incytr exclusively
├── load.R                          # AD-specific data loaders (expression, PTM, design)
├── persist.R                       # parquet writer + hive-partition layout helpers
├── views.sql                       # all derivations defined once
└── run_factorial.sh                # one-line shell entry: Rscript alz/integration/factorial.R
```

The directories `wrappers/`, `adapters/`, `intermediates/`, `sidecar/`, and `tests/edge_pruning/` cease to exist.

The persistence layout under `outputs/reports/incytr_factorial/`:

```
outputs/reports/incytr_factorial/
├── receiver_cache/
│   └── receiver=<R>/data.parquet         # long-form Export_results output
├── routes/
│   └── sender=<S>/receiver=<R>/data.parquet  # MEA-joined kinase route data (if retained)
├── pair_metadata.parquet                 # 1 row/pair: timing, status, n_pre, n_post
└── views.sql                             # symlink or copy of alz/integration/views.sql
```

No CSV. No per-pair JSON. No per-pair directory of unstructured artifacts.

---

## 5. Phases

Phases must be merged in strict order. The repo at the end of each phase is fully working — no half-states left between phases.

### Phase 1 — Build the production incytr engine

Work against `/home/hchung/Projects/work/incytr/`. Each PR must leave the package closer to the final production engine. Do not merge scaffolds whose only value is preventing execution or reserving an API name.

**Critical clarification, 2026-05-08:** Phase 1 does **not** mean "call the existing one-pair Incytr pipeline 462 times." That interpretation is rejected. The old single-pair sequence (`create_Incytr()` → `pathway_inference()` → `run_factorial()`) materializes an unrestricted cartesian pathway search space and is not a production algorithm for AD-scale all-pairs analysis. The shadow wrapper avoided this by using constrained pathway construction, expression-aware pruning, batched sender scoring, and receiver-level persistence. Those are required capabilities, not proof that the wrapper's internal phases or file structure belong in the package.

At the end of Phase 1, `incytr` must expose a production factorial engine for multi-sender, multi-receiver analysis. If the clean implementation is receiver-centric, that is an execution strategy only: sender/receiver pair semantics are preserved, and `sender` remains an output column. The package may keep single-pair APIs for interactive use, but the production API consumed by this repo must not be implemented as a loop around `pathway_inference()`.

**P1.1 — Define the production contract.** Before implementation, specify the package-level contract in roxygen and tests:
- inputs: expression, metadata, design, contrasts, senders, receivers, pathway database, optional omics/kinase inputs;
- outputs: one primary long-form result table with sender, receiver, pathway, contrast, scores, diagnostics, and metadata needed by downstream views;
- streaming/persistence boundary: the package can return receiver chunks or call a writer, but the repo must not reconstruct score math;
- safety contract: the package must never materialize an unbounded candidate universe before pruning or estimating scale.

This PR must remove or fail any placeholder implementation that satisfies the name but not the behavior.

**P1.2 — Package-owned pathway construction.** Implement scalable pathway construction in package terms. The required behavior is:
- use expression and database constraints before expensive joins;
- avoid unrestricted `L1 × L2 × L3` materialization;
- apply pathway viability filters as early as mathematically valid;
- produce the same sender/receiver/pathway semantics as the current accepted output.

Do not blindly port "Phase A/B" as a named package concept. If splitting receiver-side backbone construction from sender ligand attachment is the cleanest implementation, use it internally. If a graph/query abstraction is cleaner, use that. The package API must describe biological/statistical semantics, not wrapper mechanics.

**P1.3 — Production scoring.** Move the scoring/evaluation behavior now duplicated in the wrapper into package-owned functions. The required behavior is:
- compute SigProb, scFC, factorial contrasts, evaluation scores, and PDS/KPDS with the same documented semantics as accepted current outputs;
- score all relevant senders for a receiver/batch without reconstructing per-pair S4 objects as the production path;
- keep numerical tolerances explicit in tests.

**P1.4 — Execution engine.** Implement the production engine behind the contract from P1.1. It must not be implemented as:

```r
for (receiver in receivers) {
  for (sender in senders) {
    create_Incytr(...)
    pathway_inference(...)
    run_factorial(...)
  }
}
```

Required properties:
- no production path calls `pathway_inference()` for AD-scale multi-sender/multi-receiver factorial work;
- no production path creates one full S4 object per sender/receiver pair;
- shared work is actually shared across senders/receivers where the algorithm makes that possible;
- large intermediate state is bounded, streamed, or query-backed rather than retained as a nested object graph;
- failure happens before expensive materialization when input scale exceeds configured limits.

Do not add an API whose only useful behavior is "refuse to run." Refusing an unsafe path is acceptable only while deleting it or replacing it in the same PR.

**P1.5 — Output schema.** The package owns the primary result schema. The output consumed by this repo is long form: one row per `(sender, receiver, Path, contrast)` with metric columns. Do not create a second hand-built schema in the repo. Retain wide output only if it already exists as a package contract and does not complicate the production path.

**P1.6 — Scale and parity tests in the package.** Add package tests that cover both correctness and production-relevant boundedness:
- a small canonical parity fixture that runs the production API end-to-end and asserts column-by-column equality (within documented numerical tolerance) against frozen reference output committed under `incytr/tests/testthat/_reference_output/factorial_smoke.parquet`;
- a high-degree synthetic DB fixture proving the engine prunes, bounds, or query-executes without materializing the unsafe candidate universe;
- an execution-structure fixture proving shared work is not repeated once per sender/receiver pair;
- a streaming/chunked-output fixture proving results can be emitted without retaining all pair objects.

These tests live in the package and stay forever. They are the contract.

**Acceptance for Phase 1:** the production package API is implemented, documented, and covered by parity and scale tests. `R CMD check` passes. No production package path used by this repo loops the single-pair API over all sender/receiver pairs. No accepted PR consists only of a guard, placeholder, name reservation, or compatibility shim.

### Phase 2 — Build the new thin wrapper

In **one commit** in this repo:

1. Write `alz/integration/factorial.R`. Approximate shape (~200 lines including I/O, argparse, logging):

   ```r
   library(Incytr)
   library(arrow)
   source("alz/integration/load.R")
   source("alz/integration/persist.R")

   args <- parse_args(commandArgs(trailingOnly = TRUE))
   inputs <- load_ad_factorial_inputs(args$config)
   contrasts <- build_factorial_contrasts(inputs$animal_meta)

   Incytr::<production_factorial_api>(
     expression = inputs$expr,
     metadata   = inputs$meta,
     ptm        = inputs$ptm,
     senders    = inputs$senders,
     receivers  = inputs$receivers,
     contrasts  = contrasts,
     callback   = function(receiver, results, metadata) {
       write_receiver_parquet(results, receiver, out_dir = args$out_dir)
       append_pair_metadata(metadata, out_dir = args$out_dir)
     }
   )
   ```

2. Write `alz/integration/load.R`. AD-specific data loading. ~150 lines. No math.

3. Write `alz/integration/persist.R`. Hive-partition writers + categorical encoding. ~80 lines.

4. Write `alz/integration/views.sql`. Defines:
   - `backbone_provenance` (per-receiver `SELECT DISTINCT`)
   - `contrast_comparison` (GROUP BY pair × contrast)
   - `temporal_dynamics` (parses contrast string, same group-by)
   - `hub_matrix_by_contrast` (same group-by, different metrics)
   - `kinase_tpds_integration` (joins routes ⋈ receiver_cache)
   - `backbone_recurrence_by_contrast` (cross-pair)
   - `target_convergence_by_contrast` (cross-pair)

   These are CREATE VIEW statements. They are not materialized.

5. Write `alz/integration/run_factorial.sh`. One-line invocation:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   exec Rscript alz/integration/factorial.R "$@"
   ```

6. Update `pixi.toml` so `pixi run incytr-factorial` calls `alz/integration/run_factorial.sh`.

The wrapper must not contain sender/receiver nested scoring loops. Pair filtering is allowed only as input selection passed to the package production engine.

**Acceptance for Phase 2:** `pixi run incytr-factorial` produces `outputs/reports/incytr_factorial/receiver_cache/receiver=L5_IT/data.parquet` for the smoke receiver/pair selection (`Microglia-PVM:L5 IT`) through the production package API. The output is column-equivalent to the pre-migration wrapper output (run on the same input, frozen reference) within documented tolerance. The run log must show package-owned constrained pathway construction and must not show `pathway_inference()` cartesian joins.

### Phase 3 — Decide what survives in the Python adapter layer

Every file in `alz/integration/adapters/` either justifies itself in writing or is deleted in Phase 5. The implementing engineer writes one paragraph per surviving adapter answering: *what does the new R wrapper not do that this script does, and which downstream consumer (named) needs it?*

**Default-keep candidates (subject to the paragraph test):**

- `compute_kinase_support.py` — joins external MEA enrichment (`outputs/reports/kinase_attribution/mea_stoichiometry.csv`) with kinase-library route data. MEA enrichment is the live pipeline's output, not incytr's. This is genuinely outside the package's scope.

**Default-delete (no paragraph defense available):**

- `aggregate_factorial.py` — every output becomes a view (Phase 2 step 4).
- `aggregate_cross_pair.py` — same.
- `compute_kinase_support_all_pairs.py` — orchestration; the new R wrapper handles iteration.
- `examine_factorial.py` — diagnostic redundant with views.
- `export_kl_output_factorial.py`, `export_kl_output.py`, `export_multiomics_evidence_factorial.py`, `export_kinase_imputed_genes_factorial.py`, `export_kinase_imputed_genes.py`, `export_expression_factorial.py`, `export_expression.py`, `export_phospho.py`, `export_kldata.py` — all data movement scripts written to feed the shadow-fork wrapper. The new architecture loads data inside `alz/integration/load.R` directly. These scripts have no consumer.
- `normalization.py` — if the math is upstream-equivalent, it dies; if AD-specific, it moves to `alz/integration/load.R` or a small `normalization.py` that survives by name only.
- `build_edge_index.py` — verify whether the unified viewer consumes its output. If yes, the consumer is rewritten to query the new parquet store directly. The adapter dies.
- `common.py` — depends on what's left. Probably dies.

**Acceptance for Phase 3:** a written disposition for every file in `alz/integration/adapters/`. Surviving files are listed in the README. Everything else is queued for Phase 5 deletion.

### Phase 4 — Migrate downstream consumers

Anything that currently reads from `alz/integration/intermediates/factorial/all_pairs/` or `outputs/reports/incytr_baseline_AB/` is rewritten to read from `outputs/reports/incytr_factorial/`. Known consumers:

- `alz/build_unified_viewer.py` — already pruned of factorial backbone reads in the recent viewer cleanup; reverify there are no stragglers.
- Any `pixi.toml` task that references the old layout.
- Any `alz/runners/` script that references the old wrapper.
- Any documentation cross-link.

A consumer that cannot be migrated is a consumer that the new architecture does not support. Either fix the consumer or document why the feature is being dropped.

**Acceptance for Phase 4:** `git grep -E "wrappers/|adapters/|intermediates/factorial|incytr_baseline_AB"` returns zero hits in source code (matches in `archive/` are acceptable).

### Phase 5 — Delete with prejudice

In **one PR**, after Phases 1–4 are merged and the new pipeline is producing correct output:

```bash
git rm -r alz/integration/wrappers/
git rm -r alz/integration/adapters/                 # except files that survived Phase 3
git rm -r alz/integration/sidecar/
git rm -r alz/integration/tests/edge_pruning/
git rm -r alz/integration/intermediates/
git rm    alz/integration/run_all_pairs.sh
git rm    alz/integration/run_factorial_all_pairs.sh
git rm    alz/integration/run_factorial_baseline_AB.sh
git rm    alz/integration/run_factorial_memory_gated.sh
git rm    alz/integration/run_imputation_verification.sh
git rm    alz/integration/incytr_runtime.sh
git rm -r outputs/reports/incytr_baseline_AB/
```

In the same PR, rewrite (do not annotate, do not deprecate):

- `alz/integration/README.md`
- `docs/integrations/kinase_incytr_integration.md`
- `docs/integrations/incytr_audit_ledger.md`
- `CLAUDE.md` — sections describing `wrappers/`, `adapters/`, the seven aggregation tables, the dual-track `run_factorial_all_pairs.sh` path, the list of "supporting prerequisites" that reference deleted scripts.
- `docs/INDEX.md` — drop dead links.
- `docs/foundation/repo_surface_index.md` — regenerate.

References to deleted files are removed, not commented out. Documentation does not retain "this used to be …" notes; the architecture is now what the docs say it is.

**Acceptance for Phase 5:**
- `git grep` of every removed filename returns empty in the diff.
- The unified viewer build (`pixi run viewer`) succeeds against the new outputs.
- The `pixi run incytr-factorial` task runs end-to-end for at least one full all-pairs cohort.
- Repo size decreases by ≥3,000 lines of source code and ≥2 GB of intermediate artifacts.

---

## 6. Anti-patterns the implementing engineer must refuse

These are the shapes of compromise that recreate the shadow fork. Each is a way of preserving the broken state under a new name. Refuse all of them.

| Anti-pattern | Why it is wrong |
|---|---|
| "Add a flag to keep the old wrapper for now." | The old wrapper is the bug. Flags preserve bugs. |
| "Rename `wrappers/` to `legacy_wrappers/`." | A renamed shadow fork is still a shadow fork. |
| "Add a native all-pairs loop around the package's one-pair mode." | This is the failed abstraction. It repeats the unsafe single-pair cartesian pathway materialization and discards the receiver-centric algorithm that made the shadow wrapper tractable. |
| "Add a guard-only package API that mostly refuses to run." | Preventing crashes is necessary but not remediation. It does not produce the required analysis and creates false progress. |
| "Port the wrapper's Phase A/B structure as a required package abstraction." | The package needs scalable pathway construction, not the wrapper's internal staging vocabulary. Implementation structure must follow the cleanest package design. |
| "Expose receiver-centric batching as a biological model." | Receiver-centric batching is only an execution strategy. Sender/receiver pair semantics must remain explicit in the output. |
| "Add a parity test in this repo so we can keep the wrapper as a fast path." | Parity tests live in the package (P1.6). A repo-side parity test institutionalizes the fork. |
| "The new pipeline is slower; let's keep the vectorized adapter." | If it's slower, Phase 1 is incomplete. The vectorized math goes upstream. |
| "This aggregation table is small, materialize it just in case." | If nothing consumes it, it doesn't get written. Disk is not a substitute for queries. |
| "Add a deprecation warning and remove next quarter." | Remove now. There are no production consumers. |
| "Leave the files and add a README explaining they are deprecated." | Delete the files. Git history is the deprecation log. |
| "Keep the per-pair JSON files for debugging." | One parquet, one row per pair. Debugging is a SQL query, not a directory listing. |
| "Preserve the seven aggregation CSVs as fallback in case the views are slow." | Views over parquet on this scale resolve in milliseconds. There is no fallback case. |
| "Keep `incytr_runtime.sh` for the env-var configuration pattern." | The env-var configuration pattern is part of the bug — it routes config decisions away from the package. Configuration is function arguments. |

---

## 7. Validation checklist (signed off before merging Phase 5)

Current state as of 2026-05-08:

```text
[x] incytr worktree had a native run_factorial_all_pairs() scaffold, now classified as the wrong abstraction.
[x] incytr worktree has Export_results(format = c("long", "wide")).
[x] Focused incytr tests for long export/all-pairs orchestration passed for the rejected scaffold. Those scaffold tests were removed and are not Phase 1 acceptance evidence.
[x] alz/integration/factorial.R, load.R, persist.R, views.sql, run_factorial.sh exist.
[x] pixi.toml defines incytr-factorial -> alz/integration/run_factorial.sh.
[x] 2026-05-08 rerun: repo R files parse cleanly.
[x] 2026-05-08 rerun: temporary-library Incytr exposed the rejected `run_factorial_all_pairs()` scaffold and `Export_results(..., format = ...)`.
[x] 2026-05-08 safety update: `alz/integration/factorial.R` now fails before loading AD inputs while the production package API is missing. It still needs to be pointed at the final production package API.
[x] 2026-05-08 package cleanup: `run_factorial_all_pairs()` and `run_factorial_receivers()` scaffold APIs, exports, docs, and scaffold tests were removed from the package worktree.
[x] 2026-05-08 production-engine start: `incytr` source now exposes two explicit production primitives, `construct_factorial_paths()` and `score_factorial_paths()`. Candidate construction and factorial scoring are separate package responsibilities; no public network/orchestrator API is accepted. The current slice performs package-owned constrained candidate construction and vectorized per-animal SigProb/OLS scoring without `pathway_inference()` or per-pair S4 objects. Synthetic package tests cover single-pair parity, multi-sender output, and pre-scoring join guards. This is not Phase 1 acceptance yet: high-degree scale tests, package-owned multiomics/SiK parity, and full AD smoke validation remain open.
[ ] incytr exposes the final production factorial API with package-owned constrained pathway construction and batched scoring.
[ ] incytr production API performs package-owned gene admission, early pruning/filtering, bounded/query-backed joins, and scale guardrails.
[ ] New repo wrapper smoke run succeeds for PAIR_FILTER="Microglia-PVM:L5 IT".
[ ] Managed environment exposes R arrow/duckdb and updated local Incytr package.
[ ] Old wrappers/adapters/intermediates are deleted.
[ ] Downstream consumers and docs are migrated off the old layout.
```

Latest execution attempt, 2026-05-08:

- `which pixi` fails in the active shell; the managed task runner is unavailable on `PATH`.
- Active R libraries report `arrow = FALSE`, `duckdb = FALSE`, `DBI = FALSE`, `Incytr = TRUE`.
- `PAIR_FILTER="Microglia-PVM:L5 IT" bash alz/integration/run_factorial.sh` fails before pipeline execution with `Error in library(arrow) : there is no package called 'arrow'`.
- `git grep "library(Incytr)" alz/integration/` still returns legacy wrapper and edge-pruning test hits; the Phase 5 deletion gate remains closed.
- `git grep -E "Mirror of|matching upstream|parity per|reimplemented from" alz/integration/` still returns legacy wrapper comments; the Phase 5 deletion gate remains closed.
- A later managed-environment 1x1 smoke attempt reached native Incytr object creation and showed the implementation is operationally unsafe. Root cause from static review: `run_factorial_all_pairs()` calls `pathway_inference(obj, DB)` without gene-use filters, and Incytr expands null gene filters to all genes before cartesian pathway joins. Algebraic DB-degree estimation, without materializing the join, gives approximately `613,028,009` unrestricted candidate pathway rows for the bundled mouse DB before downstream filtering/evaluation. This must be fixed upstream before any further execution.

Final sign-off checklist:

```text
[ ] incytr/main contains all six revised PRs from Phase 1.
[ ] incytr/tests/testthat/test-factorial-parity.R passes.
[ ] incytr has tests proving production execution avoids repeated one-pair work and avoids unbounded candidate materialization.
[ ] alz/integration/factorial.R exists and is < 250 lines.
[ ] alz/integration/wrappers/ does not exist.
[ ] alz/integration/adapters/ contains only files with written justification.
[ ] No file in alz/integration/ contains the strings: "Mirror of", "matching upstream", "parity per", "reimplemented".
[ ] git grep "library(Incytr)" alz/integration/ returns exactly one match.
[ ] No CSV is written by alz/integration/ at runtime.
[ ] No JSON-per-pair file is written at runtime.
[ ] outputs/reports/incytr_factorial/views.sql exists; the seven legacy CSVs do not.
[ ] pixi run viewer succeeds.
[ ] pixi run incytr-factorial succeeds end-to-end on one full all-pairs run.
[ ] Repo line count drops by ≥3,000.
[ ] CLAUDE.md, docs/integrations/, docs/INDEX.md reflect the new architecture with no historical annotations.
```

---

## 8. Estimate

For an engineer holding both codebases, after the 2026-05-08 clarification that Phase 1 requires a production package engine rather than an all-pairs loop:

- Phase 1: 8–12 working days. The math and scalable pathway/scoring strategy are debugged in the wrapper; the work is turning the required behavior into proper package code with tests and guardrails, without preserving unnecessary wrapper constructs.
- Phase 2: 2–3 days.
- Phase 3: 1 day (mostly writing the dispositions).
- Phase 4: 1–2 days, depending on how many consumers turn up.
- Phase 5: 1 day, mostly review and verification.

Total: ~2 weeks of focused work. Not parallelizable across phases.

---

## 9. Final note

The state described in §1 was not produced by malice or incompetence. It is the natural outcome of a project that started as a wrapper around a slower package, hit a performance wall, and resolved the wall by reimplementing rather than upstreaming. Every individual decision that led here was locally reasonable. The aggregate is structurally indefensible.

The remediation must not repeat the same dynamic. If during Phase 1 the package owner pushes back on absorbing the vectorized math, the answer is to negotiate, not to keep a private fast path. If during Phase 2 a consumer needs an aggregation that the views don't expose, the answer is to add a view, not to write a CSV. Every shortcut available to the implementing engineer is a way of recreating the problem this document exists to eliminate.

The shadow forks and the reshapers go. With prejudice. No exceptions.
