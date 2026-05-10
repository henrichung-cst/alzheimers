INCYTR INTEGRATION ARCHIVE
==========================

Provenance: relocated from /home/hchung/Projects/work/alzheimers/code/integration/
            on 2026-05-08 as part of an alzheimers-repo cleanup sprint;
            moved into archive/incytr_integration/ on 2026-05-10
            (alongside archive/deconvolution/) when the new wrapper landed.

WHY THIS EXISTS
---------------
The alzheimers repo is being contracted. Per docs/incytr_remediation_plan.md
in that repo, the Incytr integration layer is being torn out and replaced
with a thin AD-specific shell that calls the incytr R package directly.

That remediation work is in progress (Phase 1 stubs already exist back in
code/integration/). To get the non-stub source code out of the alzheimers
repo without losing it, the legacy tree was moved here.

WHAT'S HERE
-----------
adapters/                       — Python data adapters (snRNA/kldata/MEA/phospho
                                  exports + cross-pair aggregation). The
                                  remediation plan flags these as
                                  "materialized derivations" slated for
                                  deletion in favor of SQL views.
wrappers/                       — R wrappers. The remediation plan
                                  identifies these as a 1500-line shadow
                                  fork: library(Incytr) is loaded then zero
                                  exported functions are called; sigprob,
                                  scFC, factorial OLS, and evaluation are
                                  all reimplemented locally.
sidecar/kinase_pack/            — substrate-based kinase support scoring
sidecar/backbone_perms/         — backbone dual-null permutation runner
tests/                          — R + Python tests of the old surface,
                                  including tests/edge_pruning/
incytr_runtime.sh               — INCYTR_LAYER_* registry (shell side;
                                  R-side mirror is in wrappers/)
run_all_pairs.sh
run_factorial_all_pairs.sh
run_factorial_baseline_AB.sh
run_factorial_memory_gated.sh
run_imputation_verification.sh

NOT MOVED
---------
- config_integration.py        kept in alzheimers; the remediation plan
                               retains it
- factorial.R, load.R,         these are the in-flight Phase 1 stubs;
  persist.R, views.sql,        they stay in alzheimers as the future
  run_factorial.sh             entry point
- intermediates/               1.9 GB of gitignored pipeline output;
                               orphaned but harmless, left in place

USING THIS
----------
Need to grep the legacy implementation for context? It's all here.
Need a file back to finish remediation? `mv` it back to
~/Projects/work/alzheimers/code/integration/. No git history was
rewritten — both the source repo and the archive are clean trees.
