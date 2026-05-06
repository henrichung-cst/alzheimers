# Incytr pre-audit divergence report

Comparison of slot-by-slot output between **upstream-native** Incytr at commit `93b9881`
(tree-identical to current `upstream/master` HEAD `2a94051`) and **current `incytr-audit` HEAD**,
both run on the synthetic harness defined in `tests/testthat/helper-golden.R`
(seed=42; 12 genes × 40 cells; 2 cell types × 2 conditions).

Source script: `tests/testthat/compare_golden_outputs.R`. Native run used a
parallel `helper-golden.R` adapted for native API (explicit `gene.use_Sender`/
`gene.use_Receiver`, `slot_or_null` for slots not present in native).

## Summary

| Slot | Status |
| --- | --- |
| `expr_bygroup` | structural diff (list inequality, both length 2) |
| `sigprob` | **match** |
| `evaluation` | dim 3×7 → 3×10; new cols `Ack_score`, `KGG_score`, `Rme1_score`; `PDS` max\|diff\|=8.87e-02 |
| `kl_pathways` | `SiK_*_EI_*` non-numeric inequality (6 cols); `SiK_score_*` max\|diff\|≈1.78e-01 |
| `kl_evidence` | added in current HEAD (absent in native) |
| `EI` | structural diff (list inequality, both length 2) |
| `sc_FC` | **match** |
| `p_value` | **match** |
| `pr_FC` | **match** |
| `ps_FC` | **match** |
| `py_FC` | **match** |
| `Ack_FC` | added in current HEAD |
| `KGG_FC` | added in current HEAD |
| `Rme1_FC` | added in current HEAD |

## Per-slot detail

### `expr_bygroup` — structural diff
List of length 2 in both, but element-level inequality (likely column ordering or
factor-level changes between native and refactored `Expr_bygroup`). To resolve in
Sprint 1: dump both lists, diff column names + factor levels per condition.

Candidate commits (from `incytr_audit_commit_list.md`):
- `4a9423c` phase 0.4: relocate expression and EM helpers
- `c4698ef` phase 0.2: generalize integrate_omics
- `719c2b1` phase 1.1: generalize barcodes_bycondition

### `sigprob` — match
Native and current produce bitwise-identical `SigProb` data frames on the synthetic
two-condition input. This anchors Sprint 1: the per-animal/`Contrast_SigProb`
extension is invoked only when `mode == "factorial"`; the legacy two-condition path
is preserved without numerical drift.

### `evaluation` — added scoring columns + non-zero `PDS` drift
Three new columns (`Ack_score`, `KGG_score`, `Rme1_score`) — corresponds to the
`Ack`/`KGG`/`Rme1` PTM tracks and pY parallel work. `PDS` differs by up to
~0.089, which under a `[0,1]`-bounded score is material.

Candidate commits:
- `e09105a` phase 3: generalize TPDS/PDS pipeline
- `c8e764e` phase 1.6: add N-condition integration
- `abde752` add EM promiscuity weighting and edge … _(likely the PDS drift driver)_
- `c3580fc` apply cutoff to all omics slots

### `kl_pathways` — `SiK_*` columns drifted
Both `SiK_score_condA/B` differ by ~0.18 (large fraction of native magnitude), and
six `SiK_*_EI_*` columns are non-numeric-inequal — indicating type or category
changes. Most likely driven by the kinase-evidence refactor in
`build_kinase_evidence` / `attach_kinase_evidence`.

Candidate commits:
- `ca6a96e` phase 1.3: generalize kinase scoring
- `d6d5c8c` kinase library adaptor
- `4e2e672` phase 0.1: refactor Integr_multiomics

### `kl_evidence` — new slot in current HEAD
Native has no `kl.evidence` slot; current HEAD attaches per-(Path, Kinase) activity
rows including external `kinase_library_score`. This is the home of the kinase
augmentation channel that is currently conflated with the native SiK channel via
`ENABLE_KINASE_AUGMENTATION`. Sprint 5 territory.

### `EI` — structural diff
Native and current both produce length-2 lists, but element contents differ.
Likely tied to `Cal_EI` rewrite + condition-naming standardization (`3954fef`).

### `Ack_FC`, `KGG_FC`, `Rme1_FC` — new slots
Acetylation/glycation/methylation tracks added past the upstream baseline — these
exist independently of the factorial extension and need explicit C-bucket
justification (or revert).

Candidate commits:
- `c3580fc` apply cutoff to all omics slots and f…
- `4e2e672` phase 0.1: refactor Integr_multiomics
- `0adcca8` (alzheimers) feat: add tyrosine phospho (pY) track parallel

## Implications for Sprint 1 (factorial-extension audit)

1. `sigprob`, `sc_FC`, `p_value`, `pr_FC`, `ps_FC`, `py_FC` all match bitwise on the
   two-condition synthetic input — Sprint 1 can lean on these as a known-good
   regression anchor while introducing factorial-mode equivalents.
2. `evaluation.PDS` drift (~0.089) on a two-condition input means at least one of
   the **scoring formula** changes (EM promiscuity weight, `logi` parameters,
   `KPDS.weight` defaults) is active even outside factorial mode. This is the
   first concrete C-bucket candidate — Sprint 3 territory.
3. The new `Ack_FC`/`KGG_FC`/`Rme1_FC` slots, plus three new evaluation columns,
   indicate scope expansion outside the factorial mandate. Each gets its own
   ledger row.
4. `kl_evidence` introduces the external kinase-library augmentation channel
   alongside native SiK. Sprint 5 must split the `ENABLE_KINASE_AUGMENTATION`
   flag to control these independently.

## Files

- Native fixture: `../incytr/tests/testthat/fixtures/golden_native_93b9881.rds`
- Current fixture: `../incytr/tests/testthat/fixtures/golden_current_head.rds`
- Test fixture (alias): `../incytr/tests/testthat/fixtures/golden_output_v1.rds`
- Comparison script: `../incytr/tests/testthat/compare_golden_outputs.R`
- Native generator (worktree): `/home/hchung/Projects/work/incytr-93b9881/tests/testthat/generate_golden_native.R`
