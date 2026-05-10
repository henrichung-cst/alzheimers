> **Archived 2026-05-09.** This document covers the legacy shadow-fork integration code at `alz/integration/{wrappers,adapters,sidecar,tests}/` + orchestrator shells, all relocated to `~/Projects/work/incytr_integration_archive/` on 2026-05-08. Forward-looking guidance lives in `docs/incytr_remediation_plan.md`.

# Incytr Audit — Commit List

Tentative bucket per commit (A = factorial extension, B = performance, C = discretionary, TBD = not yet assigned).
Detailed justification + final disposition tracked in `incytr_audit_ledger.md`.

## `../incytr` repo — `1e64f41..HEAD` (40 commits)

Reference upstream pin: `93b9881` (tree-identical to current `upstream/master` HEAD `2a94051`).

- [ ] `44d3008` 2026-04-15 21:54:40 -0400 — simplify: address code review findings _bucket: TBD_
- [ ] `e09105a` 2026-04-15 21:46:10 -0400 — phase 3: generalize TPDS/PDS pipeline... _bucket: TBD_
- [ ] `9d888ea` 2026-04-15 20:55:48 -0400 — simplify: address code review findings _bucket: TBD_
- [ ] `388b6b7` 2026-04-15 20:48:33 -0400 — update NAMESPACE with factorial mode ... _bucket: TBD_
- [ ] `ff123e0` 2026-04-15 20:28:20 -0400 — phase 2.5: wire factorial pipeline an... _bucket: TBD_
- [ ] `ad4042e` 2026-04-15 20:26:10 -0400 — phase 2.4: add Contrast_SigProb with ... _bucket: TBD_
- [ ] `84f258f` 2026-04-15 20:20:11 -0400 — phase 2.2-2.3: add per-animal express... _bucket: TBD_
- [ ] `4a05aae` 2026-04-15 20:15:28 -0400 — phase 2.1: add factorial S4 slots and... _bucket: TBD_
- [ ] `c88f479` 2026-04-15 18:14:45 -0400 — simplify: address code review findings _bucket: TBD_
- [ ] `c8e764e` 2026-04-15 15:46:05 -0400 — phase 1.6: add N-condition integratio... _bucket: TBD_
- [ ] `bb0c991` 2026-04-15 15:44:13 -0400 — phase 1.5: generalize permutation out... _bucket: TBD_
- [ ] `d7f3254` 2026-04-15 15:42:04 -0400 — phase 1.4: generalize evaluation and ... _bucket: TBD_
- [ ] `ca6a96e` 2026-04-15 15:40:36 -0400 — phase 1.3: generalize kinase scoring ... _bucket: TBD_
- [ ] `75bdd0d` 2026-04-15 15:36:53 -0400 — phase 1.2: generalize Cal_SigProb and... _bucket: TBD_
- [ ] `719c2b1` 2026-04-15 15:34:59 -0400 — phase 1.1: generalize barcodes_bycond... _bucket: TBD_
- [ ] `a2e99fd` 2026-04-15 15:09:06 -0400 — simplify: address code review findings _bucket: TBD_
- [ ] `6da9bcc` 2026-04-15 15:02:01 -0400 — phase 0.6: update S4 class slot docum... _bucket: TBD_
- [ ] `3954fef` 2026-04-15 15:01:09 -0400 — phase 0.5: standardize condition nami... _bucket: TBD_
- [ ] `c4698ef` 2026-04-15 14:54:00 -0400 — phase 0.2: generalize integrate_omics... _bucket: TBD_
- [ ] `4e2e672` 2026-04-15 14:51:53 -0400 — phase 0.1: refactor Integr_multiomics... _bucket: TBD_
- [ ] `4a9423c` 2026-04-15 14:48:04 -0400 — phase 0.4: relocate expression and EM... _bucket: TBD_
- [ ] `ad7462b` 2026-04-15 14:45:21 -0400 — phase 0.3: add tests for untested exp... _bucket: TBD_
- [ ] `b50c2cc` 2026-04-15 14:42:50 -0400 — phase 0.0: add golden output fixture ... _bucket: TBD_
- [ ] `4aa30c8` 2026-04-15 14:24:05 -0400 — clean up codebase: remove dead code, ... _bucket: TBD_
- [ ] `abde752` 2026-04-10 14:55:24 -0400 — add EM promiscuity weighting and edge... _bucket: TBD_
- [ ] `b654fdd` 2026-04-10 14:25:41 -0400 — add condition-label permutaiton test ... _bucket: TBD_
- [ ] `f922c3a` 2026-04-10 14:21:42 -0400 — add condition-label permutation test ... _bucket: TBD_
- [ ] `6858063` 2026-04-10 00:22:17 -0400 — memory permutation pass _bucket: TBD_
- [ ] `d6d5c8c` 2026-03-27 18:46:28 -0400 — kinase library adaptor _bucket: TBD_
- [ ] `c3580fc` 2026-03-27 17:41:51 -0400 — apply cutoff to all omics slots and f... _bucket: TBD_
- [ ] `847a014` 2026-03-10 13:43:03 -0400 — docs: Add docs/notes to .gitignore to... _bucket: TBD_
- [ ] `5031094` 2026-03-10 13:04:40 -0400 — add env.yml _bucket: TBD_
- [ ] `650bca1` 2026-03-09 13:36:38 -0500 — Remove CLAUDE.md from tracking and ad... _bucket: TBD_
- [ ] `b2a4c7f` 2026-03-09 13:34:18 -0500 — Organize docs into reference and note... _bucket: TBD_
- [ ] `e04bf18` 2026-03-06 14:06:43 -0600 — Improve evaluation/kinase code organi... _bucket: TBD_
- [ ] `76becbd` 2026-03-06 12:22:12 -0600 — Remove dplyr dependency, replace all ... _bucket: TBD_
- [ ] `c1b6fb9` 2026-03-05 22:54:28 -0600 — Optimize performance, fix correctness... _bucket: TBD_
- [ ] `7e6735d` 2026-03-05 22:04:36 -0600 — Refactor codebase: split monolith, de... _bucket: TBD_
- [ ] `90c60e8` 2026-03-05 12:51:04 -0600 — Reorganize project directory structure _bucket: TBD_
- [ ] `6eeeac5` 2026-03-05 12:28:23 -0600 — Update documentation for development ... _bucket: TBD_

## `alzheimers` repo — `alz/integration/` history (21 commits)

- [ ] `8ca0a64` 2026-05-04 13:22:24 -0400 — checkpoint: subclass-attribution branch bef... _bucket: TBD_
- [ ] `0adcca8` 2026-04-28 17:49:34 -0400 — feat: add tyrosine phospho (pY) track paral... _bucket: TBD_
- [ ] `cc10112` 2026-04-21 18:01:58 -0400 — pipeline: complete Unit 6.4 retire legacy a... _bucket: TBD_
- [ ] `b9f302b` 2026-04-21 16:15:16 -0400 — pipeline: complete Unit 6.2 retire backbone... _bucket: TBD_
- [ ] `5bab404` 2026-04-21 16:12:42 -0400 — pipeline: complete Unit 6.1 retire old viewers _bucket: TBD_
- [ ] `2ab47fb` 2026-04-20 19:23:01 -0400 — pipeline: lossless edge sharding for unifie... _bucket: TBD_
- [ ] `e096f36` 2026-04-20 18:18:18 -0400 — update edges _bucket: TBD_
- [ ] `99ab2b3` 2026-04-20 13:37:39 -0400 — pipeline: complete Unit 1.3 build_edge_inde... _bucket: TBD_
- [ ] `97616a6` 2026-04-16 12:51:03 -0400 — Factorial permutation tests with within-rec... _bucket: TBD_
- [ ] `2dda892` 2026-04-15 10:12:11 -0400 — brain atlas data compression _bucket: TBD_
- [ ] `f6e2b98` 2026-04-14 22:08:52 -0400 — Simplify permutation code after review _bucket: TBD_
- [ ] `9cf31de` 2026-04-14 22:00:03 -0400 — Backbone-level permutation tests and per-pa... _bucket: TBD_
- [ ] `019334b` 2026-04-14 21:21:20 -0400 — Cross-pair aggregation and Parquet-only cle... _bucket: TBD_
- [ ] `8cf4366` 2026-04-14 18:13:32 -0400 — Update integration README for Phase 2 recei... _bucket: TBD_
- [ ] `8679681` 2026-04-14 18:11:02 -0400 — Vectorized receiver scoring: Phase 2 of all... _bucket: TBD_
- [ ] `291fa00` 2026-04-14 13:52:10 -0400 — Receiver-centric backbone enumeration: Phas... _bucket: TBD_
- [ ] `98d2b9d` 2026-04-14 13:10:11 -0400 — remove dead code _bucket: TBD_
- [ ] `3e688bf` 2026-04-14 10:33:41 -0400 — All-pairs Incytr pipeline: DuckDB enumerati... _bucket: TBD_
- [ ] `9f339ed` 2026-04-13 15:38:36 -0400 — Kinase-imputed pathway expansion: admit rec... _bucket: TBD_
- [ ] `19de928` 2026-04-13 14:27:47 -0400 — Substrate-based external reranking with med... _bucket: TBD_
- [ ] `5acec95` 2026-04-09 23:11:30 -0400 — Phase 1 Incytr integration: adapters, R wra... _bucket: TBD_
