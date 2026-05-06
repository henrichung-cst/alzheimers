# Incytr audit ledger

Schema per `docs/integrations/incytr_audit_plan.md` §4.4. One row per
introducing commit; rows are merged or split during the audit when changes
span buckets. **Every row starts as `bucket = TBD`** and gets a verdict during
Sprints 1–5.

Columns: `id` | `repo` | `bucket` | `introducing_commits` | `description` | `location` | `native_counterpart` | `equivalence_test` | `justification` | `disposition` | `parking_tag` | `verdict_date` | `signoff`

| id | repo | bucket | introducing_commits | description | location | native_counterpart | equivalence_test | justification | disposition | parking_tag | verdict_date | signoff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `INC-1` | incytr | TBD | `44d3008` | 2026-04-15 — simplify: address code review findings | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-2` | incytr | TBD | `e09105a` | 2026-04-15 — phase 3: generalize TPDS/PDS pipeline... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-3` | incytr | TBD | `9d888ea` | 2026-04-15 — simplify: address code review findings | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-4` | incytr | TBD | `388b6b7` | 2026-04-15 — update NAMESPACE with factorial mode ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-5` | incytr | TBD | `ff123e0` | 2026-04-15 — phase 2.5: wire factorial pipeline an... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-6` | incytr | TBD | `ad4042e` | 2026-04-15 — phase 2.4: add Contrast_SigProb with ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-7` | incytr | TBD | `84f258f` | 2026-04-15 — phase 2.2-2.3: add per-animal express... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-8` | incytr | TBD | `4a05aae` | 2026-04-15 — phase 2.1: add factorial S4 slots and... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-9` | incytr | TBD | `c88f479` | 2026-04-15 — simplify: address code review findings | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-10` | incytr | TBD | `c8e764e` | 2026-04-15 — phase 1.6: add N-condition integratio... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-11` | incytr | TBD | `bb0c991` | 2026-04-15 — phase 1.5: generalize permutation out... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-12` | incytr | TBD | `d7f3254` | 2026-04-15 — phase 1.4: generalize evaluation and ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-13` | incytr | TBD | `ca6a96e` | 2026-04-15 — phase 1.3: generalize kinase scoring ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-14` | incytr | TBD | `75bdd0d` | 2026-04-15 — phase 1.2: generalize Cal_SigProb and... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-15` | incytr | TBD | `719c2b1` | 2026-04-15 — phase 1.1: generalize barcodes_bycond... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-16` | incytr | TBD | `a2e99fd` | 2026-04-15 — simplify: address code review findings | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-17` | incytr | TBD | `6da9bcc` | 2026-04-15 — phase 0.6: update S4 class slot docum... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-18` | incytr | TBD | `3954fef` | 2026-04-15 — phase 0.5: standardize condition nami... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-19` | incytr | TBD | `c4698ef` | 2026-04-15 — phase 0.2: generalize integrate_omics... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-20` | incytr | TBD | `4e2e672` | 2026-04-15 — phase 0.1: refactor Integr_multiomics... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-21` | incytr | TBD | `4a9423c` | 2026-04-15 — phase 0.4: relocate expression and EM... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-22` | incytr | TBD | `ad7462b` | 2026-04-15 — phase 0.3: add tests for untested exp... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-23` | incytr | TBD | `b50c2cc` | 2026-04-15 — phase 0.0: add golden output fixture ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-24` | incytr | TBD | `4aa30c8` | 2026-04-15 — clean up codebase: remove dead code, ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-25` | incytr | TBD | `abde752` | 2026-04-10 — add EM promiscuity weighting and edge... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-26` | incytr | TBD | `b654fdd` | 2026-04-10 — add condition-label permutaiton test ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-27` | incytr | TBD | `f922c3a` | 2026-04-10 — add condition-label permutation test ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-28` | incytr | TBD | `6858063` | 2026-04-10 — memory permutation pass | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-29` | incytr | TBD | `d6d5c8c` | 2026-03-27 — kinase library adaptor | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-30` | incytr | TBD | `c3580fc` | 2026-03-27 — apply cutoff to all omics slots and f... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-31` | incytr | TBD | `847a014` | 2026-03-10 — docs: Add docs/notes to .gitignore to... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-32` | incytr | TBD | `5031094` | 2026-03-10 — add env.yml | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-33` | incytr | TBD | `650bca1` | 2026-03-09 — Remove CLAUDE.md from tracking and ad... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-34` | incytr | TBD | `b2a4c7f` | 2026-03-09 — Organize docs into reference and note... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-35` | incytr | TBD | `e04bf18` | 2026-03-06 — Improve evaluation/kinase code organi... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-36` | incytr | TBD | `76becbd` | 2026-03-06 — Remove dplyr dependency, replace all ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-37` | incytr | TBD | `c1b6fb9` | 2026-03-05 — Optimize performance, fix correctness... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-38` | incytr | TBD | `7e6735d` | 2026-03-05 — Refactor codebase: split monolith, de... | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-39` | incytr | TBD | `90c60e8` | 2026-03-05 — Reorganize project directory structure | TBD |  | TBD | TBD |  | TBD |  |  |
| `INC-40` | incytr | TBD | `6eeeac5` | 2026-03-05 — Update documentation for development ... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-1` | alzheimers | TBD | `8ca0a64` | 2026-05-04 — checkpoint: subclass-attribution branch bef... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-2` | alzheimers | TBD | `0adcca8` | 2026-04-28 — feat: add tyrosine phospho (pY) track paral... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-3` | alzheimers | TBD | `cc10112` | 2026-04-21 — pipeline: complete Unit 6.4 retire legacy a... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-4` | alzheimers | TBD | `b9f302b` | 2026-04-21 — pipeline: complete Unit 6.2 retire backbone... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-5` | alzheimers | TBD | `5bab404` | 2026-04-21 — pipeline: complete Unit 6.1 retire old viewers | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-6` | alzheimers | TBD | `2ab47fb` | 2026-04-20 — pipeline: lossless edge sharding for unifie... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-7` | alzheimers | TBD | `e096f36` | 2026-04-20 — update edges | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-8` | alzheimers | TBD | `99ab2b3` | 2026-04-20 — pipeline: complete Unit 1.3 build_edge_inde... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-9` | alzheimers | TBD | `97616a6` | 2026-04-16 — Factorial permutation tests with within-rec... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-10` | alzheimers | TBD | `2dda892` | 2026-04-15 — brain atlas data compression | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-11` | alzheimers | TBD | `f6e2b98` | 2026-04-14 — Simplify permutation code after review | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-12` | alzheimers | TBD | `9cf31de` | 2026-04-14 — Backbone-level permutation tests and per-pa... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-13` | alzheimers | TBD | `019334b` | 2026-04-14 — Cross-pair aggregation and Parquet-only cle... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-14` | alzheimers | TBD | `8cf4366` | 2026-04-14 — Update integration README for Phase 2 recei... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-15` | alzheimers | TBD | `8679681` | 2026-04-14 — Vectorized receiver scoring: Phase 2 of all... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-16` | alzheimers | TBD | `291fa00` | 2026-04-14 — Receiver-centric backbone enumeration: Phas... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-17` | alzheimers | TBD | `98d2b9d` | 2026-04-14 — remove dead code | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-18` | alzheimers | TBD | `3e688bf` | 2026-04-14 — All-pairs Incytr pipeline: DuckDB enumerati... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-19` | alzheimers | TBD | `9f339ed` | 2026-04-13 — Kinase-imputed pathway expansion: admit rec... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-20` | alzheimers | TBD | `19de928` | 2026-04-13 — Substrate-based external reranking with med... | TBD |  | TBD | TBD |  | TBD |  |  |
| `ALZ-21` | alzheimers | TBD | `5acec95` | 2026-04-09 — Phase 1 Incytr integration: adapters, R wra... | TBD |  | TBD | TBD |  | TBD |  |  |
