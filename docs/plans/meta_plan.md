# Meta Plan — Parallel Orchestration Experiment

**Goal:** implement many (not all) of the roadmap themes via parallel agentic operations, executed as a **gated sequence of per-wave workflows** — not one monolithic run. Each wave fans out file-disjoint code edits; heavy compute and viewer verification happen at human gates *between* waves.

**Confirmed intent:** prepare and then run parallel agentic implementation of the workflow-ready theme subset. B1 / E / H are blocked on external dependencies and are excluded from the orchestration pending a spike.

**Source of themes:** [`TODO.md`](../../TODO.md) (Themes A–H + Parallelization Strategy). Per-theme plans live in `docs/plans/theme_<x>/`.

---

## Why not one monolithic workflow

A workflow is a single-phase fan-out. Two gates are structurally un-automatable and force a waved, gated model:
- **Viewer verification is human** — A, C, B2, F end in a browser click-through ("visual changes are authoritative").
- **Heavy compute runs outside the fan-out** — state_mea, NSCLC regen, B3 incytr (song/5xfad/tcell) are memory-capped, possibly multi-hour → tmux scripts run at gates, not inline in parallel agents.

The waved model keeps parallelism *and* the gated checkpoints.

---

## Prep phases (must complete before orchestration)

- [ ] **P1 — Safety floor (hooks cut).** No memory/git hooks. A command-string hook can't see inside `python script.py` / `pixi run <task>` where the real OOM risk lives, so a denylist is theater; and the git policy is already CLAUDE.md + human gates. The real floor is structural and already in the plan: **heavy compute stays out of the fan-out** (out-of-band tmux scripts under the cap, at gates) and **worktree isolation** means a bad agent can't touch the working tree or peers (revert = `git worktree remove` / branch delete). P1 reduces to: tag the baseline + confirm worktree isolation round-trips before Wave 1.
- [x] **P2 — Cross-theme contract grill.** Done — `docs/plans/_contracts.md` locks C2, C1, B3, B5, F1, F2. *Keystone — prevents consistent-looking but contradictory parallel output.* Notable finding: **B3 is already built** (PTM schema + `wide_ptm/` convention exist and run for 5xFAD; song/tcell have no acet/ubiq data; Mukesh is bulk-MEA only with no incytr pathway, so PTM-on-Mukesh is out of scope) — B3 drops out of the contract-producer set below.
- [ ] **P3 — Per-theme audit → grill → plan**, in contract-dependency order. Depth = what Theme A got (its grill caught two stale-artifact bombs). Long pole of the prep.
- [x] **P4 — Dependency DAG + git topology.** Done — [`p4_dag.md`](p4_dag.md). Merge-to-main-per-gate off baseline tag `orchestration-baseline-2026-06-25`; worktree isolation per parallel agent (must symlink gitignored `data/`+`outputs/` in — ignore-all gitignore means a fresh worktree has neither); heavy compute at gates in the primary tree, never in a worktree. Tiers from harvested edges: **W1** C2/B5/D1/B4/A/G2 → **W2** C1/C5 → **W3** C3 → **W4** F1-then-F2 sweep → tail B4.2/B2/G1.
- [x] **P5 — Theme triage.** (Below.)

---

## Theme triage

### Blocked — spike before any plan (excluded from orchestration)
| Theme | Blocker |
|---|---|
| **B1** IncytrDB provenance | Waiting on Changhan email |
| **E1/E2** Kinase hierarchy | PhosphoSite Kinase Library access unconfirmed in-repo |
| **H1** TMT / IMAC replication | External data fetch availability unknown |

### Contract producers — decide centrally in P2, plan first in P3
| Theme | Shared contract it defines |
|---|---|
| **C2** Cohort naming | Display names across both viewers, axis labels, export filenames |
| **C1** Song genotype split | Payload schema consumed by C3, B2 |
| **F1** Signed-sort | Sort convention for *all* tables, *both* viewers |
| **F2** CSV export standard | Export format/columns for all tables |
| **B5** Backbone / specificity filter | Pathway reduction feeding B2 |

(B3 removed — already built; see P2 finding. B4 consumes the existing 5xFAD `wide_ptm/` schema directly.)

### Workflow-ready (after P2 contracts + P3 grilled plan)
| Theme | Notes / collision class |
|---|---|
| **A** T-cell | Plan written + grilled. Viewer builder (`build_tcell_viewer.py`) |
| **B3** acet/ubiq incytr | Backend, disjoint; heavy compute at gate |
| **B4** kinase→pathway | Depends on B3 |
| **B5** backbone filter | Algorithm/analysis; feeds B2 |
| **C3** disease-direction view | Depends on C1; unified viewer |
| **C5** 50-kinase mouse↔human substrates | Read-only analysis, disjoint |
| **D1** substrate comparator | Greenfield module |
| **G2** positive controls | Analysis/doc, disjoint |

### Cross-cutting / special handling
| Theme | Handling |
|---|---|
| **F1 / F2** | Decide convention in P2; **apply as a single sweep** after table-adding themes land, not concurrently |
| **B2** sankey | Viewer builder; gated after B5 |
| **G1** docs + workflow diagrams | **Last** — depends on everything; confluence skill |
| **C4** cross-species specificity guard | **Review constraint, not a build task** — enforce at every theme's review gate |

---

## Collision map (drives wave assignment)

Contention is concentrated in the **two viewer builders** + **F (cross-cuts both)**. Backend/greenfield/analysis themes are file-disjoint and parallelize freely.

- **Disjoint, parallel-safe:** B3, C5, D1, G2 (+ B5 algorithm)
- **Viewer-builder (serialize within builder):** A (`build_tcell_viewer`), C1/C3 (`build_unified_viewer`), B2, viewer-side of E
- **Cross-cutting (sweep, never concurrent with table-adders):** F1, F2

---

## Execution model (post-prep)

1. Tag baseline (`git tag orchestration-baseline-<date>`).
2. **Wave 1** — contract producers' code (C2 rename, C1 split, B3 schema, B5 filter) as worktree-isolated agents. Gate: human review + merge + any compute.
3. **Wave 2** — disjoint consumers (B4, C5, D1, G2, C3) fan out. Gate: compute runs + review.
4. **Wave 3** — viewer-builder edits (A, B2) — serialized per builder. Gate: viewer rebuild + **browser click-through (human)**.
5. **Wave 4** — F1/F2 cross-cutting sweep + G1 docs. Gate: final review.

Each wave = one workflow; you gate between waves. Rollback granularity: per-theme branch revert, or reset to baseline tag.

---

## Tooling — which Claude Code feature does which job

### In use
| Job | Tool / feature |
|---|---|
| **P2 contract grill + every P3 grill** | `grilling` skill — interactive, sequential, with you. Not parallelizable; it's the human-in-the-loop decision gate |
| **P3 audits** | Subagents (Agent tool): `audit-pipeline` for cross-file consistency, `map-pipeline` at the start of data-contract themes, general sonnet agents for the rest (as Theme A) |
| **P3 plans, contracts, DAG** | `Write` (me, after each grill). `_contracts.md`, theme plans, `meta_plan.md` are the trackers |
| **Execution engine (the waves)** | **`Workflow`** — one workflow per wave; workers run with `isolation: "worktree"` so parallel file edits can't collide |
| **Per-unit close + between-wave gates** | `code-review` (correctness on the diff) + `simplify` (cleanup), then the repo verification harness (`summary.py`, `verify_decomposition.py`) |
| **Heavy compute at gates** | Self-contained tmux scripts you run (not agent-babysat), under the P1 memory cap |
| **Cross-cutting decisions as they're made** | file-based **Memory** (`MEMORY.md` + entries) so contracts survive across sessions |
| **G1 docs theme (last)** | `confluence` skill (renders Mermaid → PNG, uploads) |

### Deliberately not used
| Tool | Why not |
|---|---|
| **Agent teams** | Reproducibility (no committable script), worktree isolation is safer than a shared tree, and N full sessions multiply OOM risk. Reserved only for live multi-cohort *exploration* — not this implementation experiment |
| **In-session task list** (`TaskCreate`/`TaskList`) | The markdown plans (`meta_plan.md`, theme plans) are the reviewable, committable tracker |
| **`loop` / `schedule` (cron agents)** | No recurring/interval work here |

The split: **prep (P1–P4) is interactive — you + me, sequential.** **Execution is workflows** — worktree-isolated fan-out, gated by you between waves.

---

## Status

| Phase | State |
|---|---|
| P1 safety floor | hooks cut — reduces to baseline tag + worktree-isolation check at Wave 1 |
| P2 contract grill → `_contracts.md` | **done** (C2, C1, B3, B5, F1, F2) |
| P3 per-theme plans | **done** — A, C2, C1, B5, F1, F2, C3, D1, C5, B4, G2 |
| P4 DAG + git topology | **done** (`p4_dag.md`) |
| P5 triage | **done** (this doc) |

**Next action:** **Prep (P1–P5) complete. Ready to execute** per `p4_dag.md`: tag `orchestration-baseline-2026-06-25` → dry-run the worktree+symlink round-trip → Wave 1 as one `Workflow` invocation. Execution is gated on the user starting it. **G2 done** (`theme_g/g2_plan.md`): deliverable is a **skill** `.claude/skills/check-controls/SKILL.md`, NOT a static list or CI gate — invoked per-cohort, carries a curated handful of control genes (kinase + non-kinase: PHKG1, ATP9A, APOE seed) with externally-expected home + AD direction, looks up actuals in existing artifacts, renders a **non-deterministic agent-judged** verdict (conversational, no committed output, no pass/fail gate). Open flag at approval: "non-deterministic" read as judged-verdict not random-subset. **B4 done** (`theme_b/b4_plan.md`): backend kinase→Incytr-node annotation join, **no regen/no heavy compute**, **Song + 5xFAD** (first audit was wrong — 5xFAD HAS MEA + pair-mode `wide_ptm/` + live expression layer; T-cell excluded — different MEA format). Position-aware cell match (Ligand→Sender, R/EM/Target→Receiver), **annotate-don't-drop** (artifact keeps all hits + `celltype_match`, default view filters to matches); expression-gate (5xFAD `expression_specificity`, graded) and disease-context (`snrna_attribution` LFC, age-matched) as **two separate annotations** (Song gets them after B5/`snrna`); **backbone grain NOT locked** — B4 emits a Receptor–EM fan characterization + parameterized `n_backbones`, firm definition **deferred to B5** (propagation note added to `b5_plan.md` — B5's 6-tuple identity re-commits the widest-enumeration conflation; reconcile). Viewer stubs (`#Backbones` col, `Driving kinases` panel) + stale preamble counts → **B4.2** once B5 fixes the key. Collisions: `song.py` (after C1/C3). **C5 done** (`theme_c/c5_plan.md`): one-off caller of D1's engine; pool = frozen 60-kinase (not 50) ST artifact `overlap_AD8_sus_clean.csv`; human profile = `b-donorset` (D1 per-donor builder over a swappable AD-donor set, default AD8∪{CTRL-08,CTRL-10} = suspect-as-AD, M=1, support surfaced); mouse = full sweep 2 tissues × 4 ages (cortex-12mo headline), ST only; "flag breakdown" = ranking not filter. **Standing policy:** official analyses = AD-only, one-offs treat suspects as AD, keep AD-group membership swappable (memory `cohort-grouping-suspect-as-ad`). D1 done (`theme_d/d1_plan.md`). C3 done (`theme_c/c3_*`). F1/F2 done (`theme_f/`). All contract producers planned.

**Gate-compute dependencies emerging (for P4 DAG):** C1 → bulk_mea recompute; B5 → `snrna` step (also restores the kinase tab's currently-NaN `song_specificity`). Both run at their wave's gate under the memory cap, not inline in the fan-out. C3 has **no** heavy compute (bounded reads only).

**Shared-file serializations (for P4 DAG):** C3 edits `song.py` (with C1, Wave 1 first) and seeds `numCmp` in `06_export_csv.js` (with F1, Wave 4 — whoever lands first defines it, the other adopts). F2 also reads C2's `COHORT_DISPLAY` and C1's extended crosstable export key array.
