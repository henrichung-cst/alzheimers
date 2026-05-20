# Epic: Adopt `feat/cr03-human-celltype-specificity` as the new main

**Status:** in progress
**Owner:** hchung
**Created:** 2026-05-20

## For agents

You are implementing one phase of this epic. Workflow:

1. Read this entire file (especially Framing, Goals, and the Implementation Log at the bottom — later phases depend on earlier-phase notes).
2. Read your assigned phase doc under `docs/plans/phases/cr03_adopt/`.
3. Run the **Preflight** in your phase doc before touching anything.
4. Execute the phase exactly as specified. If you discover the plan is wrong, **stop and write a note in the Implementation Log explaining the divergence** — do not silently improvise.
5. Before declaring done, append an entry to the **Implementation Log** at the bottom of this file with: phase, date, commit SHA(s) produced, anything the next phase needs to know.
6. Mark the phase doc's status as `done` at the top.

Hard rules (memory-derived, do not violate):
- `[[feedback_no_claude_coauthor]]` — no `Co-Authored-By: Claude` in commit messages.
- `[[feedback_no_intentional_wrong_behavior]]` — never reintroduce the row-cap / parallel-fetch Incytr pattern; never ship known-broken output to dodge a hard fix.
- `[[feedback_no_backcompat_on_research_pivots]]` — no flags/fallbacks layered to keep both worlds working.
- `[[project_direct_levy_t5_mapping]]` — branch's levy_t5-as-sole-spine direction is canonical.
- `[[project_incytr_pair_pvalue_untrustworthy]]` — pair-mode is the active Incytr path; factorial archive stays archived.

Do not push or open PRs unless the phase doc says so. Do not run `git reset --hard`, force-push, or any destructive op outside the explicit Phase 1 step (which has its own safety tags).

## Framing

`feat/cr03-human-celltype-specificity` is 11 commits ahead of `main`, branched at `03dcf60`. After widening the audit beyond the viewer, the branch turns out to be a **substantive architectural pivot**, not a feature add:

| Domain | Branch change |
| --- | --- |
| Spine | Drops levy19 back-compat; **levy_t5 is the sole spine everywhere** (`ea440f9`) — matches `[[project_direct_levy_t5_mapping]]` |
| Incytr integration | **Factorial path retired**; pair-mode is the active integration (`3428a1b`, ≈2900 LOC deleted) |
| Human attribution | New pipeline: `human_celltype_attribution.py`, `human_reference_expression.py`, atlas SEA-AD/HBCA flags, config additions, dedicated runners |
| Viewer (data) | PAYLOAD.human.celltype_specificity block + sub-tab + Crosstable tab + Family columns |
| Viewer (perf) | Three commits that pre-date and conflict with main's per-shard pagination (`c498c42`, `f39e661`, `adc7ffa`) — **do not preserve** |
| Path layout | `data/incytr/` → `data/incytr_frozen/` rename + factorial archive (`3428a1b`) |
| Docs | READMEs / CLAUDE.md / integration README rewritten for levy_t5-only world |
| Runners | New: `run_levy_t5_attribution_rebuild.sh`, `run_pair_mode_pipeline.sh`, `run_hbca_download.sh` |
| Env | pixi.toml + lockfile churn |

Diff stat: **4,850 insertions / 5,454 deletions across 71 files** — net contraction (closed path retired).

**Conclusion: `main` has been the side branch since `03dcf60`.** The right operation is "adopt the branch as main and forward-port the few useful main-side commits onto it" — not "merge the branch into main."

What's on main that isn't on the branch (must be preserved):

| Commit | Subject | Disposition |
| --- | --- | --- |
| `56fd2bf` | feat(viewer): CR-04 trajectory_index + recur_index computation | Forward-port (Phase 2) |
| `5609eab` | feat(viewer): CR-04 trajectory chips, Recur-in filter, temporal detail | Forward-port (Phase 3) |
| per-shard pagination (uncommitted) | `incytr_pathways.js` rewrite | Commit in Phase 0, forward-port in Phase 4 |
| `a152c52` | feat(viewer): Incytr pathways Measurement Trace panel | Forward-port (Phase 5) |
| audit + epic docs (uncommitted) | `docs/plans/viewer_audit_2026-05-20.md` + this file + phase docs | Commit in Phase 0, forward-port in Phase 6 |

## Goals

1. Main points at a commit containing: entire CR-03 branch + the 3 forward-ported main commits + per-shard pagination + audit/epic docs.
2. History stays linear-ish: branch tip plus 4 cherry-picks on top, not a merge tangle.
3. The 3 superseded perf commits are dropped — do **not** carry forward.
4. `feat/cr03-human-celltype-specificity` deletable at the end; safety tags retained.

Non-goals: any new feature work; rebuilding pipeline outputs; restoring deleted factorial files.

## Resolved decisions

| Question | Answer |
| --- | --- |
| Archive or regenerate `outputs/reports/decomposition/levy19/...`? | Leave to regenerate as `levy_t5/` on next pipeline run. |
| Preserve any deleted factorial Incytr files? | No. Recovery via `pre-cr03-adopt-main` tag if ever needed. |
| Does `human_celltype_attribution.py` need a one-time run before Phase 7? | **No.** Output `outputs/reports/kinase_attribution_human/celltype_specificity.csv` (540 KB) already exists on disk; builder uses try/except so even a stale CSV won't break the build. |

## Phases

Each phase is a separate doc under `docs/plans/phases/cr03_adopt/`. Run in order; do not skip.

| Phase | Doc | Summary |
| --- | --- | --- |
| 0 | [phase00_safety_anchors.md](phases/cr03_adopt/phase00_safety_anchors.md) | Commit uncommitted main work, discard the rename-only diffs, tag both endpoints, push tags. |
| 1 | [phase01_reset_main_to_branch.md](phases/cr03_adopt/phase01_reset_main_to_branch.md) | `git reset --hard` main onto branch tip. Verify env, history. |
| 2 | [phase02_forward_port_trajectory_index.md](phases/cr03_adopt/phase02_forward_port_trajectory_index.md) | Cherry-pick `56fd2bf` (builder-side trajectory_index). |
| 3 | [phase03_forward_port_trajectory_chips.md](phases/cr03_adopt/phase03_forward_port_trajectory_chips.md) | Cherry-pick `5609eab` (viewer-side trajectory chips + Recur-in filter). |
| 4 | [phase04_forward_port_pagination.md](phases/cr03_adopt/phase04_forward_port_pagination.md) | Cherry-pick the per-shard pagination commit; strip branch's row-cap pattern. |
| 5 | [phase05_forward_port_measurement_trace.md](phases/cr03_adopt/phase05_forward_port_measurement_trace.md) | Cherry-pick `a152c52` (Measurement Trace panel). |
| 6 | [phase06_forward_port_docs.md](phases/cr03_adopt/phase06_forward_port_docs.md) | Cherry-pick audit + epic doc commit. Update docs to reflect completion. |
| 7 | [phase07_verify_and_wire_crosstable.md](phases/cr03_adopt/phase07_verify_and_wire_crosstable.md) | Build viewer, hard-refresh, smoke every tab. Wire Crosstable into `TAB_MANIFEST` (branch added JS but not registration). |
| 8 | [phase08_cleanup.md](phases/cr03_adopt/phase08_cleanup.md) | Delete CR-03 branch local + remote; push main; retain safety tags. |

## Implementation Log

Append one entry per completed phase. Format:

```
### Phase N — <title>
**Agent:** <model + date>
**Commits produced:** <sha1>, <sha2>, ...
**Notes for next phase:** <anything surprising, deferred decisions, files left dirty, conflict resolutions worth remembering>
```

<!-- entries below this line -->

### Phase 0 — Safety anchors
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:**
- `7384ca4` fix(viewer): per-shard pagination + env-controlled memory budget — **Phase 4 cherry-picks this**
- HEAD of main at end of Phase 0 = docs(viewer): audit + CR-03 reconcile epic and phase plans (amended to carry this log entry) — **Phase 6 cherry-picks this** (the original pre-amend SHA was `eedcb7d`; resolve via `git log --grep="audit + CR-03"` after Phase 1 reset since the SHA changes with each Implementation Log amend)

**Tags pushed to `public` remote** (no `origin` configured; only `public` → github.com/henrichung-cst/alzheimers):
- `pre-cr03-adopt-main` → `eedcb7d` (main tip after Phase 0)
- `pre-cr03-adopt-branch` → `0cb721f` (feat/cr03-human-celltype-specificity tip)

**Notes for next phase:**
- All 11 rename-only files were confirmed pure `data/incytr/` → `data/incytr_frozen/` renames (every added line contained `incytr_frozen`; equal add/remove counts) and were discarded via `git checkout --`. The branch already carries this rename in `3428a1b`.
- Remote is named `public`, not `origin`. Phase 8 push instructions should use `public`.
- Left untracked (intentionally, per phase doc §0.1): `docs/plans/incytr_mea_seed_list_expansion_plan.md`, `docs/plans/incytr_pathway_measurement_trace_plan.md`. They are unrelated drafts; Phase 1's hard reset will leave them in the working tree (untracked files survive `reset --hard`).
- Main is now 2 commits ahead of `public/main`; do not push main before Phase 1 hard-resets it.

## References

- Branch tip: `feat/cr03-human-celltype-specificity` (will be tagged `pre-cr03-adopt-branch` in Phase 0)
- Main tip at start: will be tagged `pre-cr03-adopt-main` in Phase 0
- Merge base: `03dcf60`
- Viewer audit: `docs/plans/viewer_audit_2026-05-20.md`
- Memory ratifications: `[[project_direct_levy_t5_mapping]]`, `[[project_incytr_pair_pvalue_untrustworthy]]`, `[[feedback_no_intentional_wrong_behavior]]`, `[[feedback_no_backcompat_on_research_pivots]]`, `[[feedback_no_claude_coauthor]]`, `[[feedback_plans_in_repo]]`
