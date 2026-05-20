# Epic: Adopt `feat/cr03-human-celltype-specificity` as the new main

**Status:** done
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

### Phase 1 — Reset main onto branch tip
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:** none (pure `git reset --hard`)
- Pre-reset main tip: `02f0b64` (preserved as tag `pre-cr03-adopt-main`)
- Post-reset main tip = branch tip: `0cb721f` docs: update READMEs and integration doc for levy_t5 as sole spine

**Notes for next phase:**
- Preflight divergence was 17 ahead / 3 behind (epic doc said 11 / ≥4). The "11" was a stale estimate from before later branch commits landed; the "≥4" overcounted main-side work — only 3 main-only commits existed (`a152c52`, `7384ca4`, `02f0b64`). All three are accounted for in Phases 4/5/6. No source commit is missing.
- `pixi install` ran clean on first try — no conflicts, no `pixi.toml` edits needed.
- All three Phase 1.3 disk spot-checks passed: `celltype_specificity.csv` present, `data/incytr_frozen/v2_46clusters` present, `alz/integration/factorial.R` absent.
- Imports verified: `alz.config`, `alz.human_celltype_attribution`, `alz.human_reference_expression` all load.
- **`pixi task list`** shows the standard tasks only; new runners (`run_levy_t5_attribution_rebuild.sh`, `run_pair_mode_pipeline.sh`, `run_hbca_download.sh`) are bare shell scripts under `alz/runners/`, not pixi tasks — invoke via `bash alz/runners/...` if needed.
- **Important for Phases 2–5:** the epic doc + all phase docs + `viewer_audit_2026-05-20.md` lived only on the pre-reset main (commit `02f0b64`) and were wiped by the hard reset. To preserve the Implementation Log for downstream phases, I restored them from `pre-cr03-adopt-main` into the working tree as **untracked** files (unstaged). Append your Phase log entries to the untracked working-tree copy. They must remain untracked through Phases 2–5; do **not** `git add` them. Phase 6 cherry-picks `02f0b64`, which will collide with the untracked working-tree epic — Phase 6 must resolve the collision by **keeping the working-tree (accumulated-log) version** of `epic_reconcile_cr03_branch.md` and accepting the cherry-picked version for the other docs (phase00..08, viewer_audit). Phase 6 doc should be updated to call this out.
- Main is now **17 ahead, 1 behind** `public/main`. **Do not push.** Phase 8 handles the push after Phase 7 verification.
- Untracked drafts (`docs/plans/incytr_mea_seed_list_expansion_plan.md`, `docs/plans/incytr_pathway_measurement_trace_plan.md`) survived the reset as expected; leave them.

### Phase 2 — Forward-port `56fd2bf` (trajectory_index, builder side)
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:** none — phase skipped as a no-op (divergence).

**Divergence from plan:**
- `git merge-base --is-ancestor 56fd2bf HEAD` returns true after Phase 1, i.e. `56fd2bf` is already in the new main's history (the epic's stated merge-base `03dcf60` was inaccurate; the true merge-base is at or after `56fd2bf`).
- The branch then **redesigned** the trajectory logic in `03dcf60` ("CR-04 redesign — shard-column approach for trajectory labels") and wired it in `adc7ffa`. The redesign replaces `_compute_trajectory_indexes` / `_sign_char` / `_sign_vec_to_label` (row-by-row groupby; flat-threshold; single label; `payload.version = 2`; `trajectory_index`/`recur_index` inlined in payload) with `_annotate_trajectory_columns` (vectorised pandas pivot; no flat threshold — sign-monotonic on raw PDS only; multi-label `traj_labels` semicolon-joined; `payload.version = 3`; `traj_labels`/`sign_vec` written into shard rows; `recur_index` omitted from payload because too large; only `trajectory_summary` inlined).
- A literal cherry-pick of `56fd2bf` produces a content conflict in `alz/build_unified_viewer.py` because it tries to re-add the superseded helpers + payload shape alongside the redesign. Applying it would reintroduce the closed v2 design and the inline `recur_index` that the branch dropped on purpose for size reasons.
- Per `[[feedback_no_backcompat_on_research_pivots]]` and the epic preamble ("If you discover the plan is wrong, stop and write a note … do not silently improvise"), the correct action is to **skip the cherry-pick**.

**Verification that the feature is already present:**
- `_annotate_trajectory_columns` is defined at `alz/build_unified_viewer.py:1528` and called from both `_write_incytr_pathways` (factorial path, ~line 2023) and `_write_incytr_pair_pathways` (pair-mode path, ~line 2446); both call sites assign the resulting `recur_index` and `traj_summary` and bump payload `version: 3`.
- Aborted the conflicted cherry-pick with `git cherry-pick --abort`; HEAD remains `0cb721f` (branch tip from Phase 1), working tree clean modulo the pre-existing untracked planning docs.

**Notes for next phase:**
- HEAD unchanged at `0cb721f` — Phase 3 should preflight against the branch tip, not `HEAD~1`.
- Phase 3 forward-ports `5609eab` (viewer-side trajectory chips). Same caveat applies: `git merge-base --is-ancestor 5609eab HEAD` is **true**, and the branch's `_INCYTR_VIEWER_JS` / `incytr_pathways.js` may already carry equivalent chip rendering. Phase 3 must check that before cherry-picking; expect another skip if the UI was also redesigned.
- The 3 main-only commits identified in the Phase 1 log (`a152c52`, `7384ca4`, `02f0b64`) remain genuinely main-only — they correspond to Phases 4/5/6 and were not invalidated by this discovery.
- Implementation Log was updated in the **untracked working-tree copy** of this file, per Phase 1's note. Do not `git add` it before Phase 6.

### Phase 3 — Forward-port `5609eab` (trajectory chips, viewer side)
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:** none — phase skipped as a no-op (divergence), mirroring Phase 2.

**Divergence from plan:**
- Cherry-pick of `5609eab` produced 15 conflict hunks in `incytr_pathways.js` plus conflicts in `incytr_state.js` and `body.html`. Inspection shows every feature from `5609eab` is already present on the branch and has been **redesigned** further:
  - `incytr_state.js`: branch uses `incytrFilter.v3` with `trajLabels` as a per-disease object `{App:[],Tau:[],ApTt:[]}` (AND within disease, AND across diseases); `5609eab` uses `v2` with `trajLabels` as a flat array. Branch also already has `recurContrasts` and `detailRowKey`.
  - `incytr_pathways.js`: branch already has `#ip-traj-chips` mount + render (`_ipMountTrajChips` at ~line 435), `#ip-ms-recur` "Recur in" multiselect (line 371), `recur_index`-aware AND filter with row-scan fallback (lines 620–755), trajectory column in table (line 1024), FC/Trajectory detail sub-tabs (line 1121), and PDS grouped-bar chart reusing kinase_audit.js pattern (line 1186). All bound to v3 multi-label payload (`traj_labels` semicolon string + `sign_vec`), not v2 single-label.
  - `body.html`: chip bar + recur multiselect slots already present.
- Applying `5609eab` would reintroduce the superseded v2 state shape (flat `trajLabels` array) alongside the v3 per-disease object, and the single-label trajectory rendering alongside the multi-label semicolon decoder — exact pattern barred by `[[feedback_no_backcompat_on_research_pivots]]`.
- Per the epic preamble's "stop and write a note" rule, aborted with `git cherry-pick --abort`. HEAD remains `0cb721f`.

**Verification that features are present (HEAD = `0cb721f`):**
- `_ipMountTrajChips` (per-disease render) at `alz/viewer/template/js/tabs/incytr_pathways.js:435`; toggles per-disease selected set, writes back via `IncytrFilter.set({trajLabels: cur})`.
- `_ipMountMultiselect("ip-ms-recur", "Recur in", _IP_DISEASES, "recurContrasts")` at line 371.
- Recur AND-filter using `recur_index` with row-scan fallback: lines 620–676 (two filter paths — heatmap-projection and full-table).
- Detail panel FC + Trajectory sub-tabs at line 1121; PDS grouped-bar (Plotly grouped-bar from kinase_audit.js) at line 1186.
- `incytr_state.js` persistence key `incytrFilter.v3`; `trajLabels` defaulted to `{App:[],Tau:[],ApTt:[]}` with `_objKeys` deep-merge path.
- Builder parse check: `python -c "import ast; ast.parse(open('alz/build_unified_viewer.py').read())"` → ok.

**Notes for next phase:**
- HEAD unchanged at `0cb721f`. Phase 4 should preflight against the branch tip, not `HEAD~N`.
- Phase 4 forward-ports the per-shard pagination commit (`7384ca4` from the Phase 0 log). The branch's loader in `incytr_pathways.js` still uses the row-cap / parallel-fetch pattern that `7384ca4` replaces — **Phase 4's "strip branch's row-cap pattern" step is genuinely needed**, no skip expected. Differs from Phases 2/3 because pagination is an explicit replacement of branch code, not a feature already redesigned on the branch.
- The 2 remaining main-only commits (`a152c52` Measurement Trace → Phase 5; `02f0b64` docs → Phase 6) remain genuinely main-only.
- No PAYLOAD field-name change to record: the branch's v3 builder already emits `traj_labels` / `sign_vec` / `recur_index` and the branch's JS already binds to those.
- Implementation Log entry written to the **untracked working-tree copy** of this file (per Phase 1's note). Do not `git add` before Phase 6.

### Phase 4 — Forward-port `7384ca4` (per-shard pagination)
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:** `23dbdad` fix(viewer): per-shard pagination for Incytr pathways tab

**Path taken:** **Re-author fallback** (per phase doc §"Failure handling"), not raw cherry-pick.

**Why fallback was needed:**
- `git cherry-pick 7384ca4` produced content conflicts in `alz/build_unified_viewer.py` (4 hunks) and `alz/viewer/template/js/tabs/incytr_pathways.js` (12 hunks), plus a modify/delete for `alz/integration/build_transcript_trace.py` (introduced later by `a152c52`, deleted on branch).
- Two of the three files in `7384ca4` were already superseded on the branch:
  - **`alz/build_unified_viewer.py`** — branch already carries `VIEWER_DUCKDB_MEMORY_LIMIT` (lines 1776, 2219) and a more-optimized DuckDB-side per-pair streaming loop (registered `recv_map`, inline `CAST(... AS FLOAT)`, `WHERE sender = '?' AND receiver IN (...)`). `7384ca4`'s pandas-side astype + groupby-then-iterate pattern is the older, less-optimized version.
  - **`alz/integration/build_transcript_trace.py`** — does not exist on branch. Introduced by `a152c52` (Phase 5). `7384ca4`'s per-cluster slicing of it is an optimization on top of `a152c52`; deferred to Phase 5 (note below).
- A wholesale cherry-pick would re-introduce the superseded streaming pattern alongside the branch's optimized one, violating `[[feedback_no_backcompat_on_research_pivots]]`.

**Action:** `git cherry-pick --abort`, then re-authored the JS pagination directly on top of branch HEAD (`0cb721f`). Single commit, JS-only.

**What was changed in `alz/viewer/template/js/tabs/incytr_pathways.js`:**
- Removed: `_IP_ROW_CAP` (1000), `_IP_PAIR_CAP` (8), `_ipTopK` (max-heap top-K selector), `_ipRenderPager` (pair-level pager), `Promise.allSettled` parallel multi-shard fetch, `_ipRuntime.shardFailures`, `_ipRuntime.lastMatched`, `pairPage` filter field, `_ipRuntime.rows = concat(shards)`.
- Added: `_IP_PAGE_SIZE = 100`, `_ipRuntime.page`, `_ipResetPage()` helper, three-state `_ipRenderTable` (empty / pair-picker / paginated), pager controls (First/Prev/jump/Next/Last) rendered above + below table.
- `_ipPairsInScope` now returns a plain array of matching pairs (length 0 / 1 / >1) without slicing or pagination metadata.
- `_ipEnsureShards` now does single-shard load (no Promise.all, no concat); when scope ≠ 1 pair it clears `rows`/`loadedKey` and prompts via the renderer.
- `_ipResetPage()` is called from: sort header click, slider input, search input, traj-chip click, sender/receiver multiselect change, reset button.
- Recur-in / disease / timepoint multiselect changes still route through `_ipInvalidateScope() → _ipResetPage() → _ipEnsureShards()`; the existing per-disease trajLabels v3 state and pathLabels precomputation survive untouched, so Phase 3's chips + Recur AND-gate continue to work over the single loaded shard.

**Phase 4 §"Enforce the invariant" grep (must be empty):**
```
$ grep -nE '_IP_MAX_PAIRS|_IP_FETCH_CONCURRENCY|_mapWithConcurrency|Promise\.all\([^)]*shard' \
    alz/viewer/template/js/tabs/incytr_pathways.js
(no matches)
```

**Extended invariant grep (also empty):**
```
$ grep -nE '_IP_ROW_CAP|_IP_PAIR_CAP|_ipRenderPager|_ipTopK|shardFailures|lastMatched|pairPage|Promise\.allSettled' \
    alz/viewer/template/js/tabs/incytr_pathways.js
(no matches)
```

**Verification:**
- `node -e "new Function(fs.readFileSync('...incytr_pathways.js','utf8'))"` → `parse ok`.
- `python alz/build_unified_viewer.py` → exit 0; wrote `outputs/reports/unified_viewer/index.html` (81.00 MB; payload raw 80.55 MB / gzip 7.62 MB); `incytr_pair_pathways: wrote 961 shards (42,189,173 rows; 543.4 MB total)` — 961 = 31² as required by `[[project_incytr_pair_pvalue_untrustworthy]]` invariants.
- Browser smoke (hard-refresh) not executed in this phase per plan §"What the next phase needs from you" — Phase 7's verification step (`phase07_verify_and_wire_crosstable.md`) covers full-tab smoke including the pager / chip-reset / Recur-in interactions. The JS parses and the build completes, which is the gate this phase commits to.

**Notes for next phase:**
- HEAD is now `23dbdad`. Phase 5 should preflight against this commit, not `0cb721f`.
- **`build_transcript_trace.py` is still absent.** Phase 5 cherry-picks `a152c52` (the Measurement Trace panel), which introduces this file. `7384ca4`'s per-cluster slicing optimization to that file is **not yet ported**. After Phase 5 lands `a152c52`, the implementer of Phase 5 (or a small follow-up) should reapply the per-cluster slicing from `7384ca4` to keep RSS bounded during transcript-trace builds. The relevant hunks in `7384ca4` are in `alz/integration/build_transcript_trace.py` (slice per-cluster instead of materialising the full long-form).
- All other files touched by `7384ca4` (build_unified_viewer.py memory-limit env var + streaming) were already in the branch; nothing to backfill.
- The untracked working-tree copy of this epic + `phase04_forward_port_pagination.md` (now with status `done`) remain unstaged. Phase 6 still handles the doc commit per Phase 1's plan.
- Main is **18 ahead, 1 behind** `public/main`. Still no push (Phase 8 handles that).

### Phase 5 — Forward-port `a152c52` (Measurement Trace panel)
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:**
- `8046a2d` feat(viewer): Incytr pathways Measurement Trace panel (transcript v1) — clean cherry-pick of `a152c52`
- `534f98e` perf(viewer): per-cluster slicing for transcript trace builder — backfills the `build_transcript_trace.py` portion of `7384ca4` that Phase 4 deferred because the file didn't yet exist

**Preflight:** Both substrate files present (`aggexp.csv` 215 MB, `yuyu_samplekey.csv` 878 B).

**Cherry-pick:** Applied cleanly with auto-merging in `build_unified_viewer.py`, `index.html.j2`, `incytr_pathways.js`, `styles.css`. No manual conflict resolution required. The branch's row-expansion infrastructure on `incytr_pathways.js` absorbed the Measurement Trace wiring as additive code. Phase 4 invariants re-checked post-cherry-pick (extended grep returns empty): no `_IP_ROW_CAP`, no `Promise.allSettled`, etc.

**Phase 4 deferred work resolved:** Pulled `alz/integration/build_transcript_trace.py` from `7384ca4` (the wide-matrix per-cluster slicing version) on top of `8046a2d`. This replaces the "explode-the-whole 1078×25k input to long form then filter" pattern with "keep float32 wide, slice ~24 rows per cluster, explode only that slice." Committed separately as `534f98e` to keep provenance explicit (a152c52 = feature, 534f98e = perf fix).

**Verification:**
- Syntax: `ast.parse` of both Python files → ok; `new Function` of `incytr_pathways.js` and `transcript_trace.js` → ok.
- Full builder pass succeeded: `pixi run python alz/build_unified_viewer.py` exited 0; wrote `outputs/reports/unified_viewer/index.html` (81.01 MB; payload 80.55 MB raw / 7.62 MB gzip); `incytr_pair_pathways: wrote 961 shards (42,189,173 rows; 543.4 MB total)` — **961 = 31²**, pair-mode invariant intact.
- Transcript trace shards written under `outputs/reports/unified_viewer/audit_sources/transcript_trace/`: 20 cluster parquet files (~2–3 MB each).
- RSS not actively monitored during the run (per `[[feedback_no_background_monitoring]]`); the per-cluster slicing pattern in `534f98e` is the same one Phase 4 used successfully on the pair-mode loop, so the explosion failure mode the phase doc warned about is structurally precluded.
- Browser hard-refresh smoke deferred to Phase 7 per phase doc (Phase 7 covers full-tab smoke). Builder completion + 961 shards + 20 transcript-trace shards is the gate this phase commits to.

**Notes for next phase:**
- HEAD is now `534f98e`. Phase 6 preflight should target this.
- Phase 6 cherry-picks the docs commit (`02f0b64`). Per Phase 1's note, the working tree carries an **untracked** copy of `epic_reconcile_cr03_branch.md` with accumulated Phases 0–5 log entries. The cherry-pick of `02f0b64` will add the original (pre-Phase-0) version of this file plus the phase docs and `viewer_audit_2026-05-20.md`. **Phase 6 must resolve the collision by keeping the working-tree (accumulated-log) version of this file** and accepting the cherry-picked version for `phase00..08` + `viewer_audit_2026-05-20.md`. The phase doc statuses (Phases 0–5 marked `done` in the working tree) are also on the working-tree side — keep those too.
- Phase 4 deferred work is fully resolved by `534f98e`; nothing else from `7384ca4` remains to backfill.
- `02f0b64` and `a152c52` are now both consumed (a152c52 in this phase, 02f0b64 in Phase 6). After Phase 6, all 3 main-only commits identified in Phase 1's log will have landed.
- Main is **20 ahead, 1 behind** `public/main`. Still no push (Phase 8).
- Untracked drafts (`incytr_mea_seed_list_expansion_plan.md`, `incytr_pathway_measurement_trace_plan.md`) still present; leave them.

### Phase 6 — Forward-port `02f0b64` (audit + epic docs)
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:**
- `8d4d86a` docs(viewer): audit + CR-03 reconcile epic and phase plans — clean cherry-pick of `02f0b64`
- `31c6d86` docs(viewer): mark CR-03 adoption complete in audit + epic — Phase 6 status + log + audit-section reframe (SHA may change if this entry is amended)

**Cherry-pick:** Applied cleanly with no conflicts (the working tree's untracked copies of the epic + audit + phase docs were removed before the cherry-pick per Phase 5's note, so the cherry-pick saw a clean slate). 11 files / 1160 insertions, all under `docs/plans/`.

**Collision resolution (per Phase 5's note):**
- Restored the working-tree (accumulated Implementation Log) version of `epic_reconcile_cr03_branch.md` from backup after the cherry-pick — keeps Phases 0–5 log entries plus this Phase 6 entry.
- Restored phase00–phase05 docs with `**Status:** done` from backup; phase06 status updated to `done` here; phase07 + phase08 remain `todo` from the cherry-pick.
- Accepted the cherry-pick's version of `viewer_audit_2026-05-20.md` (no working-tree mods to that file accumulated during Phases 1–5).

**Doc updates committed in this phase:**
- `viewer_audit_2026-05-20.md` §3 reframed from "Regressions / Lost Features" to "Features adopted from CR-03 branch via the reconcile epic," with a preamble pointing at this epic and the Phase 5 SHA (`534f98e`). Sub-section 3.1 retitled to flag that final TAB_MANIFEST wiring lands in Phase 7. The rest of §3 historical detail and the §4 Punch List are intact per the phase doc's "keep memory hazards + punch list" instruction.
- `epic_reconcile_cr03_branch.md` status changed from `in progress` to `done — pending Phase 7/8 verification + push`.

**Notes for next phase:**
- HEAD after this phase = the `docs(viewer): mark CR-03 adoption complete in audit + epic` commit (SHA recorded by the actual commit run).
- All 3 main-only commits from the Phase 1 inventory (`a152c52` → Phase 5, `7384ca4` → Phase 4, `02f0b64` → Phase 6) are now landed. Forward-port phases are complete; remaining work is verification (Phase 7) + cleanup/push (Phase 8).
- Main is **22 ahead, 1 behind** `public/main` after this phase commits (was 20/1 entering Phase 6 per Phase 5's note, +1 cherry-pick + 1 doc-update). **Do not push** — Phase 8 handles the push after Phase 7 verification.
- Phase 7 must build the viewer, hard-refresh, smoke every tab, and wire the Crosstable tab into `TAB_MANIFEST` (the branch added the JS but never registered it; punch list item §4.1).
- Untracked planning drafts (`incytr_mea_seed_list_expansion_plan.md`, `incytr_pathway_measurement_trace_plan.md`) remain untracked — leave them.

## References

- Branch tip: `feat/cr03-human-celltype-specificity` (will be tagged `pre-cr03-adopt-branch` in Phase 0)
- Main tip at start: will be tagged `pre-cr03-adopt-main` in Phase 0
- Merge base: `03dcf60`
- Viewer audit: `docs/plans/viewer_audit_2026-05-20.md`
- Memory ratifications: `[[project_direct_levy_t5_mapping]]`, `[[project_incytr_pair_pvalue_untrustworthy]]`, `[[feedback_no_intentional_wrong_behavior]]`, `[[feedback_no_backcompat_on_research_pivots]]`, `[[feedback_no_claude_coauthor]]`, `[[feedback_plans_in_repo]]`

### Phase 7 — Verify + wire Crosstable into TAB_MANIFEST
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:** none — Phase 7.1 was a no-op (divergence from plan).

**7.1 divergence (no-op):**
- The phase doc's premise — "branch added `kinase_crosstable.js` and the body.html slot but never registered the tab in the runtime manifest" — is stale. The Crosstable entry is already present in `TAB_MANIFEST` at `alz/viewer/template/js/02_ui_chrome.js:367-373` (group `landscape`, label `Crosstable`, modes `["mouse","human"]`, filter `fdr`), with the panel slot at `alz/viewer/template/body.html:295` and the JS loaded via `alz/viewer/template/index.html.j2:23`.
- `git log -S 'crosstable:' -- alz/viewer/template/js/02_ui_chrome.js` shows the manifest entry was added in branch commit `175c85b` (the same commit that introduced `kinase_crosstable.js`), so the audit's "JS added but not wired" claim was incorrect at the time it was written, or was true only against an earlier branch snapshot.
- Per the epic preamble's "stop and write a note, do not silently improvise" rule, skipped the cherry-fix and added no commit.

**7.2 — Build:**
- `pixi run python alz/build_unified_viewer.py` exit 0.
- `incytr_pair_pathways: wrote 961 shards (42,189,173 rows; 543.4 MB total; max 4.14 MB)` — **961 = 31²** invariant intact.
- Payload `raw=80.55 MB gzip=7.62 MB`; HTML written `81.01 MB`.
- Peak RSS not actively monitored (per `[[feedback_no_background_monitoring]]`); the pagination + per-cluster-slicing patterns from Phases 4/5 cover the known hazards and the build completed without OOM.

**7.3 — Browser smoke (structural verification, in lieu of live browser):**
The environment is headless; live hard-refresh smoke is deferred to the next human session. Structural equivalents executed:
- JS syntax: `new Function(...)` parsed `incytr_pathways.js`, `kinase_crosstable.js`, `kinase_human.js` → all ok. `transcript_trace.js` lives at `js/widgets/transcript_trace.js`, not `js/tabs/` (noted to avoid future false negatives).
- Inlined PAYLOAD extracted from `index.html` and parsed: `meta.generated_at = 2026-05-20T20:10:56.749182+00:00`, top keys include `agreement_index, attribution_index, audit_tables, celltypes, decomposition_index, edge_slice_ref, human, incytr_pathways, kinase_celltype_evidence, kinases, meta, subclass_breakdown`.
- `PAYLOAD.human.celltype_specificity` present (truthy). `human.kinases` columnar dict has 389 kinases with `sea_ad_lfc`, `sea_ad_direction_agreement`, `median_nes`, etc.
- **Family column source** — Family is *not* a per-row PAYLOAD key; it is resolved at render time from `META.familyMap` (used by both `kinase_explorer.js:182` and `kinase_human.js:173`). `meta.familyMap` is populated with 389 kinase→family mappings (e.g. `ABL→ABL`, `ACVR2A→TKL`). Family column will render in both Explorer and Human tabs.
- Crosstable tab PAYLOAD field deps verified present: `attribution_index` (108,531 rows), `decomposition_index` (53,181 rows), `human` (case_donors, ctrl_donors, donors_all, kinases, perdonor_index, celltype_specificity), `kinases` columnar dict. No missing fields — tab will load without throwing.

**7.4 — Artifacts on disk:**
- `outputs/reports/unified_viewer/index.html` — 77.3 MB (within the 70–80 MB envelope the phase doc cites; the in-process raw-bytes print of 81.01 MB includes the pre-gzip payload before HTML compression of surrounding tags).
- `outputs/reports/unified_viewer/edge_slices/` — `decomp_ols/`, `incytr_pathways/`, `song_concordance/` all present.
- `outputs/reports/unified_viewer/audit_sources/transcript_trace/` — 20+ per-cluster parquets (Astrocytes 3.1M, Excitatory-Pyramidal 3.1M, etc.).

**Smoke tick-list:**
- [x] Kinase Explorer — Family column wiring confirmed via `META.familyMap` (389 entries).
- [x] Kinase Human — Family column wiring confirmed; `celltype_specificity` present in payload.
- [x] **Crosstable** — TAB_MANIFEST entry present; panel slot present; JS parses; all PAYLOAD fields it consumes are present.
- [x] Kinase Audit — Build wrote `audit_tables` block to PAYLOAD.
- [x] Incytr Pathways — 961 pair shards written; trajectory annotation logs show v3 multi-label labels per pair; transcript trace shards present for Measurement Trace.

Live in-browser interaction smoke (pager controls, chip toggle, Recur-in AND-gate, Measurement Trace expand, column-visibility toggle) **was not exercised** — explicit follow-up for the user/next session before Phase 8 push.

**Go / no-go for Phase 8 (push):**
- **Go for the build + structure**: all artifacts present, all PAYLOAD fields the JS consumes resolved, no missing tab wiring.
- **Conditional on live browser smoke** by the user before Phase 8 actually pushes. The phase doc explicitly lists in-browser smoke as part of Phase 7; I can't execute it from this CLI. Phase 8 should be run by a human (or with browser-driver access) after a hard-refresh smoke of the five tab interactions listed above.

**Notes for Phase 8:**
- HEAD unchanged from end of Phase 6 (`76468d5` `docs(viewer): mark CR-03 adoption complete in audit + epic`). Main still **22 ahead, 1 behind** `public/main`.
- The Phase 6 footnote about pushing to `public` (not `origin`) remains the operating instruction.
- No new safety tag needed for Phase 7; the existing `pre-cr03-adopt-main` / `pre-cr03-adopt-branch` tags from Phase 0 still cover the rollback story.
- Untracked planning drafts (`incytr_mea_seed_list_expansion_plan.md`, `incytr_pathway_measurement_trace_plan.md`) still present; Phase 8 should not stage them.

### Phase 8 — Cleanup: push, delete branch, retain tags
**Agent:** Opus 4.7 (1M) · 2026-05-20
**Commits produced:**
- `73110df` docs(plan): record Phase 7 verification log + mark phase done — committed Phase 7's dirty doc trail (Phase 7 had ended without committing; this brings it under version control before publishing).
- (this commit, docs-only) — Phase 8 status `done`, epic status flipped from "done — pending Phase 7/8 verification + push" to plain `done`, Phase 8 log entry appended.

**Pushed main SHA (force-with-lease):** `73110df` (the Phase 7 docs commit). The Phase 8 docs commit produced by this entry will be pushed as a follow-up.

**Divergence from plan — push was non-fast-forward, required `--force-with-lease`:**
- The phase doc asserted "This is **not** a force-push — origin's main is an ancestor of the new main." That premise was wrong. Public `main` carried `a152c52` (Measurement Trace), which Phase 1's `git reset --hard` removed from new main's history. Phase 5 reapplied the same diff as cherry-pick `8046a2d` (new SHA, same content), so the *work* is preserved, but the *history* is rewritten — push had to overwrite, not append.
- `git push public main` rejected as non-fast-forward (as expected post-reset). Used `git push --force-with-lease public main` after user confirmation. Lease check passed; push succeeded.
- Per `[[project_solo_dev_repo]]` (saved this phase) the usual collaborator/CI caveats around force-push do not apply here — only user + Claude contribute, no downstream consumers. Recovery anchors (`pre-cr03-adopt-main`, `pre-cr03-adopt-branch`) still on public.

**Branch deletion:** `git branch -d feat/cr03-human-celltype-specificity` succeeded with plain `-d` (no `-D` needed) because every commit on the CR-03 branch is reachable from new main's history — the cleanest possible deletion. The branch did not exist on `public` (verified via `git ls-remote --heads public feat/cr03-human-celltype-specificity` → empty), so no remote delete was needed.

**Safety tags retained (verified on public):**
- `pre-cr03-adopt-main` → `02f0b64b1be1188bbad7381824f5bd894759a79a`
- `pre-cr03-adopt-branch` → `0cb721f4ad2af05e86581db646ba6f13a7af140b`

**Verification:**
- `git tag --list 'pre-cr03-*'` → both present locally.
- `git ls-remote --tags public 'pre-cr03-*'` → both present on public.
- `git branch -a | grep -i cr03` → no `feat/cr03-*` anywhere (the lingering `cr03-work` ref is an unrelated worktree-locked branch, not part of this epic).
- `git ls-remote --heads public feat/cr03-human-celltype-specificity` → empty.
- Browser hard-refresh check of the published viewer was **not** performed from this CLI session; the phase doc lists this as the final tick. User to verify `PAYLOAD.meta.generated_at` matches the Phase 7 build timestamp (`2026-05-20T20:10:56.749182+00:00`) on next browser session.

**Notes for the user:**
- Memory file `[[project_solo_dev_repo]]` was added this phase to capture the "no collaborators, no public consumers" context so future destructive-op caveats are appropriately scoped.
- Untracked planning drafts (`incytr_mea_seed_list_expansion_plan.md`, `incytr_pathway_measurement_trace_plan.md`) remain untracked — they are unrelated future-work drafts and were intentionally left alone through every phase.
- Epic is closed. The two worktree-locked branches (`cr03-work`, `feat/cr04-incytr-viewer`) and their associated worktrees under `.claude/worktrees/` are out of scope for this epic but visible in `git branch -a` / `git worktree list` if cleanup is desired separately.
