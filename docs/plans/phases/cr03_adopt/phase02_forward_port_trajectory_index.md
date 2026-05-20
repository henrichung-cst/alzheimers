# Phase 2 — Forward-port `56fd2bf` (trajectory_index, builder side)

**Status:** done (skipped — superseded; see Implementation Log)
**Depends on:** Phase 1
**Reversible:** yes via `git reset --hard HEAD~1` if the cherry-pick is the only commit since Phase 1.

## Goal

Bring main's CR-04 builder-side trajectory_index + recur_index computation onto the new (branch-tip) base, without losing anything CR-03 added to `build_unified_viewer.py`.

## Context

Commit `56fd2bf` ("feat(viewer): CR-04 trajectory_index + recur_index computation in build_unified_viewer") added per-pair trajectory annotation logic in `build_unified_viewer.py`. It is the **data** half of the trajectory-chip feature; Phase 3 brings the **UI** half (`5609eab`).

The CR-03 branch independently rewrote `_write_incytr_pathways` to per-pair streaming + later trajectory shard-column work. The 03dcf60 redesign on main superseded the original streaming approach. After Phase 1 you have the branch's version; Phase 2 grafts the trajectory_index computation onto it.

## Preflight

```bash
git rev-parse HEAD                                # = branch tip after Phase 1
git log --oneline HEAD --not pre-cr03-adopt-branch  # empty (no new commits yet)
git show --stat 56fd2bf                            # remind yourself of file scope
```

## Steps

### 2.1 — Cherry-pick

```bash
git cherry-pick 56fd2bf
```

### 2.2 — Resolve conflicts

Expected conflicts in `alz/build_unified_viewer.py`. Resolution principle:

- **Keep CR-03's per-pair streaming structure** and PAYLOAD shape — they are the canonical forward direction.
- **Add the trajectory_index + recur_index computation** at the appropriate point in the per-pair loop (after the trajectory annotation, before writing the shard).
- The `_annotate_trajectory_columns` helper from main may already be present in a similar form on the branch (via `03dcf60` if that landed on the branch — check `git log feat/cr03-human-celltype-specificity --oneline | grep -i trajectory`). If both exist, prefer the version from `56fd2bf` and adapt the call site.

If the conflict resolution is non-trivial:
1. Read both sides fully before resolving.
2. Write what you decided into your Implementation Log entry — Phase 3 will hit related conflicts in the JS half and needs to know which builder structure won.

### 2.3 — Verify the commit

```bash
git diff HEAD~1 HEAD -- alz/build_unified_viewer.py | head -100
# Sanity check: the diff should add trajectory_index/recur_index logic, not remove CR-03 features.
```

```bash
pixi run python -c "import ast; ast.parse(open('alz/build_unified_viewer.py').read()); print('syntax ok')"
```

If `outputs/reports/unified_viewer/unified_viewer.payload.json` exists from a previous build:

```bash
pixi run python alz/build_unified_viewer.py --html 2>&1 | tail -20
# --html mode skips DuckDB and shard writers; verifies the builder still parses & runs.
```

Do **not** run a full rebuild yet — that's Phase 7.

## Failure handling

If the conflict cannot be cleanly resolved:

```bash
git cherry-pick --abort
```

Then stop. Add a detailed Implementation Log entry explaining what conflicted, and propose either (a) a different graft strategy, or (b) re-authoring the trajectory_index logic on top of the branch as a fresh commit. The user must decide before you continue.

## What the next phase needs from you

In your Implementation Log entry, record:
- New HEAD SHA after cherry-pick.
- Conflict files + one-sentence summary of how you resolved each.
- Whether `_annotate_trajectory_columns` already existed on the branch (Phase 3 depends on knowing this).
- Whether the `--html` smoke test passed.
