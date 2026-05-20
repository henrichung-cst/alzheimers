# Phase 3 — Forward-port `5609eab` (trajectory chips + Recur-in filter, viewer side)

**Status:** done (no-op — features already present and redesigned on branch)
**Depends on:** Phase 2
**Reversible:** yes via `git reset --hard HEAD~1` (cherry-pick is local-only at this point).

## Goal

Bring main's CR-04 viewer-side trajectory chips, Recur-in filter, and temporal-detail bar chart into `incytr_pathways.js` on top of the branch's heavier rewrite of the same file. This is the conflict-heaviest phase.

## Context

The branch has 650 lines of changes in `alz/viewer/template/js/tabs/incytr_pathways.js` (perf commits + factorial→pair-mode adaptations). Main's `5609eab` added UI: trajectory pill chips bound to per-row trajectory_index (from Phase 2), a "Recur-in" filter, and a temporal-detail bar chart.

The branch's perf commits (`c498c42`, `f39e661`) used a row-cap / parallel-fetch loader pattern. That pattern is **gone** as of the Phase 4 pagination commit, but in Phase 3 it is still present in the file. Don't touch it here — Phase 4 handles it. Your job is only to add the trajectory chips + Recur-in filter + temporal detail without breaking the existing structure.

## Preflight

```bash
git rev-parse HEAD                          # = Phase 2 HEAD
git show --stat 5609eab                     # files touched
git show 5609eab -- alz/viewer/template/js/tabs/incytr_pathways.js | head -80
```

Re-read your Phase 2 Implementation Log entry — it tells you which builder-side trajectory structure won, which determines which PAYLOAD field names the JS should bind to.

## Steps

### 3.1 — Cherry-pick

```bash
git cherry-pick 5609eab
```

### 3.2 — Resolve conflicts

Almost certain conflict in `alz/viewer/template/js/tabs/incytr_pathways.js`. Resolution principle:

- **Keep the branch's loader structure** (even though it's about to be replaced in Phase 4 — don't conflate phases).
- **Add the trajectory chip UI**, Recur-in filter, and temporal-detail bar chart as new code blocks. They are largely additive — chips render in a new container, Recur-in is a new filter row, the bar chart is a new widget. They don't fundamentally conflict with loader internals.
- Bind chips/filter to the PAYLOAD field names produced by Phase 2's builder (record those names in the Phase 2 log entry if not already there).
- If `5609eab` also touched the loader internals (it may have, since it predated `03dcf60`), prefer the branch's loader and re-implement the chip-related loader hooks on top.

Other files touched by `5609eab` — likely `styles.css`, possibly `body.html`. These are usually clean.

### 3.3 — Verify

```bash
pixi run python -c "import ast; ast.parse(open('alz/build_unified_viewer.py').read()); print('builder ok')"
node -e "new Function(require('fs').readFileSync('alz/viewer/template/js/tabs/incytr_pathways.js','utf8'))" \
  2>&1 | head -5
# Pure-parse check; rejects syntax errors. (If `node` not in env, skip.)
```

If `outputs/reports/unified_viewer/unified_viewer.payload.json` exists:

```bash
pixi run python alz/build_unified_viewer.py --html
# Open the page, hard-refresh (Ctrl+Shift+R). Confirm chips render and don't crash.
```

## Failure handling

If conflicts get tangled:

```bash
git cherry-pick --abort
```

Two fallback options (pick one, document in log):
1. Take the branch version of `incytr_pathways.js` wholesale and re-author the trajectory chips + Recur-in filter + bar chart as a fresh commit — the JS additions are small and well-scoped.
2. Defer Phase 3 until after Phase 4 — once pagination is in, the loader surface is simpler and the chip integration may be easier. (Only choose this if Phase 4 doesn't depend on chip-related code.)

Stop and ask before committing either fallback.

## What the next phase needs from you

In your Implementation Log entry:
- New HEAD SHA.
- Which fallback was needed (if any).
- Confirmation that the existing branch loader was preserved unchanged (Phase 4 will remove it cleanly; if you started replacing it here, Phase 4 needs to know).
- Any JS bindings that depended on a PAYLOAD field name choice from Phase 2.
