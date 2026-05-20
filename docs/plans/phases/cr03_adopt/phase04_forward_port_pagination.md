# Phase 4 — Forward-port per-shard pagination commit

**Status:** done
**Depends on:** Phase 3
**Reversible:** yes via `git reset --hard HEAD~1`.

## Goal

Replace the branch's row-cap / parallel-fetch Incytr loader (still in place after Phase 3) with the per-shard pagination loop authored on main in Phase 0.2.

## Context

Phase 0.2 committed the per-shard pagination rewrite of `incytr_pathways.js`. That commit's SHA is in your Phase 0 Implementation Log entry — **read it before starting**.

The branch's loader (commits `c498c42`, `f39e661`) caps loaded pairs to 8, fetches in parallel, and concatenates client-side. This is the anti-pattern that caused the browser OOM. The pagination commit:
- Renames `_IP_ROW_CAP` → `_IP_PAGE_SIZE = 100`
- Adds `_ipRuntime.page` field
- Rewrites `_ipPairsInScope` to return ≤1 pair
- Rewrites `_ipEnsureShards` to single-shard load (no Promise.all, no concat, no row-cap)
- Rewrites `_ipRenderTable` for three states (empty / pair-picker / paginated)
- Adds pager controls above + below the table
- Resets page on sort/slider/chip changes
- Removes dead refs (`_IP_MAX_PAIRS`, `_IP_FETCH_CONCURRENCY`, `_mapWithConcurrency`)

This phase must enforce: **no row-cap pattern remains anywhere in `incytr_pathways.js`** — `[[feedback_no_intentional_wrong_behavior]]` forbids leaving a known-broken fallback path.

## Preflight

```bash
git rev-parse HEAD                          # = Phase 3 HEAD
PAGINATION_SHA=$(grep -A1 'Phase 0' docs/plans/epic_reconcile_cr03_branch.md | \
                 grep -oE '\b[0-9a-f]{7,40}\b' | head -1)
echo "pagination commit = $PAGINATION_SHA"  # cross-check with your Phase 0 log entry
git show --stat "$PAGINATION_SHA"           # confirm it's the right one
```

Confirm pre-state:

```bash
grep -nE '_IP_MAX_PAIRS|_IP_FETCH_CONCURRENCY|Promise\.all' \
  alz/viewer/template/js/tabs/incytr_pathways.js
# Should find matches — the branch's anti-pattern is still in place.
```

## Steps

### 4.1 — Cherry-pick the pagination commit

```bash
git cherry-pick <PAGINATION_SHA>
```

### 4.2 — Resolve conflicts

Expected major conflict in `incytr_pathways.js`. Resolution principle:

- **Take the pagination commit's version of the loader and renderer wholesale.**
- **Preserve Phase 3's trajectory-chip + Recur-in + temporal-detail additions** by re-grafting them onto the new loader structure (they were additive; the new loader exposes the same render hooks just with a different data source).
- The pagination commit's `_ipResetPage()` must be called from the chip-click and Recur-in-change handlers added in Phase 3.

If the conflict is unresolvable, fall back to re-authoring rather than carrying anti-patterns forward. See Failure handling.

### 4.3 — Enforce the invariant

```bash
grep -nE '_IP_MAX_PAIRS|_IP_FETCH_CONCURRENCY|_mapWithConcurrency|Promise\.all\([^)]*shard' \
  alz/viewer/template/js/tabs/incytr_pathways.js
# MUST return zero matches.
```

If any of those identifiers survive, the cherry-pick was incomplete. Either fix or abort.

### 4.4 — Verify

```bash
pixi run python alz/build_unified_viewer.py --html 2>&1 | tail -10
```

Open the viewer, hard-refresh. Open the Incytr Pathways tab. Confirm:
- With nothing selected: prompt to pick sender + receiver appears.
- After picking sender + receiver: paginated table renders with pager controls.
- Sorting / chip-click / Recur-in filter all reset the page to 0.

## Failure handling

If the conflict is tangled or the pagination cherry-pick leaves the file in a half-state:

```bash
git cherry-pick --abort
```

Fallback: re-author the pagination rewrite directly on top of the Phase 3 HEAD by reading the pagination commit's diff and applying its principles to the current file. This is acceptable — we wrote both halves recently and the diff is small. Commit message:

```
fix(viewer): per-shard pagination for Incytr pathways tab

Replaces the row-cap / parallel-fetch loader with single-shard loading +
client-side pagination. ≤1 sender×receiver pair resident at a time;
100 rows per page; pager controls above and below the table.

Forward-port of <PAGINATION_SHA> with manual reapplication after merge
conflicts with the branch loader and Phase 3 chip additions.
```

## What the next phase needs from you

In your Implementation Log entry:
- New HEAD SHA.
- Whether the cherry-pick succeeded or the re-author fallback was used.
- Confirmation of the four no-anti-pattern grep checks (paste the empty output).
- Confirmation that Phase 3's chips/filter/bar-chart still work in the rendered viewer.
