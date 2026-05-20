# Phase 5 — Forward-port `a152c52` (Measurement Trace panel)

**Status:** done
**Depends on:** Phase 4
**Reversible:** yes via `git reset --hard HEAD~1`.

## Goal

Cherry-pick the Measurement Trace panel (per-row expandable transcript pseudobulk widget) onto the now-paginated Incytr Pathways tab.

## Context

`a152c52` adds:
- `alz/integration/build_transcript_trace.py` (new file — per-cluster transcript pseudobulk shards)
- `alz/viewer/template/js/widgets/transcript_trace.js` (new widget)
- `alz/viewer/template/js/tabs/incytr_pathways.js` (wires the widget into row expansion)
- `alz/viewer/template/styles.css` (panel styling)
- `alz/viewer/paths.py` (transcript_trace dir + index constants)
- `alz/build_unified_viewer.py` (calls the trace builder; 33 lines added)
- `alz/viewer/template/index.html.j2` (script tag for the widget)

The branch's `incytr_pathways.js` already has its own row-expansion infrastructure for pair-mode. The Measurement Trace row-expand wiring must integrate with whatever expansion model survived Phase 4.

## Preflight

```bash
git rev-parse HEAD                          # = Phase 4 HEAD
git show --stat a152c52
test -d data/incytr_frozen/v2_46clusters/provenance \
  && ls data/incytr_frozen/v2_46clusters/provenance/aggexp.csv 2>&1 \
  || echo "MISSING: aggexp.csv — Measurement Trace builder will hard-fail"
test -f data/incytr_frozen/v2_46clusters/provenance/yuyu_samplekey.csv \
  && echo "samplekey present" \
  || echo "MISSING: yuyu_samplekey.csv"
```

If either substrate file is missing, stop. The Measurement Trace builder hard-fails on missing inputs (by design — `[[feedback_no_intentional_wrong_behavior]]`). Note in the Implementation Log and ask the user how to obtain them before proceeding.

## Steps

### 5.1 — Cherry-pick

```bash
git cherry-pick a152c52
```

### 5.2 — Resolve conflicts

Expected conflicts:
- `alz/build_unified_viewer.py` — the trace-builder call must land somewhere sensible in the main build flow. After Phase 2's trajectory_index integration, the file structure differs from `a152c52`'s base.
- `alz/viewer/template/js/tabs/incytr_pathways.js` — the row-expansion hook must integrate with whatever expansion the branch had. If the branch had no row expansion, add the hook fresh.
- `alz/viewer/paths.py` — likely clean (new constants added in their own section).
- `alz/integration/build_transcript_trace.py` — should apply clean (new file).

Resolution principle: prefer the latest behavior (post-Phase-4 main) for shared structure; add the Measurement Trace wiring as additive code blocks.

### 5.3 — Verify

```bash
pixi run python -c "import ast; ast.parse(open('alz/build_unified_viewer.py').read()); ast.parse(open('alz/integration/build_transcript_trace.py').read()); print('syntax ok')"
```

If the rest of the pipeline supports it, do a full builder pass (this is the first phase where we actually exercise the transcript-trace shard writer):

```bash
pixi run python alz/build_unified_viewer.py 2>&1 | tail -30
```

Watch RSS via another terminal (`top -p $(pgrep -f build_unified_viewer)` — sample, don't poll). It must stay under 4 GB. If it spikes higher, stop and capture the trace point.

Hard-refresh viewer. Confirm:
- A row in the Incytr Pathways table has an expand affordance.
- Expanding it shows the transcript trace panel.
- The panel header notes "N=1 per arm (males-only)".

## Failure handling

If the full builder OOMs:

```bash
git reset --hard HEAD                       # keep the commit; just abort the run
```

Investigate the trace builder's per-cluster slicing (the rewrite was in Phase 0.2's pagination commit). Confirm it slices the wide matrix and writes parquet per cluster — if the cherry-pick reverted that to the old long-form explosion, fix it before declaring done.

## What the next phase needs from you

In your Implementation Log entry:
- New HEAD SHA.
- Whether the full builder pass succeeded; peak RSS observed.
- Whether the panel renders correctly in browser.
- Any conflict resolution Phase 6 should know about.
