# Phase 0 — Safety anchors

**Status:** done
**Depends on:** none
**Reversible:** yes (no destructive ops, only commits + tags)

## Goal

Commit the uncommitted main-side work in clean, separable commits; discard the path-rename diffs that the branch already carries; tag both endpoints for recovery; push tags.

## Preflight

```bash
git status                              # confirm we are on main
git rev-parse HEAD                      # should match epic's "main tip" reference
git rev-parse feat/cr03-human-celltype-specificity  # branch exists locally
```

Expected modified files at start (from epic-time `git status`):
- `M alz/build_unified_viewer.py`
- `M alz/config.py`
- `M alz/decomposition/paths.py`
- `M alz/integration/build_cluster_spine.py`
- `M alz/integration/build_transcript_trace.py`
- `M alz/integration/build_yuyu_kldata.py`
- `M alz/integration/config_integration.py`
- `M alz/integration/diagnostics/dropout_coverage.py`
- `M alz/integration/diagnostics/na_zero_blast_radius.py`
- `M alz/integration/diagnostics/recompute_sigprob_d095.py`
- `M alz/integration/export_factorial_inputs.py`
- `M alz/integration/extract_cluster_assignments.R`
- `M alz/integration/plot_cluster_spine.py`
- `?? docs/plans/incytr_mea_seed_list_expansion_plan.md`
- `?? docs/plans/incytr_pathway_measurement_trace_plan.md`

Additionally, on disk but not in `git status` because they are inside `alz/viewer/template/js/tabs/`: the `incytr_pathways.js` per-shard pagination rewrite, the audit doc `docs/plans/viewer_audit_2026-05-20.md`, the epic doc `docs/plans/epic_reconcile_cr03_branch.md`, and the phase docs under `docs/plans/phases/cr03_adopt/`. Re-run `git status` to see the full picture; the list above is only the snapshot at epic creation.

## Steps

### 0.1 — Classify modified files

For each modified file, inspect `git diff <file>` and decide:
- **Rename-only** (only `data/incytr/` → `data/incytr_frozen/`): **discard** — the branch already does this in `3428a1b`.
- **Real change**: stage for a topic commit below.

Expected outcome: the 11 path-rename files in the list above should all be rename-only. The two `?? docs/plans/incytr_*_plan.md` files are unrelated drafts — leave them untracked unless you want a fourth commit for them.

```bash
for f in alz/config.py alz/decomposition/paths.py alz/integration/build_cluster_spine.py \
         alz/integration/build_yuyu_kldata.py alz/integration/config_integration.py \
         alz/integration/diagnostics/dropout_coverage.py \
         alz/integration/diagnostics/na_zero_blast_radius.py \
         alz/integration/diagnostics/recompute_sigprob_d095.py \
         alz/integration/export_factorial_inputs.py \
         alz/integration/extract_cluster_assignments.R \
         alz/integration/plot_cluster_spine.py; do
  echo "=== $f ===" && git diff -- "$f"
done
```

If they are exclusively the rename: `git checkout -- <files>`. If any contains non-rename hunks, stop and add a note to the Implementation Log — do not blindly discard.

### 0.2 — Commit the per-shard pagination JS work

Files: `alz/viewer/template/js/tabs/incytr_pathways.js` (rewritten in the previous session to per-shard pagination), plus `alz/build_unified_viewer.py` and `alz/integration/build_transcript_trace.py` if they have non-rename hunks (DuckDB env-var memory limit + per-pair streaming + transcript-trace per-cluster sliced rewrite).

```bash
git add alz/viewer/template/js/tabs/incytr_pathways.js \
        alz/build_unified_viewer.py \
        alz/integration/build_transcript_trace.py
git commit -m "fix(viewer): per-shard pagination + env-controlled memory budget

Replaces the row-cap / parallel-fetch Incytr loader with per-shard pagination
(≤1 sender×receiver pair resident, 100 rows/page). DuckDB memory_limit is now
controlled by VIEWER_DUCKDB_MEMORY_LIMIT (default 4GB), and the transcript
trace builder slices per cluster instead of materialising the full long-form."
```

Capture the SHA for Phase 4.

### 0.3 — Commit the audit + epic + phase docs

```bash
git add docs/plans/viewer_audit_2026-05-20.md \
        docs/plans/epic_reconcile_cr03_branch.md \
        docs/plans/phases/cr03_adopt/
git commit -m "docs(viewer): audit + CR-03 reconcile epic and phase plans"
```

Capture the SHA for Phase 6.

### 0.4 — Tag and push

```bash
git tag pre-cr03-adopt-main main
git tag pre-cr03-adopt-branch feat/cr03-human-celltype-specificity
git push origin pre-cr03-adopt-main pre-cr03-adopt-branch
```

Do **not** push main yet — main is about to be hard-reset in Phase 1.

## Verification

```bash
git status                              # working tree clean
git log --oneline -5                    # last two commits = pagination, then docs
git tag --list 'pre-cr03-*'             # both tags present
git ls-remote --tags origin pre-cr03-adopt-main pre-cr03-adopt-branch  # both on origin
```

## What the next phase needs from you

In your Implementation Log entry, record:
- The two new commit SHAs (Phase 0.2 pagination, Phase 0.3 docs) — Phase 4 and Phase 6 cherry-pick these.
- Confirmation that the 11 path-rename files were rename-only (or what else they contained).
- Anything left untracked that the next agent should know about.
