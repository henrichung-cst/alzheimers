# Phase 6 — Forward-port audit + epic docs

**Status:** done
**Depends on:** Phase 5
**Reversible:** yes.

## Goal

Cherry-pick the audit + epic + phase docs commit from Phase 0.3. Then update those docs to reflect that the adoption has completed.

## Preflight

```bash
git rev-parse HEAD                          # = Phase 5 HEAD
DOCS_SHA=$(...)                             # look up from your Phase 0.3 Implementation Log entry
git show --stat "$DOCS_SHA"                 # confirm it's docs/plans/* only
```

## Steps

### 6.1 — Cherry-pick

```bash
git cherry-pick <DOCS_SHA>
```

Should apply clean — all files are new under `docs/plans/`.

### 6.2 — Update the audit doc

Open `docs/plans/viewer_audit_2026-05-20.md`. In the "Regressions / lost features" section:
- Reclassify the entries as **"Features adopted from CR-03 branch via the reconcile epic"**.
- Link to `docs/plans/epic_reconcile_cr03_branch.md` and the final commit SHA from Phase 5.
- Keep the rest of the audit (memory hazards, punch list) intact — those are still actionable.

### 6.3 — Update the epic doc

Open `docs/plans/epic_reconcile_cr03_branch.md`:
- Change `**Status:** in progress` to `**Status:** done — pending Phase 7/8 verification + push`.
- Do not delete the phase table or Implementation Log; they are the historical record.

### 6.4 — Commit the doc updates

```bash
git add docs/plans/viewer_audit_2026-05-20.md docs/plans/epic_reconcile_cr03_branch.md
git commit -m "docs(viewer): mark CR-03 adoption complete in audit + epic"
```

## Verification

```bash
git log --oneline -10
# Recent commits should be: docs-update, a152c52 cherry-pick, pagination, 5609eab, 56fd2bf,
# then the 11 branch commits below.
```

## What the next phase needs from you

In your Implementation Log entry:
- New HEAD SHA.
- Confirmation that both docs were updated and the update commit landed.
