# Phase 8 — Cleanup: push, delete branch, retain tags

**Status:** done
**Depends on:** Phase 7 (must end with an explicit go).
**Reversible:** push is partially reversible; branch deletion is reversible only via the `pre-cr03-adopt-branch` tag.

## Goal

Publish the reconciled main, delete the obsolete branch locally + on origin, retain the safety tags indefinitely.

## Preflight

Verify Phase 7 left an explicit go in its Implementation Log entry. If it didn't, stop — do not push on partial verification.

```bash
git rev-parse HEAD                              # final reconciled main
git status                                      # clean
git log --oneline main..origin/main             # empty — main is ahead of origin
git log --oneline origin/main..main             # the 11 branch commits + 5 forward-port commits
```

## Steps

### 8.1 — Push main

```bash
git push origin main
```

This is **not** a force-push — origin's main is an ancestor of the new main (the branch commits all branched off `03dcf60` which is in origin main's history; the forward-port cherry-picks are on top). Standard fast-forward push.

If push is rejected as non-fast-forward, stop and investigate. Do NOT use `--force` without an explicit instruction from the user.

### 8.2 — Delete the CR-03 branch

```bash
git branch -d feat/cr03-human-celltype-specificity
# -d (lowercase) refuses if the branch is not merged into HEAD. Since we cherry-picked
# its commits' content but not their SHAs, -d will fail. That's expected.

# After confirming the safety tag `pre-cr03-adopt-branch` exists on origin:
git ls-remote --tags origin pre-cr03-adopt-branch  # MUST return a line
git branch -D feat/cr03-human-celltype-specificity
git push origin --delete feat/cr03-human-celltype-specificity
```

### 8.3 — Retain safety tags

Do **not** delete `pre-cr03-adopt-main` or `pre-cr03-adopt-branch`. They are the recovery anchors.

```bash
git tag --list 'pre-cr03-*'                     # both present locally
git ls-remote --tags origin 'pre-cr03-*'        # both present on origin
```

## Verification

```bash
git log --oneline -20
git branch -a                                   # no feat/cr03-* anywhere
```

Final check: open the published viewer once more in a fresh browser tab (no cache). Confirm `PAYLOAD.meta.generated_at` matches the Phase 7 build timestamp.

## Failure handling

If push fails (non-fast-forward):

```bash
git fetch origin
git log --oneline main..origin/main             # what landed on origin while we were working?
```

If origin has new commits, stop and bring the user in. Do not rebase or merge without an explicit instruction.

If branch deletion fails or you accidentally lose tags:

```bash
# Recover the branch from origin:
git fetch origin pre-cr03-adopt-branch
git branch feat/cr03-human-celltype-specificity pre-cr03-adopt-branch
```

## What the next phase needs from you

This is the terminal phase. In your Implementation Log entry:
- Final pushed main SHA.
- Confirmation that both safety tags survive on origin.
- Confirmation that the CR-03 branch is gone from local and origin.
- Mark the epic Status as `done` (not "done — pending verification").
