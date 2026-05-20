# Phase 1 — Reset main onto branch tip

**Status:** done
**Depends on:** Phase 0 (both safety tags must exist on origin)
**Reversible:** yes via `git reset --hard pre-cr03-adopt-main`, but only locally — do NOT push until Phase 7 verify passes.

## Goal

Make main's history equal `feat/cr03-human-celltype-specificity`'s history. Subsequent phases cherry-pick the 4 main-only commits on top.

## Preflight

```bash
git status                              # MUST be clean
git tag --list 'pre-cr03-*'             # both tags present
git ls-remote --tags origin pre-cr03-adopt-main  # tag exists on origin (recovery survives local disk loss)
git rev-parse main                      # record current main SHA in your log entry
git rev-parse feat/cr03-human-celltype-specificity  # target SHA
git rev-list --count main..feat/cr03-human-celltype-specificity  # should be 11
git rev-list --count feat/cr03-human-celltype-specificity..main  # should be ≥4 (the work to forward-port + Phase 0 commits)
```

If `git status` is not clean, stop. Phase 0 should have produced a clean tree.

## Steps

### 1.1 — Reset

```bash
git checkout main
git reset --hard feat/cr03-human-celltype-specificity
```

### 1.2 — Smoke-check the new environment

```bash
git log --oneline -15
# Expect top of log to show CR-03 branch's commits:
#   0cb721f docs: update READMEs and integration doc for levy_t5 as sole spine
#   3ebac8e docs: prune superseded incytr plans and seed_list_labels
#   4e61332 refactor(human): move HBCA supercluster→levy_t5 crosswalk to CSV
#   ...
```

```bash
pixi install                            # pixi.toml + lockfile changed on the branch; must resolve clean
```

If `pixi install` fails, do NOT try `pixi install --force` blindly — read the error, identify the conflict, fix `pixi.toml` minimally, then re-run. Document the change in your Implementation Log entry.

```bash
pixi run python -c "import alz.config; import alz.human_celltype_attribution; import alz.human_reference_expression; print('imports ok')"
```

```bash
pixi task list                          # confirm new runners visible
```

### 1.3 — Spot-check that the disk state matches expectations

```bash
test -f outputs/reports/kinase_attribution_human/celltype_specificity.csv \
  && echo "CR03 output present" \
  || echo "CR03 output MISSING — Phase 7 sub-tab will be empty"
test -d data/incytr_frozen/v2_46clusters \
  && echo "incytr_frozen dir present" \
  || echo "incytr_frozen dir MISSING — Phase 5 measurement trace will fail"
test ! -f alz/integration/factorial.R \
  && echo "factorial files deleted (expected)" \
  || echo "factorial files still present — branch reset may have failed"
```

All three checks should print the OK message.

## Verification

```bash
git diff feat/cr03-human-celltype-specificity main  # empty
git status                                          # clean
```

## Failure handling

If anything goes wrong before you push:

```bash
git reset --hard pre-cr03-adopt-main
```

restores main to its pre-Phase-0 state minus the Phase 0 commits. To recover those:

```bash
git cherry-pick <phase-0.2-sha> <phase-0.3-sha>
```

## What the next phase needs from you

In your Implementation Log entry, record:
- `pixi install` outcome (clean / which conflicts, if any).
- Whether the three disk spot-checks all passed.
- The new main SHA (= branch tip SHA before Phase 2 cherry-picks).
- **Do not push main.** Phase 8 pushes after verification.
