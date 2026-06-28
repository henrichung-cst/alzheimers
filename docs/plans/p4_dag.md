# P4 — Dependency DAG + git topology

Finalizes the execution graph for the orchestration experiment: which themes run in which wave, what serializes against what, where the heavy compute sits, and the git/worktree mechanics. Grounded in the harvested cross-theme edge list (not the meta_plan's original 4-wave guess, which missed the `C2→C1→C3` and `D1→C5` chains).

In-scope themes: **A, B4, B5, C1, C2, C3, C5, D1, F1, F2, G2.** Excluded: B1/E/H (blocked). Tail (separate gates, after the waves): B2 sankey, B4.2 viewer stubs, G1 docs.

---

## Locked decisions

- **Integration = merge-to-main-per-gate.** Each theme branch merges into `main` at its wave gate. Wave 1 branches off the baseline tag; every later wave branches off the updated `main` tip (so it inherits prior waves' merged edits and the canonical schema). No long-lived integration branch.
- **Baseline tag:** `git tag orchestration-baseline-2026-06-25` on `main` before Wave 1. Total-abort = `git reset --hard` to this tag (destructive — **user-initiated only**, never an agent).
- **Branch-per-theme, worktree-isolated.** One `git worktree add` + `theme/<x>` branch per parallel agent. Atomic commit per unit within the branch. Per-theme revert = `git revert` the theme's merge commit on `main`.
- **Heavy compute never runs inside a parallel worktree agent.** All gate-compute (regen of *shared* artifacts) runs once in the **primary tree** at the wave gate, under the memory cap. Parallel agents do code edits + light verification (`pixi run viewer`) only.

## Worktree infra rule (critical — validated by dry-run against the baseline tag)

`git worktree add` produces a checkout containing **only tracked files** — and on this repo that strips four things a working agent needs. Setup, per worktree, immediately after `git worktree add`:

```
git worktree add --detach <wt> orchestration-baseline-2026-06-25
git -C <wt> -c protocol.file.allow=always submodule update --init --recursive   # (2)+(3)
ln -s <primary>/data    <wt>/data            # (1) gitignored, shared read-only
ln -s <primary>/outputs <wt>/outputs         # (1) gitignored; agents write disjoint subdirs
pixi install --manifest-path <wt>/pixi.toml  # (4) own env from tracked lock — NOT a shared symlink
# when running pixi: export CONDA_OVERRIDE_CUDA=""  (avoids the nvidia-smi hang outside direnv)
```

The four traps, each of which would otherwise fail all parallel agents at once:
1. **`data/`+`outputs/` are untracked** (ignore-all gitignore) ⇒ a fresh worktree has no inputs/artifacts. Symlink from primary. Agents treat `data/` **read-only** and write only their own **new** `outputs/reports/<new>/` subdir (D1/B5/C5 disjoint — safe to share the parent).
2. **`vendor/rclone-ingest` is a git submodule**, empty until `submodule update --init`. No wave *uses* it, but `pixi.toml` declares it an editable dep, so the env won't build unless the path is a valid project. Init it; do **not** strip it from `pixi.toml` (that breaks the primary's ingest tasks).
3. **That submodule's remote is a `file://` path**, blocked by git's default `protocol.file.allow=never`. Pass `-c protocol.file.allow=always`.
4. **`.pixi` cannot be shared via symlink** — pixi mutates the env (rebuilds the editable dep) per invocation, so parallel agents on one shared env would race. Each worktree gets its **own** `pixi install` — materialized from the tracked `pixi.lock` (no re-solve, ~seconds), ~4 GB/worktree (real copy) ⇒ ~24 GB transient for the six Wave-1 agents, reclaimed on `git worktree remove --force <wt>`.

Gate-compute that **overwrites** a shared artifact (snrna, bulk_mea recover, nsclc regen, state_mea) runs in the primary tree at the gate — never concurrently in a worktree — so there is no concurrent-overwrite hazard on the shared `data/`+`outputs/`.

---

## The DAG

```
baseline tag
   │
   ├─ C2 ─────────────┐                     Wave 1 (off baseline)
   ├─ B5 ····(prov.)   │   B4 ──(recep_em_fan.csv)··┐
   ├─ D1 ───────────┐  │   A   G2                   │
   │                │  │                            ·(B4↔B5 key reconcile @ W1 gate)
   ▼ merge to main  │  ▼                            ▼
   ├─ C5 ◄── D1     │  C1 ◄── C2                     Wave 2 (off main)
   ▼ merge          │  ▼
   │                └─ C3 ◄── C1, C2                 Wave 3 (off main)
   ▼ merge
   └─ F1 → F2  ◄── C1, C2, C3, A (sweep)             Wave 4 (off main, F1 then F2)
   ▼ merge
   tail: B4.2 ◄── B5,C1   ·   B2 ◄── B5   ·   G1 ◄── all
```

Hard edges (consumer cannot start until producer is merged): `C2→C1`, `C2→C3`, `C1→C3`, `C2→F2`, `C1→F2`, `D1→C5`. Soft edges (ordering only, no data dep): `[C1,C3]→F1/F2` (sweep-after-table-adders), `B4→B5` (B5 runs provisional, reconciled at the W1 gate).

---

## Wave assignment

| Wave | Themes (parallel agents) | Branch base | Within-wave serialization | Gate-compute (primary tree) |
|---|---|---|---|---|
| **1** | C2, B5, D1, B4, A, G2 | baseline tag | `pixi.toml` edited by B5+D1+B4 → merge the three branches **sequentially** at the gate (additions are non-overlapping table rows; keep all). Otherwise disjoint (C2=unified tree, A=tcell tree, B5/D1/B4/G2=backend). | `pixi run snrna` (B5); `nsclc_expression.py` regen + `state_mea.py` donor1 (A) — all memory-capped |
| **2** | C1, C5 | `main` @ W1 | Disjoint (C1=unified viewer+pipeline, C5=backend; C5 appends its `pixi.toml` task onto the merged W1 additions) | bulk_mea recover (C1) → `peak_NES_{g}`, `n_sig_{g}`, `trajectory_{g}`; verify `summary.py` |
| **3** | C3 | `main` @ W2 | Solo (hard-needs C1's merged payload) | none (bounded reads) |
| **4** | F1 **then** F2 | `main` @ W3 | **F1 and F2 edit the identical JS file set → not parallel.** One worktree, two atomic commits: F1 first (signed comparator / `numCmp`), F2 second (adopts F1's signed `_exportPeakAbsNes`). | none |
| **tail** | B4.2, then B2, then G1 | `main` @ W4 | sequential, each its own gate | B4.2: none; B2: viewer rebuild; G1: confluence |

**Why these tiers, not 4 flat waves:** C2→C1→C3 is a 3-deep hard chain, so those three occupy three successive waves. D1→C5 puts C5 in Wave 2. F1/F2 collide with *each other* on every kinase-tab JS file, so they serialize within Wave 4 rather than fan out. A and B4 have no in-scope upstream and carry heavy compute / new backends, so they front-load into Wave 1.

---

## File collision registry → how each is resolved

Cross-wave collisions are **auto-resolved by wave ordering**: by the time a later wave's agent branches off `main`, the earlier editors are already merged, so it edits on top with no live conflict. The only collisions needing explicit handling are *within* a wave:

| File | Editors (wave) | Resolution |
|---|---|---|
| `pixi.toml` | B5,D1,B4 (W1); C5 (W2) | Sequential merge at each gate; keep all added task rows (non-overlapping). |
| `kinase_{crosstable,explorer,audit}.js` | C2(W1), C1(W2), F1/F2(W4) | Wave ordering — each branches off the merged prior. |
| `kinase_fivexfad.js`, `incytr_pathways.js`, `kinase_human.js` | C2(W1), F1/F2(W4) | Wave ordering. |
| `06_export_csv.js` | C3(W3 seeds `numCmp`), F1/F2(W4 adopt) | Wave ordering; F1 adopts C3's seed, no redefine. |
| `song.py` | C1(W2), C3(W3); B4.2(tail) | Wave ordering (C1 first). |
| `build_unified_viewer.py`, `01_state.js`, `body.html`, `02_ui_chrome.js` | C2(W1), C3(W3); F1 also `01_state.js`(W4) | Wave ordering. |
| `tcell_viewer/.../kinase_{audit,explorer}.js` | A(W1), F1/F2(W4) | Wave ordering. |

No two *concurrent* (same-wave) agents edit the same source file except `pixi.toml`.

---

## Gate procedure (run at every wave boundary)

1. **Merge** the wave's theme branches into `main` (sequentially; resolve `pixi.toml` by keeping all task rows).
2. **Gate-compute** (if any, per the wave table) in the primary tree, under `systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0`.
3. **Verify:** `python alz/bulk_mea/summary.py` and/or the relevant harness; for viewer waves, `pixi run viewer` + **hard-refresh browser click-through (human)** — visual changes are authoritative.
4. **Reconciliation checkpoints:**
   - *W1 gate:* compare B5's provisional backbone 6-tuple against B4's merged `recep_em_fan.csv`. If they disagree, schedule B5's key fix before B4.2.
   - *W4 gate:* confirm the F1/F2 sweep covers C3's new `diseasedirection` table.
5. **Tag** the post-merge state (`orchestration-w<N>-2026-06-25`) so a later wave's failure can roll back to a clean wave boundary, not just the baseline.
6. Cut the next wave's worktrees off the updated `main` tip, each via the full per-worktree setup (submodule init + `data/`+`outputs/` symlinks + own `pixi install`) from the infra rule above.

## Revert granularity

- **One unit:** `git revert` its commit on the theme branch (pre-merge) — local.
- **One theme:** `git revert` the theme's merge commit on `main`, or reset `main` to the wave-boundary tag if still at the gate.
- **One wave:** reset `main` to the previous `orchestration-w<N>` tag.
- **Everything:** reset `main` to `orchestration-baseline-2026-06-25`. Destructive — **user-initiated only.**

---

## Tail (post-wave, separate gates)

- **B4.2** — viewer stubs (`#Backbones` column, `Driving kinases` panel) + reconcile orphaned preamble counts. Hard-needs B5's settled backbone key + C1's merged `song.py`. Gate after the W1 B4↔B5 reconciliation resolves.
- **B2** — sankey viewer builder; gated after B5.
- **C3-S4** — Disease Direction Stage 4: site-level early-change. New build step reads `site_level_ols.csv` (DuckDB/chunked, memory-safe), classifies each `site_id` per genotype as *early in g* iff `stoich_fdr_{g}_2mo < MEA_FDR_THRESH` AND not `< thresh` at `{g}_4mo`/`{g}_6mo`; emits a per-gene shard (`gene_symbol → early_sites[]` with genotype + 2mo LFC). Biomarker panel (C3 Stage 3) then **flips its row entity from kinase genes to substrate genes**, gains an "early sites" sub-column joined with the secretome flag (the real "early + secretable" view), and folds in the "substrate gene" label correction. Non-kinase substrate-row LFC = per-site `stoich_lfc` from the shard (kinase rows keep `top_celltype_1_song_lfc`); omit honestly where absent. Spec: `theme_c/c3_plan.md §Stage 4`. Gated after W3 (C3 merged @ `orchestration-w3-2026-06-25`).
- **G1** — methods/workflow docs + diagrams; depends on everything; **last**; confluence skill.

---

## Status / next

P1–P5 complete. Baseline tagged (`orchestration-baseline-2026-06-25` → `35edf31`); the worktree+submodule+env round-trip is validated (the recipe above is what it produced). Ready to launch Wave 1 as one `Workflow` invocation (worktree-isolated agents); the human gate sits between waves. Execution is **not** part of P4 — this doc is the gate for starting it.
