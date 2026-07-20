# Plans — implementation sequencing & adjudication

Goal: implement every plan in `docs/plans/`. This maps the cross-cutting threads, the blocking
decisions, the conflicts, and a parallel execution order.

Plan shorthand used below:
`geneuse` = `ad_geneuse_unpin_from_sce4`
`apriori` = `tcell_apriori_expectations` (blind exhaustion prediction set → `docs/reference/`, unbuilt)
`deploy` = `deployment/todo9_viewer_aws_deployment` · B2/C3/E1/E2/G1/H1/I2 = backlog files.

---

## 1. Cross-cutting threads (shared substrates)

**T1 — kinase→kinase network.** The kinase→kinase edge model ships in
`alz/cross_reference/kinase_kinase_edges.py` (PSP `Kinase_Substrate_Dataset` ∪ cohort motif
co-enrichment → weighted, provenance-tagged interactome; spec in
`foundation/kinase_sidechain_incytr_graph.md`). `E1` — a reference regulation hierarchy (A↑⇒B↓) with
an observed disease-direction overlay — is the one remaining consumer. It **reuses that backend rather
than building a parallel network**: the Johnson Kinase Library is a motif-scoring atlas and cannot emit
kinase→kinase edges, so PSP is the forced source, not a preference (D2).

**T2 — cell-type specificity engine.** `E2` consumes the existing per-kinase cell-type specificity
basis to split same-family kinases — the only open consumer. The review-time cross-species breadth
constraint lives in `foundation/specificity_confidence.md` §3a.

**T3 — Incytr recompute + viewer rebuild.** `geneuse` (un-pin → AD re-run), `B2`, `C3`, and `E1/E2`
tabs all end in a viewer payload rebuild. Recomputes are
**not DB-blocked**: the six IncytrDB objects are version-pinned (2026-03-09, MD5s in the audit manifest
at `~/Projects/work/incytr/inst/extdata/incytrdb_audit/`) and species selection is per-dataset correct;
a first-principles DB rebuild is blocked on source artifacts that no longer exist, which the no-contact
rule forbids requesting. Payload writers (`build_unified_viewer.py`, `build_tcell_viewer.py`,
cohort `*.py`, `slices_incytr.py`) cannot be written concurrently — serialize integration, not authoring.

**T4 — T-cell labeling standard (SETTLED, nothing outstanding).** Canonical = **per-cell marker
assignment** (`alz/analysis/tcell_state_labels.py`, via `pixi run tcells-label`) → the 12-state CD4/CD8
vocabulary in `outputs/reports/tcell_labeling/cells/{donor}_state_labels.csv`. By-cluster annotation was
**rejected**: native Seurat clusters are day-confounded (it called 59.6% of donor1 day-2 and 69.2% of
donor2 day-2 cells "activated" purely by cluster occupancy). ProjecTILs is **corroboration-only** — it
lands in the `projectils_*` evidence columns and never sets a label.

The standard is enforced in code, not by convention: one producer, and one shared validator
(`alz/ingest/tcells_state_labels.R`) that every consumer routes through and that hard-fails on
barcode/day/cluster drift or any type outside the 12-value vocabulary. There is no competing live
labeling path. This is the current **default, not a lock** — re-running `tcells-label` re-derives
everything downstream, so a deliberate future re-labeling is a supported operation.

The whole t-cell chain is built on this standard and consistent with it: Incytr `wide/` (donor1 ×3,
donor2 ×4), the viewer (`build_tcell_viewer.py` asserts `incytr_pair_mode_tcells` is the
resolver default), and the report suite. No t-cell item is open against this standard.

---

## 2. Adjudications required (blocking — resolve before the dependent wave)

**D2 — E1 scope. RESOLVED: shared backend.** E1's reference hierarchy (A→B regulation) is sourced from
PSP's `Kinase_Substrate_Dataset`, which `kinase_kinase_edges.py` already loads. The Johnson Kinase
Library is a motif-scoring atlas and cannot emit kinase→kinase edges, so it is not a viable alternate
source; the data source is forced, not a preference. E1 reuses `kinase_kinase_edges.py` and renders as
a directional/disease-direction **overlay layer** on the existing sidechain interactome. The host
surface now exists — the sidechain view is a sub-tab of the Incytr pathways detail panel — so E1 adds
an overlay mode within it and splits to its own tab only if the UX is cramped.

**D3 — geneuse un-pin. RESOLVED: un-pin.** AD switches to the derived `DEG∪prG` recipe (same path as
t-cells); single cross-cohort recipe, wider enumeration (downstream SigProb/|PDS| filter still applies).
Removes the `use_frozen_geneuse` branch + `extract_sce4_geneuse.R` (anti-shim). Production-output-
changing — requires a full AD re-run + viewer rebuild (Wave 3). Touch points in
`ad_geneuse_unpin_from_sce4.md`.

**D5 — I2 timing. UNBLOCKED.** The kinase-sidechain work that imported the frozen-layer modules `I2`
relocates (`*_decompose.py`, `song.py`, R extractors) has landed, so the gate that held `I2` back is
gone. It remains an **isolated-window** job: relocating those modules breaks every importer at once, so
run it alone, updating all importers in the same pass, and never interleave it with other file-touching
plans.

---

## 3. Dependency graph

```
D3 ──► geneuse re-run ──► AD viewer rebuild ─────────────────────┐
                                                                 ├─► serialized
D2 ──► E1 (on kinase_kinase_edges.py) ──► E2 ────────────────────┤   viewer
T2 specificity ──► E2                                            │   integration
B2 (sankey), C3 (early-change) ──────────────────────────────────┘

tmt (H1), G1 (diagrams), apriori (docs), deploy-B ── independent leaves, no gate
I2 ── isolated window (D5)
```

---

## 4. Conflicts

- **I2 ↔ everything file-touching** — relocates frozen-layer modules the R pipeline and the
  cross-reference layer import (D5). Isolate.
- **Viewer payload write contention** — B2, C3, E1/E2, and the geneuse rebuild all write the payload
  builders. Not a logic conflict; serialize the *integration/rebuild* step (or use worktree isolation
  for authoring, single-threaded merge).
- **geneuse re-run ↔ AD-inclusive payloads** — the AD Incytr outputs change; any payload built during
  the re-run window is stale. Sequence the AD rebuild after the re-run completes.

---

## 5. Parallel execution waves

**Wave 1 — independent builds (parallel, disjoint outputs, no gate).**
- `H1` TMT paper IMAC fetch + enrichment.
- `G1` workflow diagrams (docs).
- `deploy` Option B (shard-hash manifest) — infra layer, separate from analysis.
- `apriori` blind exhaustion prediction set (docs) — literature-grounded, authored before final
  outputs; feeds the `/check-controls` plausibility skill. Blindness caveat: the t-cell
  Incytr report suite is already built, so ground the predictions in literature only, not those outputs.

**Wave 2 — kinase backends + specificity thread.**
- `E1` on the shared `kinase_kinase_edges.py` backend (D2), then `E2` (needs T2 specificity).

**Wave 3 — Incytr recompute.**
- `geneuse` un-pin (D3) → AD re-run → AD viewer rebuild.

**Wave 4 — viewer tabs (serialize the payload/integration step).**
- `B2` sankey, `C3` early-change tab, `E1/E2` tab(s). Author in parallel; merge and rebuild one at a
  time.

**Isolated — `I2`** frozen-layer namespace moves, dedicated window (D5).

Nothing is blocked on an open decision: D2, D3, and D5 are settled and direct the waves above.
