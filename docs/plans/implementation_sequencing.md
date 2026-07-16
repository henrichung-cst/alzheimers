# Plans — implementation sequencing & adjudication

Goal: implement every plan in `docs/plans/`. This maps the cross-cutting threads, the blocking
decisions, the conflicts, and a parallel execution order. Companion to `plans_state_audit_2026-07-10.md`.

Plan shorthand used below:
`sidechain` = `kinase_sidechain_incytr_graph/` (01–04) · `geneuse` = `ad_geneuse_unpin_from_sce4`
`corr` = `tcell-proteome-transcriptome-correlation` · `labels` = `tcell-matt-report-restoration` (per-cell labeling standard)
`apriori` = `tcell_apriori_expectations` (blind exhaustion prediction set → `docs/reference/`, unbuilt)
`deploy` = `deployment/todo9_viewer_aws_deployment` · A3/A4/B1/B2/C3/E1/E2/G1/H1/I1/I2 = backlog files.

---

## 1. Cross-cutting threads (shared substrates)

**T1 — kinase→kinase network.** `sidechain-01` and `E1` both build a kinase→kinase graph from
PhosphoSite-family data with a viewer tab. `sidechain-01` = PSP `Kinase_Substrate_Dataset` unified with
MEA motif evidence → weighted interactome + per-pathway cytoscape sidechains. `E1` = a reference
regulation hierarchy (A↑⇒B↓) with an observed disease-direction overlay + click-kinase-neighbors tab.
**Resolved (D2): shared `kinase_kinase_edges.py` backend — E1's A→B hierarchy comes from PSP's
`Kinase_Substrate_Dataset`, the same file sidechain-01 loads (the Johnson Kinase Library is a motif
atlas and cannot emit kinase→kinase edges). E1 is a directional/disease-overlay layer on the sidechain
interactome, not a parallel network.**

**T2 — cell-type specificity engine.** `E2` consumes the existing per-kinase cell-type specificity
basis to split same-family kinases — the one remaining open consumer. (`C4`, the human-vs-mouse-breadth
review guard, cleared to `foundation/specificity_confidence.md` §3a; `G2`, positive controls, retracted
— the curated per-cohort control list is gone, replaced by the general in-the-moment plausibility
`check-controls` skill that sources expectations from external biology, not the specificity basis.)

**T3 — Incytr recompute + viewer rebuild.** `geneuse` (un-pin → AD re-run), `sidechain` (consumes
Incytr `wide/` output), `B2`, `C3`, `E1/E2` tabs, and the t-cell relabel all end in a viewer payload
rebuild. `B1` (IncytrDB version audit) is **covered** — the six DB objects are version-pinned
(2026-03-09, MD5s in `current_manifest.json`) and species selection is per-dataset correct, so
recomputes are not DB-blocked. Payload writers (`build_unified_viewer.py`, `build_tcell_viewer.py`,
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
resolver default), and the report suite. `sidechain-02` and `A3` are the only open t-cell items.

---

## 2. Adjudications required (blocking — resolve before the dependent wave)

**D1 — T-cell labeling ontology. RESOLVED** (see T4): canonical = per-cell marker assignment;
by-cluster annotation rejected as day-confounded; ProjecTILs corroboration-only. No longer an open
adjudication — the consolidation is enforced in code by the shared validator, and the labels are
verified current. Nothing outstanding.

**D2 — E1 vs sidechain scope. RESOLVED: shared backend.** E1's reference hierarchy (A→B regulation)
is sourced from PSP's `Kinase_Substrate_Dataset` — the same file `sidechain-01` already loads. The
Johnson Kinase Library is a motif-scoring atlas and cannot emit kinase→kinase edges, so it is not a
viable alternate source; the data source is forced, not a preference. E1 reuses `kinase_kinase_edges.py`
and renders as a directional/disease-direction **overlay layer** on the sidechain interactome. Tab
structure (single tab with a regulation-overlay mode vs. a separate tab) is deferred to subplan-04 build
time — build the sidechain tab, add the overlay mode within it, split only if the UX is cramped.

**D3 — geneuse un-pin. RESOLVED: un-pin.** AD switches to the derived `DEG∪prG` recipe (same path as
t-cells); single cross-cohort recipe, wider enumeration (downstream SigProb/|PDS| filter still applies).
Removes the `use_frozen_geneuse` branch + `extract_sce4_geneuse.R` (anti-shim). Production-output-
changing — requires a full AD re-run + viewer rebuild (Wave 3). Touch points in
`ad_geneuse_unpin_from_sce4.md`.

**D4 — C4 form. RESOLVED: prose checklist / skill. CLEARED.** C4 stays a review-time operational
constraint (the human checks mouse-vs-human cell-type breadth per candidate before reporting), not an
automated guard — a veto would contradict the never-veto invariant in `specificity_confidence.md` §3.
Landed as `foundation/specificity_confidence.md` §3a; plan file removed.

**D5 — I2 timing.** `sidechain-01/02` import the exact frozen-layer modules `I2` relocates
(`*_decompose.py`, `song.py`, R extractors). Running `I2` mid-stream breaks those imports.
Recommendation: **do `I2` last, in an isolated window**, updating all importers in the same pass; or
skip until the sidechain work lands. Never interleave it with other file-touching plans.

---

## 3. Dependency graph

```
D3 ──► geneuse re-run ──► AD viewer rebuild ─────────────────────┐
                                                                 ├─► serialized
t-cell labels / Incytr re-run / t-cell viewer ── done, no gate    │   viewer
   └─► sidechain-02 ∥ A3 ── ungated, runnable now                 │   integration
                                                                 │
sidechain-01 (PSP backend) ──┬─► sidechain-03 ──► sidechain-04 ──┤
sidechain-02 ────────────────┘                                   │
                                                                 │
D2 ──► E1 (reuse sidechain-01 backend) ──► E2 ───────────────────┤
T2 specificity ──► E2                                            │
B2 (sankey), C3 (early-change) ─────────────────────────────────┘

tmt (H1), G1 (diagrams), apriori (docs), deploy-B  ── independent leaves, no gate
B1 (covered), A4 (verified), corr (built), tcell-incytr reports (built), C4 (cleared), G2 (retracted) ── done
I1 ── independent (low conflict) · I2 ── isolated window, last (D5)
```

---

## 4. Conflicts

- **E1 ↔ sidechain** — feature/backend overlap (T1). Hard conflict until D2 resolves; do not build E1
  before sidechain-01's edge model exists and the shared-backend decision is made.
- **I2 ↔ everything file-touching** — relocates modules that sidechain and the R pipeline import (D5).
  Isolate.
- **Viewer payload write contention** — sidechain-03/04, B2, C3, E1/E2, geneuse rebuild all write the
  payload builders. Not a logic conflict; serialize the *integration/rebuild* step (or use worktree
  isolation for authoring, single-threaded merge).
- **geneuse re-run ↔ AD-inclusive payloads** — the AD Incytr outputs change; any payload built during
  the re-run window is stale. Sequence the AD rebuild after the re-run completes.

---

## 5. Parallel execution waves

**Wave 0 — DONE / resolved.**
- `B1` IncytrDB — covered by the existing `incytrdb_audit/` (version-pinned, species-correct; rebuild
  blocked on unobtainable source artifacts, which the no-contact rule forbids requesting).
- `A4` T-cell data structure — verified on disk (donor1 has IMAC, donor2 does not; both total+pY+scRNA).
- `labels` labeling standard — per-cell marker assignment (D1/T4); one producer, one validator, no
  competing path. Labels verified current; nothing outstanding.
- t-cell Incytr **re-key + pair-mode re-run** — DONE (donor1 ×3, donor2 ×4 contrasts in
  `incytr_pair_mode_tcells`), and the **t-cell viewer** reads that root as its asserted
  production default (`build_tcell_viewer.py`).
- `corr` proteome↔transcriptome correlation — BUILT 2026-07-13 (rho=0.546, n=7693); archived to `standalone_done/`.
- `C4` cross-species specificity guard — CLEARED; documented as review constraint in `foundation/specificity_confidence.md` §3a (D4: prose, not code).
- `G2` positive controls — RETRACTED; curated per-cohort control list dropped, `check-controls` skill rewritten as a general in-the-moment biological-plausibility check (expectations from external biology, cited where possible).
- t-cell Incytr **report suite** — BUILT (cell-cell, a-priori/novelty, anchor-landing, protein-RNA
  timecourse) on the current labels; plan files removed from `docs/plans/` on ship.
- All adjudications resolved: D2 (shared backend), D3 (un-pin), D4 (prose checklist), D5 (I2 last).

**Wave 1 — independent builds (parallel, disjoint outputs, no gate).**
- `H1` TMT paper IMAC fetch + enrichment.
- `G1` workflow diagrams (docs).
- `deploy` Option B (shard-hash manifest) — infra layer, separate from analysis.
- `I1` sidecar cleanup — low file-conflict, any time.
- `apriori` blind exhaustion prediction set (docs) — literature-grounded, authored before final
  outputs; feeds the `/check-controls` plausibility skill and the `A3` validation lens. Blindness caveat: the t-cell
  Incytr report suite is already built, so ground the predictions in literature only, not those outputs.
- `A3` t-cell Incytr trend coverage — verifies the completed t-cell Incytr run (donor1 ×3, donor2 ×4);
  ungated, runnable now.

**Wave 2 — kinase backends + specificity thread.**
- `sidechain-01` (PSP interactome) ∥ `sidechain-02` (t-cell motif — ungated).
- `E1` on the shared `kinase_kinase_edges.py` backend (D2), then `E2` (needs T2 specificity).

**Wave 3 — Incytr recompute.**
- `geneuse` un-pin (D3) → AD re-run → AD viewer rebuild.

**Wave 4 — viewer tabs (serialize the payload/integration step).**
- `sidechain-03 → 04`, `B2` sankey, `C3` early-change tab, `E1/E2` tab(s). Author in parallel; merge
  and rebuild one at a time.

**Isolated — `I2`** frozen-layer namespace moves, dedicated window, last (D5).

All adjudications (D1–D5) are resolved, and the T-cell labeling gate is cleared (labels verified current
2026-07-16). No T-cell work is blocked on labeling.
