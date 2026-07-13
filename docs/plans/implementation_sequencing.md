# Plans — implementation sequencing & adjudication

Goal: implement every plan in `docs/plans/`. This maps the cross-cutting threads, the blocking
decisions, the conflicts, and a parallel execution order. Companion to `plans_state_audit_2026-07-10.md`.

Plan shorthand used below:
`sidechain` = `kinase_sidechain_incytr_graph/` (01–04) · `geneuse` = `ad_geneuse_unpin_from_sce4`
`corr` = `tcell-proteome-transcriptome-correlation` · `matt` = `tcell-matt-report-restoration` (active WIP, user-owned)
`deploy` = `deployment/todo9_viewer_aws_deployment` · A3/A4/B1/B2/C3/C4/E1/E2/G1/G2/H1/I1/I2 = backlog files.

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

**T2 — cell-type specificity engine.** `C4`, `E2`, and `G2` all consume the existing per-kinase
cell-type specificity basis. `C4` gates targets on human-vs-mouse breadth; `E2` uses specificity to
split same-family kinases; `G2` validates the specificity metric against known controls (PHKG1
astrocyte, ATP9A endothelial). Same data source, three lenses — build the read path once, share it.

**T3 — Incytr recompute + viewer rebuild.** `geneuse` (un-pin → AD re-run), `sidechain` (consumes
Incytr `wide/` output), `B2`, `C3`, `E1/E2` tabs, and the t-cell relabel all end in a viewer payload
rebuild. `B1` (IncytrDB version audit) is **covered** — the six DB objects are version-pinned
(2026-03-09, MD5s in `current_manifest.json`) and species selection is per-dataset correct, so
recomputes are not DB-blocked. Payload writers (`build_unified_viewer.py`, `build_tcell_viewer.py`,
cohort `*.py`, `slices_incytr.py`) cannot be written concurrently — serialize integration, not authoring.

**T4 — T-cell labeling ontology (RESOLVED).** Canonical = **Matt cluster-relabel** (Seurat clusters →
a small biological label set via `tcell_state_labels.py` `MATT_RELABEL_TABLE`). ProjecTILs is
**permanently retired** (never canonical again; kept as evidence columns only); the marker-panel
per-cell labeler (`a7d53d2`) is abandoned. The labeler is an **active WIP owned by the user** — the
t-cell Incytr input re-key + pair-mode re-run (donor1 3 contrasts, donor2 4), viewer rebuild, report
regen, `sidechain-02`, and `A3` all wait on those labels **freezing**, not on any further decision.
**`corr` is the one t-cell plan independent of labeling** (raw-Seurat pseudobulk, no state idents) — it
was built independently of the labeling gate (**DONE 2026-07-13**).

---

## 2. Adjudications required (blocking — resolve before the dependent wave)

**D1 — T-cell labeling ontology. RESOLVED** (see T4): canonical = Matt cluster-relabel; ProjecTILs
permanently retired; marker-panel per-cell abandoned. No longer an open adjudication — the only
remaining gate is the user's WIP labels freezing before the t-cell Incytr re-key/re-run.

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

**D4 — C4 form. RESOLVED: prose checklist / skill.** C4 stays a review-time operational constraint
(the human checks mouse-vs-human cell-type breadth per candidate before reporting), not an automated
guard. No code artifact; no dependency on a candidate-target-list surface. Matches the current
`cross-species-specificity-guard.md` framing — no rewrite needed.

**D5 — I2 timing.** `sidechain-01/02` import the exact frozen-layer modules `I2` relocates
(`*_decompose.py`, `song.py`, R extractors). Running `I2` mid-stream breaks those imports.
Recommendation: **do `I2` last, in an isolated window**, updating all importers in the same pass; or
skip until the sidechain work lands. Never interleave it with other file-touching plans.

---

## 3. Dependency graph

```
D3 ──► geneuse re-run ──► AD viewer rebuild ─────────────────────┐
                                                                 ├─► serialized
t-cell labels FREEZE (Matt cluster-relabel WIP)                  │   viewer
   └─► Incytr re-key + re-run ──► t-cell viewer ──┬─► sidechain-02│   integration
                                                  └─► A3          │
                                                                 │
sidechain-01 (PSP backend) ──┬─► sidechain-03 ──► sidechain-04 ──┤
sidechain-02 ────────────────┘                                   │
                                                                 │
D2 ──► E1 (reuse sidechain-01 backend) ──► E2 ───────────────────┤
T2 specificity ──► C4, G2, E2                                    │
B2 (sankey), C3 (early-change) ─────────────────────────────────┘

tmt (H1), G1 (diagrams), deploy-B  ── independent leaves, no gate
B1 (covered), A4 (verified), corr (built) ── done
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
- `matt` labeling — RESOLVED to Matt cluster-relabel (D1/T4); the labeler is user-owned active WIP.
- `corr` proteome↔transcriptome correlation — BUILT 2026-07-13 (rho=0.546, n=7693); archived to `standalone_done/`.
- All adjudications resolved: D2 (shared backend), D3 (un-pin), D4 (prose checklist), D5 (I2 last).

**Wave 1 — independent builds (parallel, disjoint outputs, no gate).**
- `H1` TMT paper IMAC fetch + enrichment.
- `G1` workflow diagrams (docs).
- `deploy` Option B (shard-hash manifest) — infra layer, separate from analysis.
- `I1` sidecar cleanup — low file-conflict, any time.
- `A3` t-cell Incytr trend coverage — deferred: it verifies the t-cell Incytr re-run, which waits on the
  labels freezing. Not Wave 1.

**Wave 2 — kinase backends + specificity thread.**
- `sidechain-01` (PSP interactome) ∥ `sidechain-02` (t-cell motif — waits on labels freezing).
- `E1` on the shared `kinase_kinase_edges.py` backend (D2), then `E2` (needs T2 specificity).
- `G2` (T2 specificity consumer) — parallel with the above, disjoint outputs.
- `C4` — prose checklist (D4); a docs task, no backend dependency, can land any time.

**Wave 3 — Incytr recompute.**
- `geneuse` un-pin (D3) → AD re-run → AD viewer rebuild.

**Wave 4 — viewer tabs (serialize the payload/integration step).**
- `sidechain-03 → 04`, `B2` sankey, `C3` early-change tab, `E1/E2` tab(s). Author in parallel; merge
  and rebuild one at a time.

**Isolated — `I2`** frozen-layer namespace moves, dedicated window, last (D5).

All adjudications (D1–D5) are resolved. The only remaining gate is the user's T-cell labeling WIP
freezing before the t-cell Incytr re-key/re-run.
