# Kinase Sidechain Incytr Graph

Architecture record for the sidechain view — a sub-tab in the Incytr pathways
detail panel that renders **one** Incytr pathway spine (Ligand → Receptor → EM →
Target) decorated with the kinases that regulate its nodes, plus the kinase→kinase
relationships among those regulators.

Shipped 2026-07-17. Decomposed into subplans 01–07 (archived under
`archive/archived_plans/kinase_sidechain_incytr_graph/`); this document is the
durable spec.

---

## Purpose

The Incytr pathways view ranks Ligand→Receptor→EM→Target signaling pathways. It
answers *which pathway*, not *which kinases drive it*. The sidechain view takes one
selected pathway and overlays:

- **Terminal edges** — kinase → pathway-node, where MEA substrate-set membership
  at least one physical phosphosite whose measured change is Significance-B
  significant and direction-concordant with an MEA-eligible kinase call. "Which
  kinase-attributed sites changed on this node's protein."
- **Interactome edges** — kinase → kinase, among the regulators drawn, from
  literature and cohort motif co-enrichment. "How those kinases regulate each
  other."

The result is a mechanistic sidechain around a single pathway backbone.

---

## Data flow

```
kinase_incytr_bridge.py                 →  kinase_kinase_edges.py            →  payload_helpers.py           →  incytr_sidechains.js
(per-cohort motif hits, streamed)          (fuse into two edge lists)           (per-cohort columnar shard)     (Cytoscape render)

kinase_node_hits.parquet  ──────────────▶  interactome.csv                  ──▶  interactome / terminal_edges ──▶  spine + fans
  (kinase, gene, role,                      terminal_edges.csv                    tuple-guarded shard
   contrast, NES, FDR,
   n_sites, sites, n_significant_concordant, …)
```

1. **Bridge** (`alz/cross_reference/kinase_incytr_bridge.py`) — per cohort, joins
   MEA (kinase-library motif enrichment) leading substrates to Incytr pathway
   nodes via the stoichiometry matrix, writing `kinase_node_hits.parquet` (one row
   per kinase × node-gene × contrast × track). **Multi-GB decompressed
   (9–23M rows)** — every downstream read is DuckDB-streamed under a memory cap,
   never `read_parquet` into pandas.

2. **Edge model** (`alz/cross_reference/kinase_kinase_edges.py`) — reduces the hit
   table into two provenance-tagged edge lists: `terminal_edges.csv`
   (kinase→node) and `interactome.csv` (kinase→kinase). One dir per cohort tissue
   under `outputs/reports/kinase_kinase_edges/`.

3. **Payload** (`alz/viewer/shared/payload_helpers.py`) — packs the two CSVs into a
   columnar per-cohort shard referenced by `edge_slice_ref`. An **exact-tuple
   schema guard** (`_INCYTR_SIDECHAIN_TERMINAL_COLUMNS` /
   `_INCYTR_SIDECHAIN_INTERACTOME_COLUMNS`) forces backend regeneration whenever
   the column set drifts.

4. **Viewer** (`alz/viewer_shared/template/js/tabs/incytr_sidechains.js`) — a
   Cytoscape.js `preset`-layout render, wired into `incytr_pathways.js` (no
   standalone tab — there is no path-selection event to drive one).

---

## The two edge classes

### Terminal edges (kinase → pathway-node)

**Source: measured phospho change, motif-attributed and MEA-gated.** In the
t-cell pilot, a terminal edge exists only when the kinase has MEA `FDR < 0.25`
for that contrast and channel and at least one MEA-member physical site on the
node gene is a Significance-B significant outlier whose Δ sign
matches signed NES. PSP literature only *corroborates* an existing edge (bumps
provenance to `both`); it never creates one.

The substrate basis is the exact MEA `kl_thresh` membership receipt (15 ST / 7
pY), for Song, 5xFAD, and t-cells alike. `kl_percentile` remains a per-site
annotation and is not a second gate. Per-edge fields carried to the viewer are
`signed_nes`, `best_abs_nes`, `best_fdr`, `n_sites`,
`n_significant_concordant`, `edge_delta`, `sites`, `provenance`, `role`, and
`contrast`. `sites` is per-physical-site JSON evidence
(`site_id`, `site_position`, `motif`, `residue_type`, `kl_percentile`, measured
`delta`, corrected `site_significance`, `changed` (the BH-significant flag),
`concordant`, and `timecourse_consistency`). `edge_delta` averages `delta` over
sites where `changed and concordant` — reusing the bridge's stored flag, so the
significance cutoff has one owner and is never re-thresholded downstream.

For t-cell direct-change edges, the sidechain pill is site-specific and
per-edge. `motif_peers_informative` is the number of candidates with a
concordant MEA claim on the edge's changed sites that are detected somewhere in
the cohort; `motif_peers_detected` is the subset detected in the edge's owning
cluster, and `motif_peer_roster` carries those candidates' owning-cluster
detection fractions. AD/Song and 5xFAD could-phosphorylate edges have no
per-edge direct-change fields and retain the global-cosine roster fallback.
Non-significant and discordant matched sites remain inside a surviving edge's
`sites` evidence; only the edge is gated.

Significance B is estimated per contrast and channel from intensity-binned,
asymmetric robust nulls on MEA-normalized stoichiometry Δ values. Its two-sided
tail probability is BH-corrected across sites. The five baseline timepoint
contrasts share D2 and are therefore corroborating states, not independent
replicates; `timecourse_consistency` counts every contrast a site is
significant-concordant in — including contrasts where the kinase is not
MEA-eligible — and is stored, never an edge gate (MEA eligibility gates only
which rows are emitted). This criterion is the `tcells_donor1` pilot; Song and
5xFAD remain unchanged this pass.

### Interactome edges (kinase → kinase)

**Source: two independent arms, unioned** in `build_interactome`:

- **PSP arm** — the full curated human→human kinase→kinase set from
  `Kinase_Substrate_Dataset_count_07_2021.txt` (shipped inside the
  `kinase_library` package; not vendored). A kinase→kinase edge is a PSP substrate
  row whose `SUB_GENE` is itself a kinase gene; autophosphorylation self-loops
  dropped; `IN_VIVO/IN_VITRO_REF_COUNT` summed per pair (carried for reference).
- **Motif arm** — motif edges whose *target* gene is itself a kinase
  (contrast-collapsed; `n_motif_contrasts` / `motif_contrasts` carried for reference).

Outer-joined on (source, target); provenance = `both` / `psp` / `motif`.
Chain edges carry **no fused strength scalar** — provenance is the only signal, and
it drives edge color only. Contrast-agnostic — per-contrast detail stays in the
terminal edges.

Mouse cohorts (song, 5xFAD): PSP is human, homology-mapped to mouse before fusion.
T-cells are human — no mapping.

---

## Edge-strength model — what drives visual emphasis

This is the correctness-critical part and the least obvious.

- **NES is a kinase-level score, not an edge property.** `mea_timecourse.csv` is
  kinase-library GSEA output: one NES per (kinase, contrast, track). The bridge
  explodes the leading-substrate string and stamps that **same** NES onto every
  gene the kinase's leading substrates hit. So NES ranks *kinases against each
  other*, and by itself a kinase's terminal fan is flat — every K→node edge from
  one kinase-contrast carries identical NES.

- **Physical-site evidence is the per-edge discriminator.** `n_sites` is the
  count of distinct physical `site_id` values mapped to the node gene. The
  `n_significant_concordant` count gates t-cell edge existence, while `edge_delta`
  is the mean measured Δ across those significant-concordant sites. The
  `load_motif_edges` aligns these fields and `signed_nes` to the *same*
  max-|NES| row via a `ROW_NUMBER()` window (deterministic tie-break: `ABS(NES)
  DESC, FDR ASC, channel ASC`), so `abs(signed_nes) == best_abs_nes` always holds.

- **Motif similarity is deliberately NOT used.** The motif→gene match is exact
  13-char string equality, not a graded PSSM score, and no per-site similarity
  column exists in the artifact. Leading-edge membership already encodes match
  quality categorically. Adding graded similarity would mean re-plumbing
  kinase-library per-site scoring — out of scope, do not add.

- **Signed NES** carries direction (+ enriched / − depleted); its magnitude is
  used for kinase node emphasis, while terminal edge strength comes from
  measured `edge_delta`.

- **PDS** (`best_abs_pds`) is carried for reference but does not drive terminal
  strength — terminal strength is `|edge_delta|` (see viewer encodings below);
  there is no fused weight scalar.

---

## Viewer encodings

Two visual axes, kept independent:

| Element | Encodes | Formula |
|---|---|---|
| Kinase **node size** | kinase strength (which kinases matter) | `_isEmphasis(best_abs_nes)` — |NES| only |
| Terminal **edge width + opacity** | measured terminal change | `_isEmphasis(|edge_delta|, 0, 4)`; 4 is the pilot absolute-Δ 95th-percentile anchor |
| Terminal **edge hue** | direction | signed NES > 0 → enriched red `#d73027`; < 0 → depleted blue `#4575b4` |
| Terminal **edge style** | edge class | dotted (distinguishes from chain edges) |
| Chain **edge color** | provenance | motif blue / PSP orange dashed / both purple |
| Chain **edge width** | none (uniform) | fixed `chainEdgeWidthPx`; chain edges carry no strength scalar |
| Spine **edge** | core pathway | bold `#1f4ea3`, always strongest |

The emphasis transform `_isEmphasis(value, lo, hi) = clamp01((value−lo)/(hi−lo))^γ`
uses γ = 3.5. Node emphasis remains anchored at the biological null `|NES| = 1`;
terminal edge emphasis is anchored at zero measured Δ and the pilot's 4-unit
absolute-Δ reference. Node/edge floors keep weak evidence faint-but-present,
never invisible.

**Layout** — the 4 spine nodes sit on a shallow arc; each node's kinase fan opens
into its own outward angular wedge (concentric arcs, strong kinases inner / weak
outer), so adjacent nodes' fans tile instead of competing for one axis.

**Interaction** — tap a node to isolate its closed neighborhood (∪ spine) and
zoom, *and* fill the relationship table (below); tap any edge for a detail panel
(terminal: signed NES, direction, FDR, measured Δ, significant/concordant count,
and the sorted per-site evidence table; chain: provenance); tap background to clear. The graph and
the dedicated right-side evidence panel remain side-by-side, including in browser
full-screen mode. A static legend built from the style constants explains every
encoding.

**Node relationship table** (`_isNodeRelationTable`) — a node tap also renders a
ranked table of that node's relationships into the dedicated side evidence panel,
so the isolated neighborhood is readable as rows rather than only as geometry.
Direction-aware:

- **Spine node** → rows are the kinases hitting it. Summary: `N kinases affecting
  · E enriched · D depleted`.
- **Kinase node** → rows are what it points at — pathway nodes via terminal edges
  (with role and signed NES), then other kinases via chain edges (provenance).
  Summary: `targets N nodes · M kinases`.

Terminal rows sort by |signed NES| desc and always precede chain rows, which sort
by relationship label. The table includes measured edge Δ and the significant/
concordant count; the adjacent site table includes site Δ, corrected significance,
concordance, and timecourse consistency. It reads the loaded graph regardless of the `showChains`
graph filter — it is a text read-out, not a second view of the canvas. Purely
viewer-side: every field already rides the Cytoscape edge data, so no backend,
payload, or schema-guard column participates.

---

## Cohorts

`COHORT_DIRS` in `kinase_kinase_edges.py`:

- `song` → `song`
- `fivexfad` → `fivexfad_cortex`, `fivexfad_hippocampus`
- `tcells` → `tcells_donor1` (donor1 only — donor2 has no within-cohort
  attribution, so the bridge emits no donor2 motif source)

---

## Regenerate

```
pixi run python -m alz.cross_reference.kinase_incytr_bridge --cohort <song|fivexfad|tcells>
pixi run python -m alz.cross_reference.kinase_kinase_edges  --cohort <song|fivexfad|tcells>
pixi run python -m alz.build_tcell_viewer --html    # t-cell shard + viewer
pixi run viewer                                     # AD/Song unified viewer
bash alz/runners/main/run_kinase_sidechain_reconciliation.sh  # all four cohorts + contract check
```

The bridge must run first when the hit-table schema changes (`n_sites` originates
there); regenerating only `kinase_kinase_edges` against a stale parquet fails on
the missing column. After `pixi run viewer`, hard-refresh — the payload is inlined
into `index.html`.

Verification: `tests/test_kinase_sidechain_weighting.py` (backend aggregation +
viewer emphasis/layout math, Node-executed against the real JS source).

---

## Invariants

- **Streamed reads only.** `kinase_node_hits.parquet` is multi-GB; all reads are
  DuckDB with `memory_limit` set. Never whole-read it into pandas.
- **`abs(signed_nes) == best_abs_nes`** for every terminal edge (same-row pick).
- **Terminal edges are measured-change-created, motif-attributed, MEA-gated, and
  PSP-corroborated.** Interactome edges can be literature-only. Do not let PSP
  create terminal edges.
- **Terminal site evidence reconciles.** For every regenerated terminal edge,
  `len(sites) == n_sites`; each site carries its physical ID and position, motif,
  residue type, raw `kl_percentile`, and direct-change evidence. A mismatch
  fails the reducer/viewer contract check.
- **Schema tuple guard is the change-detector.** Any new backend column must be
  appended to the payload column tuples in the same pass, or the shard build
  raises.
- **`n_sites` lives only on terminal edges.** `interactome.csv` has no `n_sites` —
  multiplicity is a kinase→node property.
