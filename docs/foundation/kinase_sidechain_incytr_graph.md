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

- **Terminal edges** — kinase → pathway-node, where a kinase's motif is enriched
  among the phosphosites of a node's gene. "Which kinases phosphorylate this
  node's protein."
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
   n_sites, …)
```

1. **Bridge** (`alz/cross_reference/kinase_incytr_bridge.py`) — per cohort, joins
   MEA (kinase-library motif enrichment) leading substrates to Incytr pathway
   nodes via the stoichiometry matrix, writing `kinase_node_hits.parquet` (one row
   per kinase × node-gene × contrast × track). **Multi-GB decompressed
   (9–23M rows)** — every downstream read is DuckDB-streamed under a memory cap,
   never `read_parquet` into pandas.

2. **Edge model** (`alz/cross_reference/kinase_kinase_edges.py`) — reduces the hit
   table into two provenance-tagged, weighted edge lists: `terminal_edges.csv`
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

**Source: cohort motif enrichment only.** A terminal edge exists because the
kinase's motif is enriched among the phosphosites of the node's gene in this
cohort. PhosphoSitePlus (PSP) literature only *corroborates* an existing edge
(bumps provenance to `both`, adds `weight_lit`); it never creates one.

Per-edge fields carried to the viewer: `signed_nes`, `best_abs_nes`, `best_fdr`,
`n_sites`, `weight_motif`, `provenance`, `role`, `contrast`.

### Interactome edges (kinase → kinase)

**Source: two independent arms, unioned** in `build_interactome`:

- **PSP arm** — the full curated human→human kinase→kinase set from
  `Kinase_Substrate_Dataset_count_07_2021.txt` (shipped inside the
  `kinase_library` package; not vendored). A kinase→kinase edge is a PSP substrate
  row whose `SUB_GENE` is itself a kinase gene; autophosphorylation self-loops
  dropped; `IN_VIVO/IN_VITRO_REF_COUNT` summed per pair. Weight
  `weight_lit = norm(log1p(in_vivo_refs))`.
- **Motif arm** — motif edges whose *target* gene is itself a kinase
  (contrast-collapsed to `max |NES|`). Weight `weight_motif = norm(max |NES|)`.

Outer-joined on (source, target); provenance = `both` / `psp` / `motif`; final
`weight = weight_lit + weight_motif` (range [0, 2]). Contrast-agnostic —
per-contrast detail stays in the terminal edges.

Mouse cohorts (song, 5xFAD): PSP is human, homology-mapped to mouse before fusion.
T-cells are human — no mapping.

---

## Edge-weighting model — what drives strength

This is the correctness-critical part and the least obvious.

- **NES is a kinase-level score, not an edge property.** `mea_timecourse.csv` is
  kinase-library GSEA output: one NES per (kinase, contrast, track). The bridge
  explodes the leading-substrate string and stamps that **same** NES onto every
  gene the kinase's leading substrates hit. So NES ranks *kinases against each
  other*, and by itself a kinase's terminal fan is flat — every K→node edge from
  one kinase-contrast carries identical NES.

- **Substrate multiplicity (`n_sites`) is the per-edge discriminator.** It is the
  count of a kinase's distinct leading-substrate motifs that map to a given node's
  gene. A node hit by 3 of a kinase's substrates is a stronger target than one hit
  by 1. This count was originally computed then discarded by a dedup; subplan 07
  recovers it. `load_motif_edges` aligns `n_sites` and `signed_nes` to the *same*
  max-|NES| row via a `ROW_NUMBER()` window (deterministic tie-break: `ABS(NES)
  DESC, FDR ASC, channel ASC`), so `abs(signed_nes) == best_abs_nes` always holds.

- **Motif similarity is deliberately NOT used.** The motif→gene match is exact
  13-char string equality, not a graded PSSM score, and no per-site similarity
  column exists in the artifact. Leading-edge membership already encodes match
  quality categorically. Adding graded similarity would mean re-plumbing
  kinase-library per-site scoring — out of scope, do not add.

- **Signed NES** carries direction (+ enriched / − depleted); magnitude drives
  strength, sign drives hue only.

- **PDS** (`best_abs_pds`) is carried for reference but does not drive terminal
  strength — the motif weight is `norm(|NES|)`.

---

## Viewer encodings

Two visual axes, kept independent:

| Element | Encodes | Formula |
|---|---|---|
| Kinase **node size** | kinase strength (which kinases matter) | `_isEmphasis(best_abs_nes)` — |NES| only |
| Terminal **edge width + opacity** | edge strength (which node a kinase hits hardest) | `nesEmphasis × siteFactor`, `siteFactor = siteFloor + (1−siteFloor)·log1p(n_sites)/log1p(sitesMax)`, `siteFloor = 0.4` |
| Terminal **edge hue** | direction | signed NES > 0 → enriched red `#d73027`; < 0 → depleted blue `#4575b4` |
| Terminal **edge style** | edge class | dotted (distinguishes from chain edges) |
| Chain **edge color** | provenance | motif blue / PSP orange dashed / both purple |
| Spine **edge** | core pathway | bold `#1f4ea3`, always strongest |

The emphasis transform `_isEmphasis(value, lo, hi) = clamp01((value−lo)/(hi−lo))^γ`
is anchored at the biological null NES = 1 (`nesNull`), γ = 3.5, so strong edges
separate aggressively from weak ones. Node/edge floors keep null-NES elements
faint-but-present, never invisible.

**Layout** — the 4 spine nodes sit on a shallow arc; each node's kinase fan opens
into its own outward angular wedge (concentric arcs, strong kinases inner / weak
outer), so adjacent nodes' fans tile instead of competing for one axis.

**Interaction** — tap a node to isolate its closed neighborhood (∪ spine) and
zoom; tap any edge for a detail popup (terminal: signed NES, direction, FDR,
|NES|, sites; chain: provenance, weight); tap background to clear. A static legend
built from the style constants explains every encoding.

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
- **Terminal edges are motif-created, PSP-corroborated.** Interactome edges can be
  literature-only. Do not let PSP create terminal edges.
- **Schema tuple guard is the change-detector.** Any new backend column must be
  appended to the payload column tuples in the same pass, or the shard build
  raises.
- **`n_sites` lives only on terminal edges.** `interactome.csv` has no `n_sites` —
  multiplicity is a kinase→node property.
