# 07 — Substrate multiplicity as the per-edge weight axis

**Unit:** recover per-gene substrate count in the bridge and carry it through to
the viewer, so a kinase's terminal fan is weighted by *which node it hits
hardest* — not flat. NES stays the kinase-strength axis; substrate multiplicity
becomes the edge-strength axis.

**Risk:** medium. The bridge count change and the `arg_max` alignment are silent
spots — a wrong count renders plausibly. Schema tuple guards catch the plumbing;
the count semantics must be verified against `kinase_node_hits.parquet`.

**Depends on:** 01 (backend edge model), 03 (payload slice), 04 (viewer sub-tab),
05 (NES weighting + emphasis), 06 (signed NES + arc/legend + edge-click) — all
shipped. 07 extends 06's terminal schema and emphasis math; it does not change
the arc/wedge geometry, the legend framework, or the edge-click framework — it
adds one field to each.

---

## Why (intent)

**NES is a kinase-level score, not an edge property.** `mea_timecourse.csv` is
kinase-library GSEA output: one row per `(kinase, contrast, track)` carrying a
single `NES`/`FDR`/`Leading substrates`. `build_substrate_bridge`
(`kinase_incytr_bridge.py:153-180`) explodes the leading-substrate motif string,
maps each 13-char motif back to gene_symbols, and **stamps that same kinase-level
NES onto every gene the kinase's leading substrates hit.**

Consequence in the current view: for kinase K in contrast C, every K→gene
terminal edge carries the *identical* NES, so K's whole fan renders at one width
and opacity. The weighting ranks kinases against each other; it says nothing
about which node K most strongly targets.

**The per-edge discriminator already exists and is discarded.** The count of K's
leading substrates that map to a given gene is the natural edge-strength signal —
a gene with 3 matched sites is a stronger K target than one with 1. The bridge
computes exactly this in the `seen_genes` loop and then throws it away by
de-duplicating to one row per `(kinase, contrast, gene)`
(`kinase_incytr_bridge.py:164-172`). Recovering it is a counter instead of a set.

**Out of scope — motif similarity.** The gene match is *exact 13-char string
equality* (`motif_to_genes.get(key)`), not a graded PSSM match. No per-site
similarity/percentile column exists in `mea_timecourse.csv` (`Subs fraction` is
kinase-level). Graded similarity would require re-plumbing kinase-library's
per-site scoring through the entire bridge — a large lift, and largely redundant
with leading-edge membership (the leading edge already *is* the best-matching,
enrichment-driving site set). **Do not add it.** This plan uses count only.

---

## Changes

### A. Bridge — recover the count · `alz/cross_reference/kinase_incytr_bridge.py`

1. **`build_substrate_bridge`** (~L146-183): replace the `seen_genes` set with a
   per-gene counter, scoped to the current mea row `(kinase, contrast, channel)`.
   Emit one row per `(kinase, contrast, channel, gene_symbol)` as today, but add
   `n_sites` = number of *distinct leading-substrate motifs* mapping to that gene
   for that row. Add `"n_sites"` to the returned column list and to the empty-frame
   column list.
2. **`FINAL_COLS`** (~L1157): append `"n_sites"`.
3. **Per-cohort hit assembly must carry `n_sites`:**
   - **tcells** (`write_tcells_streamed`, SQL ~L1328-1352): add `s.n_sites` to the
     `hits` select and `h.n_sites` to `annotated`.
   - **song / 5xFAD** (~L887-995): `n_sites` rides through `gene_node_hits`'s merge
     automatically; confirm it reaches the FINAL_COLS select at ~L1093 and isn't
     dropped by an explicit column list.

`n_sites` is a property of `(kinase, contrast, channel, gene)` — constant across
the node-index fan (role/sender/receiver), so the join never distorts it.

### B. Backend edges — aggregate + carry · `alz/cross_reference/kinase_kinase_edges.py`

1. **`load_motif_edges`** DuckDB agg (~L175-189): add
   ```sql
   arg_max(n_sites, ABS(NES)) AS n_sites,
   ```
   — the count from the *same row* that supplied `best_abs_nes` / `signed_nes`, so
   the NES/sign/count trio all describe one enrichment result. Add `"n_sites"` to
   the returned column list (~L202-206).
2. **`build_terminal_map`** (~L318-322): add `"n_sites"` to the output `cols`.
   Weight math unchanged here — `weight_motif` stays `norm(best_abs_nes)`; the
   count enters emphasis in the viewer, not the backend fusion weight. (Terminal
   edges only. `build_interactome` / chain edges untouched — multiplicity is a
   terminal-edge property.)
3. **MANIFEST text** (~L407): note `n_sites` on terminal edges (per-edge substrate
   multiplicity; combines with |NES| for edge emphasis).

Output: `terminal_edges.csv` appends `n_sites`. `interactome.csv` unchanged.

### C. Payload — extend terminal tuple · `alz/viewer/shared/payload_helpers.py`

- `_INCYTR_SIDECHAIN_TERMINAL_COLUMNS`: append `"n_sites"` (order must match the
  CSV column order from B). The exact-tuple guard forces backend regeneration
  before the shard rebuilds.

### D. Viewer — `alz/viewer_shared/template/js/tabs/incytr_sidechains.js`

The two visual axes become separable:

- **Node size** = |NES| emphasis (kinase strength) — **unchanged.** Which kinases
  matter.
- **Terminal-edge width & opacity** = |NES| emphasis **× substrate factor** — which
  node that kinase hits hardest. A kinase's fan is no longer flat.

1. Compute `sitesMax = max(n_sites)` over drawn terminal edges in `_isGraphForRow`
   (alongside `nesMax`), carry it on the graph object.
2. Terminal-edge emphasis (~L421-427): combine
   ```
   nesEmphasis = _isEmphasis(best_abs_nes, nesNull, nesMax)   // as today
   siteFactor  = siteFloor + (1 - siteFloor) * clamp01(log1p(n_sites) / log1p(sitesMax))
   emphasis    = nesEmphasis * siteFactor
   ```
   feeding the existing `_isEmphasisWidth` / `_isEmphasisOpacity`. `siteFloor`
   (≈0.4, new `_IS_EMPHASIS` constant) keeps a strong-kinase single-site edge
   visible while a multi-site edge of the same kinase is emphasized. Log-scale so
   one dominant high-count edge doesn't collapse the rest. Carry `n_sites` onto
   the terminal-edge element data.
   - Node size still reads `_isEmphasis(best_abs_nes, …)` directly — do **not** fold
     `siteFactor` into node size; the two axes must stay independent.
3. **Edge-click popup** (~L591-595): append `, sites <n_sites>` to the terminal-edge
   detail line.
4. **Legend**: change the width/opacity entry from "|NES| strength" to
   "|NES| × #substrates (edge strength)"; node-size entry stays "|NES|". Keep it
   built from constants so it can't drift.

---

## Regenerate + verify

```
pixi run python -m alz.cross_reference.kinase_incytr_bridge --cohort tcells   # rebuild kinase_node_hits with n_sites
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort tcells
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort song
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort fivexfad
pixi run python -m alz.build_tcell_viewer --html
pixi run viewer
```

(The bridge rebuild is required — `n_sites` originates there; regenerating only
`kinase_kinase_edges` against a stale parquet will fail on the missing column.)

### Acceptance checks

1. **Schema**: `terminal_edges.csv` has `n_sites`; the shard's `terminal_edges`
   columns include it (else C's tuple guard raises). `kinase_node_hits.parquet`
   carries `n_sites` for all three cohorts.
2. **Count is real & consistent**: in the regenerated t-cell shard, pick a kinase
   hitting ≥2 nodes in `d13_d2` — its `n_sites` **varies across its target nodes**
   (the fan is no longer flat). `n_sites` at a `(kinase, gene, role, contrast)`
   equals the distinct-leading-substrate-motif count for the max-|NES| track in
   `kinase_node_hits.parquet` (arg_max picks that row).
3. **Fan differentiates**: on the reference row `FURIN|ADAM19|TOP2A|SFSWAP`
   (tcells/donor1), a single kinase's terminal edges to different nodes render at
   different widths when their `n_sites` differ; a strong-kinase single-site edge
   is still visible (floored, not vanished).
4. **Axes independent**: two edges with equal `best_abs_nes` but different
   `n_sites` differ in width; node size tracks only |NES| (unchanged from 06).
5. **Click + legend**: tapping a terminal edge shows `sites <n>`; the legend's
   edge-strength entry reads "|NES| × #substrates" and node-size still reads
   "|NES|"; no legend entry references an undrawn encoding.
6. **Regression**: chain (kinase→kinase) edges unchanged (`interactome.csv` has no
   `n_sites`); song / 5xFAD shards regenerate without schema error; signed-NES hue,
   arc spine, wedges, and edge-click from 05/06 still render.
7. `tests/test_kinase_sidechain_weighting.py` updated: backend test asserts
   `n_sites` carried through `build_terminal_map` and that `load_motif_edges`
   reports the arg_max-aligned count; viewer test asserts two same-kinase terminal
   edges with different `n_sites` get different width, and that node size is
   unchanged by `n_sites`.

Served over HTTP, hard-refresh, expand `FURIN|ADAM19|TOP2A|SFSWAP` → Sidechains:
each kinase's fan now leans toward the node it hits with more substrates, node
size still reads kinase enrichment, click shows the site count.
