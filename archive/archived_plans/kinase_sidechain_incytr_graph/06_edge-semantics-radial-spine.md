# 06 — Edge semantics, signed NES, and a radial spine

**Unit:** three usability changes to the sidechain view — (1) a static legend
explaining every encoding, (2) signed (non-absolute) NES surfaced as edge hue +
on click, and (3) an arc/per-node-wedge spine layout replacing the linear one.

**Risk:** medium. Backend/payload changes fail loudly (schema tuple check). The
signed-NES aggregation (`arg_max`) and the wedge geometry are the silent spots —
a wrong sign or a mis-tiled wedge renders plausibly. Verify against the numbers
and the reference row.

**Depends on:** 01 (backend edge model), 03 (payload slice), 04 (viewer sub-tab),
05 (NES weighting + emphasis + radial fan + hide-on-select) — all shipped. This
extends their outputs; C1 replaces 05's fan geometry (one layout, not two).

---

## Why (intent)

Three gaps on the current view (`FURIN|ADAM19|TOP2A|SFSWAP`, tcells/donor1):

1. **Encodings are unexplained.** Edge color (chain motif/PSP/both), dash style,
   width, opacity, and node size all carry meaning the user can't decode.
2. **NES sign is invisible and unclickable.** The shard carries only `best_abs_nes`
   (magnitude). Whether a kinase is *enriched* (+NES) or *depleted* (−NES) on a
   node — the biological direction — is discarded, and there's no way to read the
   exact value off an edge.
3. **The spine is linear.** All four nodes sit on one horizontal axis, so every
   node's kinase fan competes for the same up/down space. Giving each node its own
   angular wedge removes that competition.

The sign **is recoverable**: `kinase_node_hits.parquet` has signed `NES` per
`(kinase, gene_symbol, role, contrast)`; the backend currently reduces it with
`MAX(ABS(NES))`. The signed value at the max-magnitude row is `arg_max(NES,
ABS(NES))` — one added aggregate, no new source.

---

## Changes

### A. Backend — carry signed NES · `alz/cross_reference/kinase_kinase_edges.py`

Signed NES is a **terminal-edge** property (kinase→spine-node motif enrichment).
Chain edges (kinase→kinase) carry a combined `weight`, not a signed NES — leave
the interactome untouched.

1. **`load_motif_edges`** DuckDB agg (~L175-188): add
   ```sql
   arg_max(NES, ABS(NES)) AS signed_nes,
   ```
   next to `MAX(ABS(NES)) AS best_abs_nes`. (`signed_nes` = the signed NES of the
   row whose magnitude `best_abs_nes` reports — same row, keeps them consistent.)
   Add `signed_nes` to the returned column list (~L201-206).
2. **`build_terminal_map`** (~L299-318): add `signed_nes` to the output `cols`
   list. No weight change — `weight_motif` stays `norm(best_abs_nes)`; sign never
   enters strength, only direction.
3. **MANIFEST text** (~L405): note `signed_nes` on terminal edges (direction:
   +enriched / −depleted; magnitude drives weight).

Output: `terminal_edges.csv` appends `signed_nes`. `interactome.csv` unchanged.

### B. Payload — extend terminal tuple · `alz/viewer/shared/payload_helpers.py`

- `_INCYTR_SIDECHAIN_TERMINAL_COLUMNS`: append `"signed_nes"` (order must match the
  CSV column order from A). The exact-tuple guard forces backend regeneration
  before the shard will build.

### C. Viewer — `alz/viewer_shared/template/js/tabs/incytr_sidechains.js`

#### C1. Arc spine + per-node wedge fans (replaces 05's horizontal spine + up/down fans)

- Place the 4 spine nodes on a shallow **arc** instead of a horizontal line:
  a circular arc of radius `spineArcRadiusPx` centered at `(cx, arcCenterY)` below
  the graph, the nodes spanning an angular range (e.g. `spineSweepDeg` ≈ 90°,
  Ligand→Target ordered along the arc). Spine edges follow the arc (bold, as now).
- Each spine node's **outward direction** = the radial from the arc center through
  that node. Its kinase fan (05's concentric arcs, strong inner → weak outer) is
  re-centered on the node and rotated to open into that node's **wedge**:
  `±halfWedge` around the outward direction, `halfWedge ≈ spineSweepDeg /
  (2·segments)` so adjacent wedges tile without overlapping. This repurposes 05's
  `fanInnerRadiusPx / fanRingStepPx / fanArcSpacingPx` ring-packing — only the
  center and the base angle per group change (from "up/down around a horizontal
  node" to "outward along the node's radial"). A kinase targeting multiple spine
  nodes anchors to the mean of its targets' wedges, as today.
- Delete the `direction = groupIndex % 2` up/down alternation — the wedge angle
  now comes from the node's arc position, not parity.

#### C2. Signed-NES hue on terminal edges + emphasized core path

- Terminal edges keep `line-style: dotted` (this is what distinguishes them from
  chain edges — style, not hue), width & opacity still `∝ |NES|` emphasis (05).
- Color now encodes **direction** via a colorblind-safe diverging pair:
  `signed_nes > 0` → enriched hue (e.g. RdBu red `#d73027`), `< 0` → depleted hue
  (e.g. RdBu blue `#4575b4`). Add `sign` (or the raw `signed_nes`) to terminal-edge
  element data and split the terminal-edge style selector on it.
  - Chain-edge provenance colors (motif blue / PSP orange / both purple) are
    unaffected — different edge class, kept dotted/dashed by provenance. The
    dotted terminal style keeps the two classes distinct despite blue appearing in
    both. (Flag for grilling: if red/blue terminal + blue/orange/purple chain still
    reads muddy, consider a non-blue depleted hue.)
- Core path: spine edges stay bold `#1f4ea3` — already the strongest visual
  element; keep. This satisfies "different edges emphasize the core path from the
  supporting kinases."

#### C3. Static legend + click-any-edge popup

- **Static legend panel** under the graph (a DOM block next to the existing
  caption, not a Cytoscape element): one line per encoding —
  bold = core pathway · red →enriched / blue →depleted (kinase→node, dotted) ·
  width & opacity = |NES| strength · node size = |NES| · chain edges:
  motif (blue) / PSP (orange, dashed) / both (purple). Build from constants so it
  can't drift from the styles.
- **Click any edge → detail popup** (a small absolutely-positioned DOM box or an
  updating detail line under the caption — no new dependency; do NOT add a popper
  lib): `cy.on("tap", "edge", …)`.
  - terminal edge: `SOURCE → TARGET`, `NES <signed>` + `(enriched|depleted)`,
    `FDR <best_fdr>`, `|NES| <best_abs_nes>`.
  - chain edge: `SOURCE → TARGET`, `provenance`, `weight`.
  - This coexists with 05's tap-node-to-isolate (`node` vs `edge` target); a
    background tap clears both the isolate and the popup.

---

## Regenerate + verify

```
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort tcells
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort song
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort fivexfad
pixi run python -m alz.build_tcell_viewer --html
pixi run viewer
```

### Acceptance checks

1. **Schema**: `terminal_edges.csv` has `signed_nes`; the shard's `terminal_edges`
   columns include it (else B's tuple guard raises).
2. **Sign is real & consistent**: in the regenerated t-cell shard, `d13_d2`,
   `ADAM19`, `Receptor` — `sign(signed_nes)` varies across kinases, and
   `abs(signed_nes) == best_abs_nes` row-for-row (arg_max picks the same row MAX
   reports). Spot-check CDK2 direction against `kinase_node_hits.parquet`.
3. **Legend**: every encoding used in the render appears in the static legend; no
   legend entry references an encoding that isn't drawn.
4. **Click**: tapping a terminal edge shows the signed NES + FDR; tapping a chain
   edge shows provenance + weight; tapping background clears popup and isolate.
5. **Wedge tiling**: on the reference row, each spine node's kinase fan sits in its
   own wedge — no two nodes' fans interleave; the arc spine reads Ligand→Target.
6. **Regression (mouse cohorts)**: song / 5xFAD shards regenerate without schema
   error; sub-tab renders.
7. `tests/test_kinase_sidechain_weighting.py` updated: backend test asserts
   `signed_nes` carried and `abs(signed_nes)==best_abs_nes`; viewer test asserts
   terminal-edge element carries sign and the wedge placement is deterministic.

Served over HTTP, hard-refresh, expand `FURIN|ADAM19|TOP2A|SFSWAP` → Sidechains:
arc spine, per-node wedges, red/blue direction on dotted kinase edges, a legend,
and click-for-value.
