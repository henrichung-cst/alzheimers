# Sidechain node-tap relationship table

**Status:** active · **Scope:** viewer-only (`incytr_sidechains.js`)

A QOL addition to the shipped kinase-sidechain view
([`foundation/kinase_sidechain_incytr_graph.md`](../foundation/kinase_sidechain_incytr_graph.md)).
Tapping a node currently isolates its neighborhood + zooms and writes nothing
tabular. Add a non-intrusive per-node relationship **table** so the isolated
neighborhood is also readable as ranked rows.

## Intent

Let the user read a node's relationships as an ordered table — one row per
relationship influencing (or influenced by) the tapped node — with signed NES
and summary counts, without leaving the graph.

## Decisions (aligned)

- **Augment, don't replace.** One node-tap does both: the existing
  `focus(node.closedNeighborhood ∪ spine)` isolate+zoom stays, and the table
  populates. No new interaction; extend the existing `cy.on("tap", "node")`
  handler (L718).
- **Table lives in the existing bottom-left `infoPanel`** (L532), which is
  already full-screen-safe. It sits alongside / reuses the `edgeDetail` region
  (L510). Background-tap and filter-change clear it back to the placeholder
  (extend the existing clear paths at L723-728, L685).
- **Direction-aware rows:**
  - **Spine (pathway) node** → rows = kinases hitting it. From connected
    `terminal-edge`s whose `target` is this node; each row = source kinase,
    `signed_nes`, direction arrow. Summary: `N kinases affecting · E enriched ·
    D depleted`.
  - **Kinase node** → rows = what it points at. Connected `terminal-edge`s whose
    `source` is this kinase (row = target spine node + role + `signed_nes` +
    direction) **plus** connected `chain-edge`s (row = other kinase +
    `provenance` + `weight`, no NES). Summary: `targets N nodes · M kinases`.
- **Order:** terminal rows by `|signed_nes|` desc; chain rows (no NES) after,
  by `weight` desc.
- **Pure viewer change.** `signed_nes`, `nes_direction`, `role`, `provenance`,
  `weight` already ride the cy edge data (L444-462, L437-442). No bridge / edge /
  payload / shard / schema-guard changes. No new columns.

## Out of scope

- No backend or `payload_helpers.py` / `verify_payload_contract.py` changes.
- No new NES/site math — reuse edge data as-is.
- Table does **not** re-implement the graph's chain-filter toggle: it lists a
  node's own relationships from the loaded graph regardless of `showChains`
  (it's a text list, not the graph view).
- No CSV/export of the table (YAGNI — add if asked).

## Shape

One helper `_isNodeRelationTable(node, cy)` returning a DOM `<table>` (or a
"no relationships" line), rendered into the info panel from the node-tap handler.
Summary line above the table. Reuse `_isNumeric` / `_isNesDirection` /
`_IS_COLORS` for direction hue. Scroll-capped height so it stays non-intrusive in
the pinned bottom-left panel.

## Success

Tap a spine node → table lists its kinases ranked by |NES| with an
enriched/depleted count line. Tap a kinase node → table lists its target nodes +
chain kinases. Tap background → table clears. Full-screen mode keeps the table
visible. No console errors; graph isolate+zoom behaves exactly as before.

## Verify

Extend `tests/test_kinase_sidechain_weighting.py` (Node-executed against the real
JS) with a case building the table for a known spine node and a known kinase node
from a synthetic shard, asserting row count, ordering by |NES|, and the
enriched/depleted summary counts.
