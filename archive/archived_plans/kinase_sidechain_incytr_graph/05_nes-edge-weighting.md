# 05 — NES-weighted sidechain edges

**Unit:** carry per-kinase motif enrichment (NES) through the sidechain backend →
payload → viewer, so kinase→gene edges can be weighted and pruned by a real
per-kinase signal. Fixes the unreadable edge nest.

**Risk:** medium. The backend/payload changes fail loudly (schema tuple checks).
The viewer selection math is the silent spot — a wrong top-N or threshold renders
plausibly but hides the wrong edges. Verify against the numbers in this doc.

**Depends on:** 01 (backend edge model), 03 (payload slice), 04 (viewer sub-tab)
— all shipped. This modifies their outputs; it does not re-lay their foundations.

---

## Why (the finding)

Reproduced on `FURIN|ADAM19|TOP2A|SFSWAP`, cohort `tcells` / context `donor1`,
contrast `d13_d2`. `_isGraphForRow` currently renders **362 nodes + 6261 chain
edges + 191 terminal edges** (interactome total is 392 nodes / 6676 edges) — the
edges occlude into an indiscernible mat. Two independent causes:

1. **The kinase→gene (terminal) edge weight is per-spine-gene, not per-kinase.**
   `build_terminal_map` sets `weight_motif = norm(best_abs_pds)`, and
   `best_abs_pds` is the pathway node's own dysregulation, replicated identically
   across every kinase attached to that node. Measured: every kinase→FURIN,
   →ADAM19, →TOP2A edge has PDS `2.139`; every kinase→SFSWAP has `1.559`. So
   `weight ∈ {0.729, 1.0}` — a 2-valued flag keyed to *which node the edge lands
   on*, carrying **zero** per-kinase ranking. All 75 kinases on ADAM19 draw at
   identical width. There is no honest way to weight or top-N these edges today.

2. **The upstream walk takes the full transitive closure.** The reverse-BFS in
   `_isGraphForRow` (unbounded depth) pulls in 92% of all interactome edges from
   any nontrivial root set, because the interactome is dense (avg degree ~17).

The signal that makes kinase→gene weighting meaningful **already exists upstream
and is discarded**: `kinase_node_hits.parquet` carries `NES` (motif enrichment of
that specific kinase on that gene) and `FDR`, per `(kinase, gene_symbol, role,
contrast)`. `load_motif_edges` aggregates with `MAX(best_abs_pds)` and drops NES.

NES discriminates and is biologically sensible. On ADAM19 (d13_d2, FDR<0.05),
ranked by |NES|: **CDK2 (2.98), CDK4 (2.66), LRRK2 (2.33), NLK (2.21), CDK12
(2.19)…** — 73 distinct values across 75 kinases (vs. 1 for `best_abs_pds`).

Present in every cohort's `kinase_node_hits.parquet` (song, fivexfad_cortex,
fivexfad_hippocampus, tcells_donor1): columns `NES`, `FDR` confirmed.

**Payoff** (top-N strongest kinases per spine gene, FDR<0.05, chain edges induced
among drawn kinases only):

| keep per gene | kinases drawn | terminal edges | chain edges |
|---|---|---|---|
| top 5  | 15 | 20 | ~4 |
| top 10 | 31 | 40 | ~14 |
| top 15 | 44 | 56 | ~14–42 |

Readable, and every edge shown carries a real per-kinase strength.

---

## Changes

### A. Backend — `alz/cross_reference/kinase_kinase_edges.py`

Carry `NES`/`FDR` out of the bridge reduction and make the **motif weight
component driven by |NES|, not |PDS|**, in both the terminal map and the
interactome. `best_abs_pds` is **retained as an informational column** (pathway
dysregulation context — a distinct axis from motif enrichment); it is no longer
the weight driver. This is not a shim: PDS and NES are different measurements,
both kept, only the *weight source* changes.

1. **`load_motif_edges`** (DuckDB aggregation, ~L174-200). Add to the SELECT /
   GROUP-BY output:
   ```sql
   MAX(ABS(NES))  AS best_abs_nes,
   MIN(FDR)       AS best_fdr,
   ```
   (NES is signed; magnitude is what ranks. FDR: the strongest-evidence value for
   the pair is the smallest.) Add `best_abs_nes`, `best_fdr` to the returned
   column list (~L198-200). Keep `best_abs_pds`.

2. **`build_terminal_map`** (~L284-314):
   - `motif_ceiling = float(motif_edges["best_abs_nes"].max())` (was `best_abs_pds`).
   - `df["weight_motif"] = _norm(df["best_abs_nes"].fillna(0.0), motif_ceiling)`
     (was `best_abs_pds`).
   - `weight = weight_lit + weight_motif` unchanged (literature corroboration on
     `both` edges still adds; terminal edges are motif-anchored so `weight_lit` is
     usually 0).
   - Add `best_abs_nes`, `best_fdr` to the output `cols` list. Keep `best_abs_pds`.

3. **`build_interactome`** (~L230-281): same bug, same fix for consistency.
   - `motif_agg` aggregation: `weight_motif_raw=("best_abs_nes", "max")` (was
     `best_abs_pds`). Requires `best_abs_nes` to survive the `load_motif_edges`
     change above (it does).
   - Everything downstream (`motif_ceiling`, `weight_motif`, `weight`) already
     reads `weight_motif_raw`; no other change. Interactome output columns
     unchanged (it never carried `best_abs_pds`).

4. **MANIFEST text** (~L398-401): update the weight description from
   `norm(|PDS|)` to `norm(|NES|)` for the motif component.

Output schema after this step:
- `terminal_edges.csv`: append `best_abs_nes`, `best_fdr` (keep all existing).
- `interactome.csv`: unchanged column set.

### B. Payload — `alz/viewer/shared/payload_helpers.py`

The shard schema is validated with exact tuple equality, so extend the constant:

- `_INCYTR_SIDECHAIN_TERMINAL_COLUMNS` (~L26-29): append `"best_abs_nes"`,
  `"best_fdr"` to the tuple (order must match the CSV column order emitted in A).
- `_INCYTR_SIDECHAIN_INTERACTOME_COLUMNS`: unchanged.

No other change — `_sidechain_columns` copies whatever columns the tuple names,
and the tuple-equality guard at L244-248 will now require the new CSV columns
(fails loudly if the backend was not regenerated first).

### C. Viewer — `alz/viewer_shared/template/js/tabs/incytr_sidechains.js`

Three edits to `_isGraphForRow` (L204-258) and its consumers:

1. **Read the new columns.** Add `"best_abs_nes"`, `"best_fdr"` to the
   `_isSafeRows` field list for `terminal` (L205-210). `edge.weight` is already
   `norm(|NES|)` after A/B, so `_isEdgeWidth(edge.weight, ...)` now scales terminal
   width meaningfully — the existing terminal-edge style already uses
   `"width": "data(width)"` (L365-368); no style change needed, but drop the
   `line-style: dotted` uniform look if desired so thickness reads.

2. **Top-N terminal kinases per role by weight, FDR-gated.** Replace the current
   "keep every terminal edge whose target matches the spine" filter (L213-217)
   with: for each spine role, keep edges with `best_fdr < FDR_MAX`, sort by
   `weight` desc, take the top `TERMINAL_TOP_N_PER_ROLE`. Suggested defaults:
   `FDR_MAX = 0.05`, `TERMINAL_TOP_N_PER_ROLE = 10` (yields ~31 nodes / 40
   terminal edges on the reference row — see payoff table). Surface both as named
   constants near `_IS_LAYOUT`.

3. **Kill the transitive closure; induce chain edges on drawn kinases.** Replace
   the reverse-BFS (L218-249) with: `kinaseGenes = set(top-N terminal sources)`;
   `chainEdges = interactome.filter(e => kinaseGenes.has(e.source_gene) &&
   kinaseGenes.has(e.target_gene) && e.weight >= CHAIN_WEIGHT_MIN)`. Drop the
   `distance`/depth machinery and its use in `_isPositionedElements` layout
   (L292-297) — with induced edges there is no depth axis; lay kinases out around
   their terminal anchor without the `depth * depthXOffsetPx` shift. Suggested
   `CHAIN_WEIGHT_MIN` = median interactome weight (~0.81) or 0; tune against the
   payoff table (p50 keeps ~13 chain edges at top-10, p0 keeps ~14).

4. **Caption** (L337-339): "N strongest regulators per node (|NES|), FDR<0.05;
   M upstream kinase edges." Note the hidden count per node so nothing is silently
   dropped — e.g. append the number of FDR-passing kinases not shown per role.
   (The single-role/multi-role framing is **not** used; selection is purely
   top-N-by-NES.)

`observedMax` (L255-256) still computes over the drawn edges for width scaling —
recompute it over the *selected* terminal + chain edges, not the full frame, so
the strongest drawn edge maps to max width.

---

## Regenerate + verify

Backend (per cohort; the t-cell one feeds the reference example):

```
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort tcells
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort song
pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort fivexfad
```

Then rebuild the viewers so the shards carry the new columns:

```
pixi run viewer            # unified (song + 5xFAD)
pixi run python -m alz.build_tcell_viewer   # or the tcell viewer task
```

### Acceptance checks

1. **Schema**: `terminal_edges.csv` has `best_abs_nes`, `best_fdr`; the shard's
   `terminal_edges` columns include them (else B's tuple guard raises).
2. **NES is per-kinase**: in the regenerated t-cell shard, for contrast `d13_d2`,
   `target_gene=ADAM19`, `role=Receptor`, `weight` has >50 distinct values (was
   2). Top kinases by weight = CDK2, CDK4, LRRK2 (matches the finding).
3. **Graph is bounded**: rendering the reference row draws ≤ ~35 nodes and ≤ ~60
   edges total (vs. 362 / 6452). No transitive closure — every chain edge has both
   endpoints in the drawn terminal-kinase set.
4. **PDS retained**: `best_abs_pds` still present in the shard (informational).
5. **Regression (mouse cohorts)**: song / 5xFAD shards regenerate without schema
   error and their sidechain sub-tab still renders.
6. `pixi run python alz/viewer/verify_payload_contract.py` (or the sidechain
   contract check) passes.

Served over HTTP (not `file://`), hard-refresh, expand the `FURIN|ADAM19|TOP2A|
SFSWAP` row → Sidechains sub-tab: legible graph, thick edges on high-|NES|
kinases.
