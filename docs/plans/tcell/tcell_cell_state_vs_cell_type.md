# T-cell viewer: separate cell *state* from cell *type* specificity

## Context

In the T-cell viewer (Kinase Explorer + Attribution tab) the columns labeled
"Cell type" and "Specificity" are both derived from **ProjecTILs T-cell states**
(the 14 `functional.cluster` labels: CD8CM, CD8Tex, Treg, CD4Tfh, … via
`PROJECTILS_LABEL_MAP`). The "Specificity" value is `tcell_state_enrichment` =
fold over the kinase's *median detected state* — i.e. **within-T-cell-state**
enrichment. Calling these "cell type" is misleading: they describe which T-cell
*state* a kinase localizes to, not whether the kinase is specific to a cell type
beyond T cells.

We *do* have true cross-cell-type information from the independent **NSCLC
reference** (10x 897k-cell), already computed in `alz/reference/nsclc_expression.py`
into `nsclc_kinase_specificity.csv` at coarse 7-group resolution (`T_NK` +
`B_plasma`, `Myeloid`, `Epithelial`, `Endothelial`, `Fibroblast`, `Mast`). This
already drives the §2 lineage strip in the Attribution tab — but it is **not**
surfaced as a sortable column in the Kinase Explorer.

**Outcome:** relabel the state-derived columns honestly ("Cell state",
"State specificity"), and add a new per-kinase **"Cell types"** column to the
Kinase Explorer that reports NSCLC cross-lineage breadth (lineages detected n/7,
top lineage as badge). No new metric computation — the values already exist; we
denormalize the per-gene coarse summaries into the per-kinase slice.

Scope: **T-cell viewer only** (`alz/tcell_viewer/` + `alz/build_tcell_viewer.py`).
The unified viewer is unaffected.

## Decisions (confirmed)

- New "Cell types" column sorts on **lineages detected (n/7)** (`n_detected_coarse`
  / number of lineage rows for that gene); cell shows the **top lineage** badge
  (`top_group_coarse`). "n/a" when the kinase is outside the NSCLC probe panel.
- Existing Explorer **"Specificity" → "State specificity"**.
- The state-count column (`n_attributed_celltypes`, currently mislabeled
  "Cell types") **→ "Cell states"** — this frees the name "Cell types" for the
  new NSCLC column.
- **No new column in the Attribution tab** — the §2 NSCLC lineage strip already
  shows full per-lineage breadth. Attribution tab gets relabels only.

## Build side — `alz/build_tcell_viewer.py`

The NSCLC coarse breadth is gene-keyed and cohort-independent; load once and
attach per-kinase slice columns next to where `top_celltype_1` is set
(`_build_donor_kinases_slice`, ~lines 835–868).

1. **New loader** mirroring `_load_nsclc_detection()` (line 553), reading
   `config.NSCLC_KINASE_SPECIFICITY_FILE` (already a config const, already
   registered in AuditDataStore at ~line 1803):
   ```python
   def _load_nsclc_coarse_breadth() -> dict:
       """Per-gene NSCLC cross-lineage breadth from the coarse spec file.
       Returns {GENE_UPPER: (n_detected_coarse, n_groups, top_group_coarse)}."""
   ```
   Group `nsclc_kinase_specificity.csv` by `gene_symbol`; per gene take
   `n_detected_coarse` and `top_group_coarse` (denormalized on every row — read
   first), and `n_groups` = count of `spec_group` rows for that gene (the strip's
   `nGroups` denominator).
2. **Emit slice columns** keyed by `gene_upper_map` (reuse `_upper_gene_map`,
   line 541) for each kinase row in `rows`:
   - `cols["nsclc_lineages_detected"]` — int or `None` (gene absent from panel)
   - `cols["nsclc_lineages_total"]` — int or `None`
   - `cols["nsclc_top_lineage"]` — str or `""`
   These ride the existing `kinases_slice` returned at line 871, exposed to JS via
   `ViewerPayload.kinases()` as parallel column arrays (same shape as the existing
   per-kinase fields like `top_celltype_1`, `peak_NES`).

## JS / template — Kinase Explorer

`alz/tcell_viewer/template/body.html`:
- Header `tcell_max_enrich` (line 74): label **"Specificity" → "State
  specificity"**; keep the existing title text (it already says "within-cohort
  state enrichment").
- Header `n_attributed_celltypes` (line 75): label **"Cell types" → "Cell
  states"**; its tooltip already says "States this kinase is enriched in".
- Toolbar filter label (line 54): **"Specificity ≥" → "State specificity ≥"**.
- **New `<th data-col="nsclc_lineages" data-metric="nsclcLineages">Cell types</th>`**
  after the "Cell states" column. Tooltip must mark provenance: independent NSCLC
  reference (10x 897k-cell), detection ≥10% gate across coarse lineages — distinct
  from the within-cohort state columns.

`alz/tcell_viewer/template/js/tabs/kinase_explorer.js`:
- **New renderer** `_renderNSCLCBreadthCell(r)` reading the slice fields via the
  `ViewerPayload.kinases()` column arrays (same accessor pattern the per-kinase
  table body already uses): top lineage as a `badge` span + `"<n>/<total>"`;
  `<span class="muted">n/a</span>` when `nsclc_lineages_total` is null. Reuse
  `_escapeHtml`.
- **New sort branch** for `col === "nsclc_lineages"` in the comparator block
  (alongside the existing `n_attributed_celltypes` branch, ~lines 331–345),
  sorting by `nsclc_lineages_detected` (null sorts last).
- Wire the new cell into the table-body row template next to the existing
  `specBadge` / cell-types cell (~line 672).

## JS / template — Attribution tab (relabels only)

`alz/tcell_viewer/template/js/tabs/attribution_manifest_tcell.js`:
- Column `cell_type` (line 45): label **"Cell type" → "Cell state"**. Keep the
  `attr-celltype` CSS class and chevron.
- Super-group header (line 119): **"Within-cohort cell-type attribution" →
  "Within-cohort cell-state attribution"**; same wording fix in the explainer
  caption if it repeats the phrase.
- Leave the §2 NSCLC lineage strip (`kinase_audit.js` `_renderNSCLCLineageStrip`)
  unchanged — it already labels its content as lineages/cell types correctly.

## Sweep for stray "cell type" strings

Grep the T-cell viewer templates (`alz/tcell_viewer/template/`) for
`cell type` / `Cell Type` / `celltype` in **user-facing** strings (headers,
tooltips, captions, the header selection chip). Relabel those that describe
ProjecTILs **states** to "cell state"; leave NSCLC-lineage strings as "cell
type". Do **not** rename internal identifiers (`selection.celltype`,
`attr-celltype`, `data-col`, payload keys) — labels only.

## Verification

1. `pixi run python -c "import ast; ast.parse(open('alz/build_tcell_viewer.py').read())"`
   and `node --check` on the two edited JS files.
2. Rebuild under a memory cap (shared box):
   `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 pixi run tcell-viewer`.
3. Open the T-cell viewer, **hard-refresh** (Ctrl+Shift+R — inlined PAYLOAD
   caches otherwise). Confirm in the **Kinase Explorer**:
   - headers read **State specificity**, **Cell states**, and a new **Cell types**;
   - the new column shows `top lineage` + `n/7`, sorts correctly, and shows **n/a**
     for a kinase outside the NSCLC panel.
   - Spot-check: a T-restricted kinase → low n/7; a housekeeping kinase → high n/7.
4. Confirm the **Attribution tab** first column now reads **Cell state** and the
   §2 lineage strip is unchanged.

On approval, copy this plan to `docs/plans/tcell/tcell_cell_state_vs_cell_type.md`
(repo is the durable review location per project rules) before implementing.
