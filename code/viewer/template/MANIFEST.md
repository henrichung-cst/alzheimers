# Template chunk disposition

Disposition labels for each chunk. Used by Phase 3 (pathway strip) and future
re-implementation work to know what to delete vs keep.

## Shell

| File | Lines | Disposition |
|---|---:|---|
| `index.html.j2` | 28 | KEEP (Jinja shell — strip removes `{{ raw('...') }}` lines for deleted chunks) |
| `head.html` | 17 | KEEP (HTML head + ESM imports) |
| `body.html` | 380 | NEEDS-AUDIT (mixed — pathway tab buttons + filter labels live here) |
| `styles.css` | 467 | NEEDS-AUDIT (pathway-only selectors candidates: `.pe-cchip-*`, `.bb-sup`, sender-matrix grid) |

## JS — shared infrastructure (KEEP)

| File | Lines | Notes |
|---|---:|---|
| `js/01_state.js` | 642 | Store, reducer, audit helpers, AuditDataStore, MeasurementTraceStore |
| `js/02_ui_chrome.js` | 506 | howto, export, drawer, hash |
| `js/03_filters_hash.js` | 191 | filter cache, prereq check, URL hash |
| `js/04_slice_cache.js` | 145 | hyparquet lazy loader (kinase + pathway branches — Phase 3 surgery) |
| `js/05_header.js` | 135 | header + tab population |
| `js/boot.js` | 201 | boot, glossary, cross-tab refresh |

## JS — kinase tabs (KEEP)

| File | Lines | Notes |
|---|---:|---|
| `js/tabs/temporal_v2.js` | 493 | Decomp viz |
| `js/tabs/kinase_explorer.js` | 770 | Kinase Explorer scoring + render |
| `js/tabs/kinase_audit.js` | 1369 | Audit infra + MEA + WMB + SEA-AD + Song panels |
| `js/tabs/kinase_detail.js` | 42 | renderActiveKinaseAuditTab + renderKinaseDetail |
| `js/tabs/kinase_wiring.js` | 344 | wireKinaseTable, filter UI |

## JS — pathway tabs (DELETE in Phase 3)

| File | Lines | Notes |
|---|---:|---|
| `js/tabs/overview.js` | 84 | renderOverview (receiver × contrast TPDS) |
| `js/tabs/sender_matrix.js` | 227 | renderSenderMatrix (sender × receiver heatmap) |
| `js/tabs/pathway_explorer.js` | 290 | renderPathwayExplorer |
| `js/tabs/kinase_backbones.js` | 124 | renderKinaseBackbones — surgical removal from kinase audit workbench |

## JS — mixed, slated for refactor (DELETE in Phase 3, rebuild in Phase 5+)

User decision 2026-05-06: these tabs are deleted alongside pure-pathway in
Phase 3. The kinase visualizations they currently host (Temporal Kinase,
Additivity Kinase, Graph) get rebuilt against the factorial output.

| File | Lines | Notes |
|---|---:|---|
| `js/tabs/temporal.js` | 347 | renderTemporal dispatcher + renderTemporalKinase + renderTemporalBackbone + wiring |
| `js/tabs/additivity.js` | 380 | renderAdditivity dispatcher + Kinase + Backbone + helpers + wiring |
| `js/tabs/graph.js` | 342 | Cytoscape graph (kinase- and pathway-driven layouts) |

## Verification

Run `pixi run python code/viewer/verify_template.py` to confirm Jinja-rendered
output is byte-equivalent to the legacy `HTML_TEMPLATE` in
`build_unified_viewer.py` (sentinels not yet substituted on either side).
