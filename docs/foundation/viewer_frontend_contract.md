# Viewer Frontend Contract

This document defines how the AD/Song unified viewer and T-cell viewer share frontend code while
preserving intentional cohort-specific behavior. The payload contract is defined separately in
`docs/foundation/viewer_payload_contract.md`; this document covers template and JavaScript ownership.

## Policy

Both viewers should render from the same payload schema and share frontend modules whenever behavior is
the same. Dataset-specific builders may stay separate, but duplicated viewer JavaScript is treated as
cleanup debt unless it represents a documented cohort-specific fork.

Shared modules live under:

```text
alz/viewer_shared/template/js/
```

The AD/Song builder and T-cell builder both resolve `raw("...")` includes from the local viewer
template first, then from `viewer_shared`. A local file with the same path intentionally overrides the
shared module.

## Current Shared Modules

The following modules are shared by both viewers:

```text
js/00_payload_adapter.js
js/03_filters_hash.js
js/04_slice_cache.js
js/05_header.js
js/boot.js
js/tabs/incytr_state.js
js/tabs/incytr_heatmap.js
js/tabs/incytr_pathways.js
js/tabs/kinase_detail.js
js/widgets/evidence_row.js
js/widgets/multiselect.js
js/widgets/sequence_logo.js
js/widgets/transcript_trace.js
```

The payload adapter is the canonical read layer for `meta.contexts`, `selection.context`,
`*.by_context`, and Incytr shard filenames. `SliceCache` is the canonical lazy parquet loader for
backbone, decomp OLS, Song concordance, human per-donor, and context-scoped Incytr pathway shards.
Hash/prerequisite handling is shared and uses `selection.context` plus `ctx=` as the canonical URL
state, while preserving old inbound `d=` links.
Header and boot handling are shared; context controls are optional DOM affordances and no-op in
single-context viewers.
The Incytr tabs are shared and derive contrast groups/timepoints from the payload block, context
axis, or parsed contrast names. Kinase detail and transcript trace modules are shared through
capability/context checks so AD/Song keeps the audit workbench path while T-cell gets donor-scoped
transcript traces and a bulk-only kinase detail summary when audit tables are absent. Evidence-row
rendering is shared and branches on cohort metadata for contrast arm labels and trace row grouping.
The remaining shared modules are byte-identical UI/state helpers used by both viewers.

## Intentional Viewer Differences

### AD/Song Unified Viewer

The AD/Song viewer is allowed to keep local modules for behavior that depends on the mouse AD design
or human reference integration:

- context `song_ad`;
- App/Tau/ApTt by 2mo/4mo/6mo contrast grid;
- human/Mukesh mode and `kinase_human.js`;
- decomposition and crosstable views;
- kinase cell-type attribution, WMB specificity, agreement profile, and confidence tiers;
- Song concordance and decomp OLS lazy evidence.

### T-Cell Viewer

The T-cell viewer is allowed to keep local modules for behavior that depends on donor-scoped T-cell
inputs:

- contexts `donor1` and `donor2`;
- donor context selector in the header;
- day-vs-d2 contrast vocabulary;
- donor1-only kinase MEA and donor2 no-IMAC message;
- trajectory-pattern filters for the donor1 kinase profile;
- T-cell-specific copy warning that per-animal SigProb p-values are unreliable and `|PDS|` is the
  primary Incytr gate.

## Consolidation Candidates

These modules are parallel implementations and should be consolidated incrementally behind shared
configuration or small view-specific hooks:

```text
js/01_state.js
js/02_ui_chrome.js
js/tabs/kinase_audit.js
```

High-priority consolidation targets:

1. Incytr tab copy: visible labels and caveats are still owned by each viewer's `body.html`; if a
   future viewer needs different Incytr controls, prefer config/data attributes over forking the tab
   logic.

## Fork Rules

A local viewer file is acceptable when:

- the payload capability differs, for example `human_reference` or `decomp_ols`;
- the table columns or detail panel are genuinely different;
- a context has intentional missing data, for example donor2 kinase MEA;
- the viewer needs cohort-specific explanatory copy.

A local viewer file should be refactored when:

- it only differs by labels such as Disease/Timepoint versus Day/Baseline;
- it hard-codes contrast names that are already present in `meta.contexts`;
- it duplicates generic sorting, filtering, hash, cache, or tab wiring logic;
- a fix would reasonably need to be applied to both viewers.

## Verification

After moving any frontend module into `viewer_shared`, run:

```bash
node --check alz/viewer_shared/template/js/<module>.js
pixi run python alz/build_unified_viewer.py --html --validate --skip-verify
pixi run python alz/build_tcell_viewer.py --html --validate
pixi run python alz/viewer/verify_payload_contract.py \
  outputs/reports/unified_viewer/unified_viewer.payload.json \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json
```

Browser smoke should cover AD/Song Incytr heatmap/pathways, AD/Song kinase explorer, T-cell donor1
Incytr/kinase, T-cell donor2 Incytr, and the donor2 no-kinase message.

Manual browser smoke for both generated viewers was completed on 2026-06-01 after the shared-module
refactor. Both the AD/Song unified viewer and T-cell viewer loaded and preserved their expected core
interactions.
