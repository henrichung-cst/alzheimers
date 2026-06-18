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
axis, or parsed contrast names. The Sender × Receiver heatmap has a shared single-contrast mode and
timeline mode. Timeline mode renders one heatmap frame at a time with a horizontal slider: Song/AD
scrubs one disease group across timepoints, while T-cell scrubs all day-vs-baseline contrasts for
the active donor. Both modes read the same precomputed count cube and click through to the shared
pathway table filters. Dense heatmaps expose display-only controls for axis limiting and log1p color
scaling. Axis limiting keeps the top sender and receiver cell types by visible path count under the
active gates; in timeline mode the axes are ranked across the full slider span so they stay stable
while scrubbing. Log1p color scaling compresses saturated cells while hover text and pathway
drill-down continue to use raw path counts. A shared sparse-cell sensitivity filter is also
available when the active Incytr payload supplies `low_signal_celltypes`; it removes sender-receiver
interactions where either endpoint has median n_cells <= 3 from the heatmap, pathway table, and
temporal pathway counts. Kinase detail and transcript trace
modules are shared through capability/context checks so AD/Song keeps the audit workbench path while
T-cell gets donor-scoped transcript traces and a bulk-only kinase detail summary when audit tables
are absent. Evidence-row rendering is shared and branches on cohort metadata for contrast arm labels
and trace row grouping. The remaining shared modules are byte-identical UI/state helpers used by both
viewers.

### Shared Kinase Motif Logo

`js/widgets/sequence_logo.js` is a shared browser-side port of
`kinase_library.Kinase.seq_logo()` for the kinase detail Measurement Trace panel. The approved
viewer logo follows Kinase Library's default `logo_type="ratio_to_median"` behavior:

```text
height = log2(position_value / per-position median)
```

Letters above zero are residues favored over the position median; letters below zero are residues
disfavored relative to that median. This is intentionally denser than an information-content logo
because it preserves Kinase Library's visual interpretation of motif preferences. For example, AKT1
shows a modest R preference at -5 and a strong R preference at -3, matching the Kinase Library
motif view.

The widget uses Kinase Library's amino-acid color map and phospho-priming display conventions:
lowercase `t` is rendered as `pS/pT`, lowercase `y` as `pY`, and lowercase `s` is dropped, matching
the package defaults. These flanking `pS/pT` and `pY` entries are phospho-priming preferences, not
the central phosphoacceptor. Position 0 is drawn separately from the kinase phosphoacceptor
preference: fixed `Y` for tyrosine kinases and S/T favorability for serine/threonine kinases. The
center stack is scaled to the tallest positive flanking stack, matching Kinase Library's
`make_seq_logo()` behavior.

### Song/AD Incytr Heatmap Saturation

The Song/AD heatmap is intentionally not renormalized by default. A 2026-06-01 audit found that the
current canonical Song outputs are heavily concentrated in three receiver/end-point cell types, and
that the same pattern is present in frozen SCE4 `10302025` outputs after normalizing dotted versus
hyphenated cell-type names. Current receiver shares are Cholinergic-Neurons 51.39%,
Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons 21.07%, and
GABAergic-inhibitory-interneurons-VIP-positive 17.90%. Frozen SCE4 receiver shares are 51.15%,
20.02%, and 17.69%, respectively. The normalized current-vs-SCE4 pair-count correlation is high
across contrast/sender/receiver cells (Pearson 0.993, Spearman 0.972), so this is not a new viewer or
canonical-output artifact.

The same audit found a strong inverse relationship between endpoint path count and cell abundance in
the current male pseudobulk counts (Spearman -0.864 versus receiver paths with |PDS| > 1).
Cholinergic,
VIP-positive, and layer-2-4 pyramidal neurons have median per-sample counts of roughly 2, 2, and 3
cells, respectively. The audit artifact is
`outputs/reports/incytr_pair_mode/cell_count_qc/median_cells_vs_receiver_paths.png`, with the source
table at `outputs/reports/incytr_pair_mode/cell_count_qc/cell_count_incytr_pathway_qc.csv`.

Treat log1p color as a visualization aid for this saturated row universe, not as a scoring
transform. Treat the sparse-cell filter as a sensitivity view; it does not rewrite canonical Incytr
outputs or scoring.

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

Projected state MEA, when present, is supporting evidence for the donor-scoped
bulk kinase MEA. Frontend copy must call it "projected state MEA" or
"state-projected MEA" and must not call it direct cell-state
phosphoproteomics. If `payload.projected_state_mea` is absent for the active
donor, the viewer should behave exactly as it does for current payloads.

Mechanism attribution labels are categorical annotations on paired
stoichiometry-vs-raw MEA evidence. Frontend rendering may show
`mechanism_call`, `sign_relation`, NES, and FDR evidence, but must not introduce
or display a numeric mechanism score.

### Shared Incytr Pathways Controls

The Incytr Pathways tab is shared between the unified and T-cell viewers through
`viewer_shared/template/js/tabs/incytr_pathways.js` and
`viewer_shared/template/js/tabs/incytr_global_index.js`. Viewer-specific `body.html` files may
provide the toolbar mount points, but filtering/export behavior should stay in shared code.

Supported shared filters:

- composite PDS magnitude: `|PDS| >= threshold`;
- composite PDS direction: both, positive/up (`PDS > 0`), or negative/down (`PDS < 0`);
- individual subscore magnitude floors from `block.score_columns`, currently `|TPDS|`, `|PPDS|`,
  `|PhPDS_ps|`, `|PhPDS_py|`, and `|SiK_score|`;
- existing disease/timepoint, sender/receiver, recurrence, trend, sparse-endpoint, and text search
  filters.

`Export CSV` must export the full current filtered result set, not only the visible page. In Top
overall mode this requires `block.global_index` for the active context. In Cell Type mode it exports
the fully filtered loaded sender/receiver shard. Do not reintroduce a build-time top-N payload cap as
the export source.

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
