# Viewer Payload Contract

This document defines the shared payload contract for interactive viewers. The goal is to let
dataset-specific builders stay separate while emitting a common frontend-facing shape. New cohorts
should adapt their data to this contract rather than forking viewer behavior.

## Design Rule

Builders are private adapters. They can read cohort-specific files, reshape data, and decide which
features are available, but they should emit the same normalized viewer payload schema.

The frontend should depend on:

- `payload.meta.contexts`
- `payload.meta.capabilities`
- `payload.<block>.by_context`
- `payload.edge_slice_ref`

It should not need to know which Python builder produced the payload.

## Schema Version

The standardized contract is `viewer_payload_schema_version = 2`.

Current builders emit shared domain blocks through v2 `by_context` only. Legacy payloads may still
include flat top-level blocks such as `payload.kinases.id` or
`payload.incytr_pathways.slice_index.present`; the shared adapter keeps a backward-compatible flat
fallback for those older artifacts. New frontend code should read through `ViewerPayload`, which
prefers v2 and treats legacy flat payloads as compatibility input only.

## Required Top-Level Keys

Every viewer payload must include:

```text
payload.meta
payload.audit_tables
payload.edge_slice_ref
```

Payloads should include these domain blocks when available:

```text
payload.kinases
payload.celltypes
payload.incytr_pathways
payload.kinase_motifs
payload.attribution_index
payload.decomposition_index
payload.agreement_index
payload.subclass_breakdown
payload.human
```

Unavailable domain blocks should be omitted or represented as empty context slices, but capability
flags must make the absence explicit.

## Meta Contract

`payload.meta` is the routing contract for the frontend.

Required fields:

```json
{
  "schema_version": 1,
  "viewer_payload_schema_version": 2,
  "generated_at": "ISO-8601 timestamp",
  "cohort": "song_ad | tcell | other",
  "default_context": "context id",
  "contexts": [],
  "capabilities": {}
}
```

### Contexts

A context is the smallest top-level unit the viewer can switch between without rebuilding. For the
current viewers:

- Song/AD has one context, for example `song_ad`.
- T-cell has one context per donor, for example `donor1`, `donor2`.
- A future cohort could use one context per donor, treatment arm, tissue, species, or batch, as long
  as each context has internally consistent axes.

Each context record should contain:

```json
{
  "id": "donor1",
  "label": "Donor 1",
  "cohort": "tcell",
  "axis_kind": "donor | cohort | treatment | tissue | other",
  "contrasts": ["d13_d2", "d17_d2", "d20_d2"],
  "contrast_axis": {
    "primary": "day",
    "baseline": "d2",
    "groups": ["d13", "d17", "d20"],
    "timepoints": ["d13", "d17", "d20"]
  },
  "celltypes": ["CD8Tex", "CD4Th17"],
  "capabilities": {
    "kinases": true,
    "incytr": true,
    "decomp_ols": false,
    "song_concordance": false,
    "human_reference": false,
    "transcript_trace": true,
    "omics_trace": false
  },
  "notes": []
}
```

The context-level `capabilities` object overrides or narrows global capabilities. For example,
T-cell globally supports kinases because donor1 has MEA, but `donor2.capabilities.kinases = false`
because donor2 has no IMAC.

### Capabilities

Global capability flags declare which feature families the payload may contain:

```json
{
  "contexts": true,
  "kinases": true,
  "celltypes": true,
  "incytr": true,
  "decomp_ols": true,
  "song_concordance": true,
  "human_reference": true,
  "subclass_breakdown": true,
  "audit_tables": true,
  "transcript_trace": true,
  "omics_trace": true
}
```

Frontend tabs should use capabilities to decide whether to render a tab, an empty-state message, or
a disabled control. They should not infer availability from file existence or from cohort names.

## Context-Aware Blocks

Blocks that vary by context must expose `by_context`.

### Kinases

```json
{
  "kinases": {
    "by_context": {
      "song_ad": {"id": [], "name": [], "gene_symbol": [], "NES_App_2mo": []},
      "donor1": {"id": [], "name": [], "gene_symbol": [], "NES_d13_d2": []},
      "donor2": {"id": [], "name": [], "gene_symbol": []}
    }
  }
}
```

The slice for a context with no kinase MEA should be column-compatible where practical, but may have
zero rows. The context's capability flag must explain the absence.

Older payloads emitted a flat fallback during migration:

```json
{
  "kinases": {
    "by_context": {"donor1": {}},
    "id": [],
    "name": []
  }
}
```

New JS should use `kinases.by_context[active_context]`. Current builders emit canonical
`by_context` blocks; the adapter accepts older flat payloads only for backward compatibility.

### Kinase Motifs

`payload.kinase_motifs` is a global lookup keyed by Kinase Library kinase name. The shared
Measurement Trace motif widget consumes this block in both the AD/Song and T-cell viewers. Each
entry should provide the normalized Kinase Library matrix and phosphoacceptor metadata:

```json
{
  "kinase_motifs": {
    "AKT1": {
      "kin_type": "ser_thr",
      "positions": [-5, -4, -3, -2, -1, 1, 2, 3, 4],
      "amino_acids": ["P", "G", "A", "C", "S", "T", "...", "s", "t", "y"],
      "matrix": [[0.1, 0.2]],
      "st_fav": {"S": 1.0, "T": 0.64}
    }
  }
}
```

The frontend renders this block with Kinase Library's default sequence-logo interpretation:
`log2(position_value / per-position median)`. Flanking lowercase `t` and `y` are displayed as
`pS/pT` and `pY` phospho-priming preferences, while lowercase `s` is dropped to match
`kinase_library.Kinase.seq_logo()` defaults. Position 0 is not stored in `positions`; it is inserted
by the frontend from `kin_type` and `st_fav`.

### Cell Types

Cell-type blocks may be global or context-aware:

```json
{
  "celltypes": {
    "by_context": {
      "song_ad": {"id": [], "name": []},
      "donor1": {"id": [], "name": []},
      "donor2": {"id": [], "name": []}
    }
  }
}
```

If a shared union is emitted for convenience, it must not replace context membership. Context
membership is required because future datasets may have non-overlapping or partially overlapping cell
type vocabularies.

### Incytr Pathways

Incytr pathway payloads should expose context-specific shard indexes:

```json
{
  "incytr_pathways": {
    "schema_version": 1,
    "by_context": {
      "song_ad": {
        "contrasts": ["App_2mo"],
        "senders": [],
        "receivers": [],
        "slice_index": {
          "filename_template": "{sender}__{receiver}.parquet",
          "present": []
        }
      },
      "donor1": {
        "contrasts": ["d13_d2"],
        "senders": [],
        "receivers": [],
        "slice_index": {
          "filename_template": "{context}__{sender}__{receiver}.parquet",
          "present": []
        }
      }
    }
  }
}
```

The frontend should never hard-code donor prefixes or mouse-only filename templates. It should read
`filename_template` and the active context's `present` list.

Song/AD Incytr payloads may also include sparse-cell QC metadata used for interpretation-only
sensitivity views:

```json
{
  "celltype_qc": {
    "source": "outputs/reports/snrna_integration/pseudobulk_cell_counts.csv",
    "sample_scope": "samples whose id contains '_ma_'",
    "low_signal_rule": "median_n <= 3",
    "low_signal_celltypes": [],
    "by_celltype": {
      "Cholinergic-Neurons": {
        "median_n": 2.0,
        "mean_n": 2.46,
        "min_n": 1,
        "total_n": 32,
        "n_samples": 13,
        "low_signal_median_le_3": true
      }
    }
  },
  "low_signal_celltypes": [],
  "pathway_counts_low_signal_excluded": {}
}
```

The sparse-cell rule is not a canonical output filter. It supports viewer-side sensitivity mode by
excluding sender-receiver interactions where either endpoint is in `low_signal_celltypes`. Temporal
pathway counts must use the precomputed `pathway_counts_low_signal_excluded` cube instead of scanning
lazy pathway shards in the browser. Payloads without this metadata should hide or disable the sparse
filter and fall back to normal `pathway_counts`.

## Verification

Run the payload contract verifier after rebuilding viewer payloads:

```bash
python alz/viewer/verify_payload_contract.py \
  outputs/reports/unified_viewer/unified_viewer.payload.json \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json
```

The verifier checks schema version, context metadata, canonical `by_context` blocks, Incytr slice
indexes, and context capability consistency. It also fails if the canonical shared blocks regress to
deprecated `by_donor` aliases.

### Edge Slice References

`payload.edge_slice_ref` remains the URL map for lazy shards:

```json
{
  "edge_slice_ref": {
    "schema_version": 1,
    "incytr_pathways_url": "edge_slices/incytr_pathways/",
    "incytr_pathways_index": "edge_slices/incytr_pathways/index.json",
    "decomp_ols_url": "edge_slices/decomp_ols/",
    "decomp_ols_index": "edge_slices/decomp_ols/index.json"
  }
}
```

Missing URLs mean the feature is unavailable. Capability flags should still declare the intended
absence.

## Adapter Requirement

Frontend code should access payload data through a small adapter layer, not by directly reaching into
flat legacy keys. The adapter should provide:

```text
ViewerPayload.schemaVersion()
ViewerPayload.contexts()
ViewerPayload.activeContext()
ViewerPayload.contextCapabilities(contextId)
ViewerPayload.kinases(contextId)
ViewerPayload.celltypes(contextId)
ViewerPayload.incytr(contextId)
ViewerPayload.incytrSliceIndex(contextId)
ViewerPayload.edgeUrl(kind)
```

Adapter behavior:

- Prefer v2 `by_context`.
- Fall back to legacy flat blocks only when reading older payloads with no v2 fields.
- Return explicit empty slices for unavailable features.
- Surface a reason string from `context.notes` or `meta.notes` when data are intentionally absent.

The adapter must not depend on T-cell `by_donor` or `selection.donor`. Those names describe source
data provenance inside builders, not the frontend payload contract.

The shared adapter source lives at `alz/viewer_shared/template/js/00_payload_adapter.js`. Both viewer
builders resolve that shared file when rendering `raw("js/00_payload_adapter.js")`; do not recreate
per-viewer copies unless the adapter behavior actually diverges. Broader frontend sharing policy is
tracked in `docs/foundation/viewer_frontend_contract.md`.

## Current Mapping

### Song/AD Unified Viewer

Current context target:

```json
{
  "id": "song_ad",
  "label": "Song AD",
  "cohort": "song_ad",
  "axis_kind": "cohort",
  "contrasts": [
    "App_2mo", "App_4mo", "App_6mo",
    "Tau_2mo", "Tau_4mo", "Tau_6mo",
    "ApTt_2mo", "ApTt_4mo", "ApTt_6mo"
  ],
  "capabilities": {
    "kinases": true,
    "incytr": true,
    "decomp_ols": true,
    "song_concordance": true,
    "human_reference": true,
    "transcript_trace": true,
    "omics_trace": true
  }
}
```

### T-Cell Viewer

Current context targets:

```json
[
  {
    "id": "donor1",
    "label": "Donor 1",
    "cohort": "tcell",
    "axis_kind": "donor",
    "contrasts": ["d13_d2", "d17_d2", "d20_d2"],
    "capabilities": {
      "kinases": true,
      "incytr": true,
      "transcript_trace": true,
      "decomp_ols": false,
      "human_reference": false
    }
  },
  {
    "id": "donor2",
    "label": "Donor 2",
    "cohort": "tcell",
    "axis_kind": "donor",
    "contrasts": ["d5_d2", "d7_d2", "d9_d2", "d11_d2"],
    "capabilities": {
      "kinases": false,
      "incytr": true,
      "transcript_trace": true,
      "decomp_ols": false,
      "human_reference": false
    },
    "notes": ["Donor 2 has no IMAC; kinase MEA is intentionally unavailable."]
  }
]
```

## Migration Plan

1. Add the adapter layer to both viewers while preserving current flat payload reads.
2. Update both builders to emit `meta.viewer_payload_schema_version = 2`, `meta.contexts`, and
   `meta.capabilities`.
3. Add `by_context` to `kinases`, `celltypes`, and `incytr_pathways`.
4. Refactor shared JS tabs to read through the adapter.
5. After both viewers validate, remove cohort-specific direct reads from shared tabs.
6. Remove emitted flat fallback blocks after both current viewers build successfully from v2-only
   blocks.

Current status as of 2026-06-01: steps 1-5 are implemented for the AD unified viewer and the T-cell
viewer. Shared
viewer modules now read active-context kinases, cell types, Incytr blocks, contrast axes, and shard
filenames through `ViewerPayload`. The T-cell viewer uses `selection.context`, `ctx=`, and
`by_context` as canonical routing. Legacy `#d=...` URLs are accepted only as an inbound shim and are
rewritten through context state. The T-cell builder no longer emits `by_donor` aliases for
`kinases`, `celltypes`, `incytr_pathways`, or `meta.transcript_trace`. The AD builder no longer
duplicates flat `kinases`, `celltypes`, or `incytr_pathways` blocks outside `by_context`. Initial
frontend deduplication has also moved byte-identical shared modules into `viewer_shared`:
`tabs/incytr_state.js`, `widgets/multiselect.js`, and `widgets/sequence_logo.js`. `SliceCache`
is also shared and uses context-scoped Incytr shard indexes by default while retaining legacy flat
index fallback for older payloads. The Incytr heatmap and pathways tabs are shared as well; they
derive disease/day and timepoint/baseline vocabularies from each active context instead of
hard-coding the AD/Song contrast grid. Kinase detail and transcript trace modules are also shared;
they branch on payload capabilities/context metadata rather than cohort-specific filenames. Hash and
prerequisite routing is shared through `ctx=` and active-context defaults. Header, boot, and
evidence-row rendering are shared with optional context-switch and cohort-label support. Manual
browser smoke for both generated viewers passed on 2026-06-01 after this shared-module refactor.

## Deprecation Targets

The migration is intended to remove old routing assumptions, not just add a parallel schema.
Deprecated code paths are:

- Frontend reads from T-cell `by_donor`; use `by_context`.
- Frontend reads directly from flat `PAYLOAD.kinases`, `PAYLOAD.celltypes`, or
  `PAYLOAD.incytr_pathways`; new shared code should use `ViewerPayload`.
- `selection.donor` and hash key `d=` as the primary routing state; `selection.context` and
  `ctx=` are canonical. `d=` is read only to preserve old inbound links.
- Hard-coded contrast vocabularies such as App/Tau/ApTt or donor day labels inside shared tabs;
  contexts should carry their own contrast axis.
- Cohort-name checks such as `PAYLOAD.meta.cohort === "tcell"` when a capability flag or context
  field can express the behavior.
- Duplicated common JS modules once both viewers read through shared adapters/capability helpers and
  validate from the v2 payload shape.

## Validation

Every builder emitting this contract must validate:

- `meta.default_context` exists in `meta.contexts`.
- Each context has unique `id`, non-empty `label`, and a contrast list.
- Every context with `capabilities.kinases = true` has a kinase slice.
- Every context with `capabilities.incytr = true` has an Incytr slice index and shard URL.
- `edge_slice_ref` URLs exist for enabled lazy-shard capabilities.
- Empty contexts include a human-readable note explaining intentional absence.
- Payload raw/gzip sizes remain within viewer limits.

For the current repo, run:

```bash
pixi run python alz/build_unified_viewer.py --payload --html --validate
pixi run python alz/build_tcell_viewer.py --payload --html --validate
pixi run python alz/viewer/verify_payload_contract.py \
  outputs/reports/unified_viewer/unified_viewer.payload.json \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json
```

The expected contract-verifier result after the 2026-06-01 cleanup is:

```text
outputs/reports/unified_viewer/unified_viewer.payload.json: schema=2 default=song_ad contexts=song_ad pass=True
outputs/reports/tcell_viewer/tcell_viewer.payload.json: schema=2 default=donor1 contexts=donor1,donor2 pass=True
```
