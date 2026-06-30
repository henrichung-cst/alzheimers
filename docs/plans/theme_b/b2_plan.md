# B2 — Backbone sankey (per-genotype)

> **STATUS: DEFERRED / STALE (2026-06-29).** This plan is built on `backbone_rem_t.parquet` and the
> `backbone_rank` / `is_cholinergic_target` / `n_timepoints_present` schema — **all removed** by the
> backbone grain pivot. The recurrence-reduction it consumes no longer exists. The merged Incytr
> screen + grain selector + within-disease timepoint filter + the B-6/J-5 drill may make a separate
> recurrence-ranked sankey redundant. **Do not implement from this spec as written.** The fate
> (drop vs. rebuild on the engine `*_backbone_output.parquet` grains) is parked — see
> [`backbone_incytr_track.md`](backbone_incytr_track.md) § Open threads. Everything below is the
> pre-pivot design, retained only for the deferred decision.

**Tail item** (gated after B5; `p4_dag.md` tail order B4.2 → **B2** → C3-S4 → G1). Consumes the
artifact the backbone fold landed: `outputs/reports/incytr_pair_mode/backbone/backbone_rem_t.parquet`.
**Contract:** `_contracts.md §B5`. **No regen / no heavy compute** — the reduction is already
materialized; B2 is a build-step aggregation + a new unified-viewer tab.

---

## Audit — what exists

- **Artifact** (`backbone_rem_t.parquet`, 2,782,293 rows × 12 cols, 47 MB): one row per R-EM-T
  key-tuple `(Sender.group, Receiver.group, Receptor, EM, Target)` with `PDS` (representative,
  signed = max-`|PDS|` occurrence), `n_timepoints_present`, `n_conditions_present`, `backbone_rank`
  (global, 1 = top), `is_cholinergic_target` (bool, `Receiver.group == 'Cholinergic-Neurons'`),
  `conditions_present` (comma-joined genotypes, e.g. `ApTt,App,Tau`), `contrasts_present`
  (comma-joined genotype_timepoint).
- **Rendering infra is already vendored** — `head.html` loads Plotly 2.35.0 (CDN). Plotly has a
  native `type:"sankey"`. Several tabs already call `Plotly` (temporal_v2, kinase_audit,
  kinase_fivexfad, kinase_human). **No new dependency, no hand-rolled SVG.**
- **Tab registration is declarative** (`MANIFEST.md §Adding a new tab`): drop `js/tabs/<x>.js`
  exporting `wire<X>()`+`render<X>()`, add a `{{ raw(...) }}` include to `index.html.j2`, a
  `<div id="tab-<x>" class="tab-panel" hidden>` to `body.html`, and one `TAB_MANIFEST` entry in
  `02_ui_chrome.js`. The tab bar + re-render dispatch are generated from the manifest — no
  `boot.js` branch.
- **Genotype axis** is `App / Tau / ApTt` (the contrast groups; `incytr_pathways.js` uses exactly
  this set). A per-genotype toolbar `<select>` mirrors the existing incytr tissue selector.
- **Payload sizing precedent**: large per-entity data is written as on-demand **slice files**
  (`_write_decomp_ols_slices`, `_write_song_concordance_slices` in `song.py`; loaded via
  `04_slice_cache.js`). The main payload `.gz` (~6 MB) carries only small/columnar slices.

## The sizing decision (memory-safe, the crux)

2.78M rows cannot ship to the browser, and a 2.78M-edge sankey is meaningless — a sankey is
legible at ~10² flows. So **B2 pre-aggregates at build time into a tiny per-genotype top-N slice**:

- Aggregation runs in `song.py` via **DuckDB streaming over the parquet** (never
  `pd.read_parquet` — the row×col bill is multi-GB decompressed; CLAUDE.md memory rule). The query
  is bounded: per genotype `g ∈ {App,Tau,ApTt}`, filter rows whose `conditions_present` contains
  `g`, order by `backbone_rank`, take the top **N = 200**, project the 5 spine columns + `PDS` +
  `is_cholinergic_target` + the two recurrence counts. Result ≈ 600 rows total → inlined into the
  payload (no slice file needed; it is smaller than one existing decomp slice).
- The slice is **pre-shaped into Plotly sankey form per genotype**: a deduplicated `nodes[]`
  (label + stage index, stages = Sender.group, Receiver.group, Receptor, EM, Target) and `links[]`
  (`{source, target, value, pds, recurrence}`) built once in Python, so the JS tab is a thin
  Plotly call. `value` = link multiplicity within the top-N (number of backbones traversing that
  edge); color carries the signed PDS.

This keeps the box safe (bounded DuckDB scalararray, capped run), keeps the payload small, and
makes the sankey readable.

## Design

**Stages (5):** `Sender.group → Receiver.group → Receptor → EM → Target`. This is the Incytr
molecular spine (Receptor→EM→Target) bracketed by the two cell-type groups, which is what the
R-EM-T key encodes. Cell-type labels use `COHORT_DISPLAY` (C2's naming contract).

**Per-genotype:** toolbar `<select>` (App / Tau / ApTt), default App. Switching re-renders the
sankey from the already-loaded slice (no refetch).

**Color = signed PDS** (the established trajectory convention): red = PDS > 0 (up), blue = PDS < 0
(down), magnitude → saturation. Reuses the `_IP_TRAJ_COLORS` palette direction so it reads
consistently with the pathways tab.

**Cholinergic anchor — flag, never filter** (`_contracts.md §B5`): `is_cholinergic_target` links
get a highlighted node color / a "pin Cholinergic" toggle that dims non-Cholinergic flows. The
reducer dropped no rows and neither does B2 — the anchor is display-only.

**Recurrence on hover:** node/link tooltips surface `n_timepoints_present` / `n_conditions_present`
and `backbone_rank` so the recurrence signal (the whole point of the backbone reduction vs the raw
per-pair enumeration) is visible.

**No specificity / cell-count join in v1.** `_contracts.md §B5` flags `mean_gene_specificity` /
`min_cell_count` as an *open divergence* — not shipped by the reducer, "compute in B2 if needed."
The first sankey ranks on `backbone_rank` (recurrence) + colors on PDS; it does not need them.
Deferring keeps B2 to the documented artifact and avoids a second join. If the sankey later needs
interactive specificity filtering, add it then (annotate-not-filter, per the contract).

## Implementation

1. **`song.py`** — `_build_backbone_sankey_slice()`: DuckDB query over `backbone_rem_t.parquet`
   (memory-capped, top-200/genotype), shape nodes/links per genotype, return
   `{"by_genotype": {"App": {...}, "Tau": {...}, "ApTt": {...}}}`. Attach under a new payload key
   `PAYLOAD.backbone_sankey`. Skip cleanly (key absent) if the parquet is missing — the tab's
   `requires` gates it.
2. **`js/tabs/backbone_sankey.js`** — `wireBackboneSankey()` (genotype select + Cholinergic-pin
   toggle) + `renderBackboneSankey()` (one `Plotly.react` sankey call from the active genotype's
   pre-shaped nodes/links). New file in `alz/viewer/template/js/tabs/`.
3. **`body.html`** — `<div id="tab-backbonesankey" class="tab-panel" hidden>` with toolbar + a
   `<div id="bs-plot">` Plotly mount.
4. **`index.html.j2`** — `{{ raw('js/tabs/backbone_sankey.js') }}` include.
5. **`02_ui_chrome.js`** — `TAB_MANIFEST.backbonesankey` (`group:"landscape"`, `label:"Backbone
   Sankey"`, `modes:["mouse"]`, `requires:[{type:"payload", key:"backbone_sankey", …}]`,
   `filters:[]`, `rerenderOn:{filters:false, selection:[]}`).
6. **`01_state.js`** — `TAB_GUIDE.backbonesankey` (what a backbone is, the recurrence ranking, the
   canonical floor it inherits, the Cholinergic anchor, top-200 cap stated honestly).
7. **`MANIFEST.md`** — add the tab row.
8. **`pixi run viewer`** (under `systemd-run … MemoryMax=24G`), verify payload key + DOM, then the
   **human browser click-through** (authoritative): sankey renders for each genotype, PDS colors
   read correctly, Cholinergic pin works, hover shows recurrence.

## Out of scope
The recurrence reduction itself (done — fold); 5xFAD/t-cell sankeys (the artifact is song-only —
the fold's `reduce()` runs on the song branch); any specificity/cell-count filter; re-gating beyond
the canonical floor; G1 docs.

## Open decisions (for approval before implementation)
- **N per genotype = 200.** Trades sankey legibility against coverage. State the cap in the guide
  (honesty-over-polish). Acceptable, or different N?
- **Rank basis = `backbone_rank`** (recurrence-driven, the contract's primary) rather than `|PDS|`.
  Recurrence is the backbone reduction's reason for existing; PDS is the color. Agree?
- **5 stages** (Sender.group→Receiver.group→Receptor→EM→Target) vs collapsing the two cell-type
  groups into one "cell context" stage. 5-stage keeps sender/receiver distinct (matters for
  cross-type signaling); recommend 5-stage.
