# Unified viewer: candidate-pathway tabs implementation plan

**Status:** rewritten 2026-05-12 — significance filter dropped; viewer now surfaces all candidates with tiered heatmap.
**Owner:** Henri
**Scope:** add two tabs to `outputs/reports/unified_viewer/index.html` that surface Incytr factorial candidate pathways from the long-form `receiver_cache/`, with client-side gating.

## Why surface all candidates

Earlier drafts gated build-time on Fig-2D thresholds (`pvalue<0.05 ∧ |PDS|>0.76 ∧ sigprob_max>0.10`, raw p). With the upstream SigProb fix landed, that gate keeps only **187 paths / 1,683 rows** of the 36,003 unique candidate paths / 1,216,854 long-form rows. The cohort is small (1–2 animals per condition cell) so the per-row p-value family is sparse and the gate over-prunes useful candidates. Decision: ship every row, pre-compute counts at four gates so the heatmap stays interpretable, and let the table filter chips threshold live.

## Data layout (run 20260512)

- `data/incytr_factorial_outputs/receiver_cache/receiver=<sanitized_celltype>/data.parquet` — hive-partitioned long-form path scores. 22 partitions, 1,216,854 rows, 37 cols, one row per `(sender, receiver, Path, contrast)`. **11 MB total on disk.**
- 4 seed-list label columns (`Ligand.label`, `Receptor.label`, `EM.label`, `Target.label` ∈ `{"DEG","prG"}`) are first-class.
- 9 contrasts, each contributing exactly 135,206 rows: 3 diseases × 3 timepoints (App / Tau / ApTt × 2mo / 4mo / 6mo).
- 349 of 484 (sender, receiver) pairs have rows; the 135 absent ones involve at least one of 5 empty-DEG cell types (Chandelier, L6b, Lamp5 Lhx6, Sncg, Sst Chodl).
- Sender column is unsanitized; receiver comes from the hive partition (sanitized via `gsub("/", "-", gsub(" ", "_", x))`).

`alz/integration/filter_significant_pathways.py` still exists as a standalone downstream tool for offline analysis, but is **no longer an input to the viewer build**.

## Locked decisions

| Decision | Choice |
|---|---|
| Source | `receiver_cache/receiver=*/data.parquet` directly — no build-time filter. |
| Slice form | **Long-form preserved** — one row per `(Path, contrast)`. Table tab filters down to one contrast for display. |
| Heatmap layout | Single 22×22 heatmap with a **gate-tier picker** (`all` / `p05` / `paper` / `strict`) + a contrast picker. Empty-DEG rows/cols rendered as a distinct "no data" state (hatched grey). |
| Pathway-table delivery | One parquet per `(sender, receiver)` under `edge_slices/incytr_pathways/`, fetched on demand via the existing `SliceCache`. |
| Tab-2 default state | Empty + prompt. No rows loaded until the user clicks a heatmap cell or sets a filter. |
| Threshold semantics | Counts at four tiers baked into the payload (read-only). Per-row metrics in shards (pvalue, PDS, log2FC, sigprob_max) — UI can apply finer sliders client-side without rebuilding. |

## Inputs

- `data/incytr_factorial_outputs/receiver_cache/receiver=*/data.parquet` — long-form factorial output, all candidates (post Phase-0 SigProb fix; all 9 contrasts populated except the structurally-absent `ApTt_4mo`).
- `data/incytr_factorial_outputs/pair_metadata.parquet` — canonical 22-cell-type list for sender↔receiver name normalization (sender raw, receiver sanitized).

## Outputs added

```
outputs/reports/unified_viewer/edge_slices/incytr_pathways/
  index.json
  <sender>__<receiver>.parquet     × 349 shards
```

Per-build observed:
- Total shard footprint: **6.1 MB** across 349 shards. Heaviest shard `OPC__Pvalb.parquet` at 0.09 MB.
- Payload bump: ~50 KB (4 tier-count grids + slice index).
- Build wall-time: ~10 s (DuckDB + pandas groupby).

## Payload block

```jsonc
{
  "incytr_pathways": {
    "schema_version": 1,
    "source": "receiver_cache/ (unfiltered)",
    "contrasts": ["App_2mo", ..., "ApTt_6mo"],
    "senders":   [/* 22, sorted */],
    "receivers": [/* 22, sorted */],
    "empty_deg_celltypes": ["Chandelier", "L6b", "Lamp5 Lhx6", "Sncg", "Sst Chodl"],
    "heatmap_tiers": {
      "all":    {"label": "all candidate paths",          "gate": {},                                "counts": [/* uint32 22*22*9 */], "total": 1216854},
      "p05":    {"label": "raw pvalue < 0.05",             "gate": {"p": 0.05},                       "counts": [...],                  "total":    5215},
      "paper":  {"label": "He 2025 Fig 2D (raw p)",        "gate": {"p": 0.05, "pds": 0.76, "sp": 0.10}, "counts": [...],               "total":     292},
      "strict": {"label": "paper gate at p<0.01",          "gate": {"p": 0.01, "pds": 0.76, "sp": 0.10}, "counts": [...],               "total":      61}
    },
    "default_tier": "paper",
    "slice_index": {
      "filename_template": "{sender}__{receiver}.parquet",
      "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
      "present": [["Astrocyte","Microglia"], /* …349 pairs */],
      "n_total_rows": 1216854,
      "pair_row_counts": {/* filename → row count */}
    }
  }
}
```

`empty_deg_celltypes` lets the heatmap distinguish "structurally zero" from "below threshold." `heatmap_tiers[*].counts` order: sender × receiver × contrast (row-major).

## Slice schema: `<sender>__<receiver>.parquet`

Long-form, 14 columns, zstd, dictionary-encoded strings, float32 metrics. No `signature` / `is_significant` / `n_contrasts_sig` columns — those collapse threshold choice into the file. UI computes them live if needed.

| Group | Columns |
|---|---|
| identity (5) | `Path`, `Ligand`, `Receptor`, `EM`, `Target` |
| seed-list labels (4) | `Ligand.label`, `Receptor.label`, `EM.label`, `Target.label` |
| contrast (1) | `contrast` (9-level dictionary) |
| metrics (4) | `pvalue`, `PDS`, `log2FC`, `sigprob_max` |

Sorted by `(contrast, pvalue)` for cheap first-render of "most significant at top."

## Build-script change (`alz/build_unified_viewer.py`)

`_write_incytr_pathways()` (one helper, ~110 lines):

- DuckDB reads the hive-partitioned `receiver_cache/`, computes `sigprob_max = GREATEST(SigProb_ref, SigProb_alt)` once.
- Tier loop: 4 SQL aggregates (`COUNT(*) WHERE <gate> GROUP BY sender, receiver, contrast`), reshaped to 22×22×9 grids. Pairs that involve empty-DEG cell types stay zero.
- Materialize all rows once (~1.2M, ~200 MB RAM), groupby `(sender, receiver)`, write one parquet per pair via `pyarrow.parquet.write_table`.
- Senders are raw; receivers are sanitized in the source. Both are mapped to display names via `pair_metadata.parquet`'s canonical 22 cell types before shard write.
- Returns the assembled payload block.

Wired into `build_payload(data)` next to the existing `_write_decomp_ols_slices(...)` call. Skip-on-missing-input is the only feature flag.

## Tabs (Phase 2 + 3)

**Heatmap tab** (`tabs/incytr_heatmap.js`):
- Single Plotly heatmap.
- Two pickers: gate tier (`all` / `p05` / `paper` / `strict`, default `paper`) and contrast (9 radios, default `App_4mo`).
- Cell value = `heatmap_tiers[tier].counts[sender_idx*22*9 + receiver_idx*9 + contrast_idx]`.
- Zero-anchored color scale derived from `heatmap_tiers[tier].counts.max()`.
- Empty-DEG rows/cols hatched grey, separate from "0 at this tier."
- Click → switches to table tab, applies `(sender, receiver, contrast)` filters, and (if tier ≠ `all`) seeds the row-level metric sliders to the tier gate.

**Table tab** (`tabs/incytr_pathways.js`):
- Virtualized table over existing `data-table` CSS.
- Filter chips: `sender`, `receiver`, `contrast`, 4 label chips (`Ligand.label = DEG/prG`, etc).
- Metric sliders: `pvalue ≤`, `|PDS| ≥`, `sigprob_max ≥`. Default = paper gate.
- Default columns: identity + labels + `pvalue`, `PDS`, `log2FC`, `sigprob_max`. Sort = `pvalue` asc.
- "Show all 9 contrasts for this path" toggle: removes the contrast filter and groups by `Path`.

## Phasing

0. ~~Upstream SigProb fix~~ — landed (run 20260512). All 9 contrasts populated except structurally-absent `ApTt_4mo`.
1. **Data plane** — `_write_incytr_pathways` rewritten to source from `receiver_cache/`, tiered heatmap counts in payload, full-candidate shards. **DONE**.
2. **Heatmap tab** — render + tier picker + contrast picker. No drill-in yet.
3. **Table tab + drill-in** — virtualized table, filter chips, metric sliders, heatmap-click handler.

Each phase ends with `pixi run viewer` + hard refresh.

## Acceptance

Phase 1 (current):
- `pixi run viewer` prints `incytr_pathways: wrote 349 shards (1,216,854 rows; ~6 MB total)`.
- `index.json:n_total_rows == 1,216,854`.
- `heatmap_tiers.all.total == 1,216,854`. `heatmap_tiers.paper.total ≈ 292` (matches old filter output).
- Heaviest shard ≤ 5 MB; total shard footprint ≤ 130 MB. Spot-check: observed max 0.09 MB, total 6.1 MB — well under budget.
- Payload diff ≤ 100 KB.

Phase 2/3:
- Switching tier from `paper` → `all` lights up the full 22×22 grid.
- Heatmap click on `(L2/3 IT, VLMC)` at any contrast switches to the table tab with shard loaded; row count matches `pair_row_counts["L2-3_IT__VLMC.parquet"]` filtered to selected contrast.
- Empty-DEG cells render hatched grey with tooltip explaining the gating.
- Default sliders match the `paper` gate; moving them updates row count immediately.

## Risks

- **Empty-DEG cells under `all` tier.** Cells involving Chandelier / L6b / Lamp5 Lhx6 / Sncg / Sst Chodl are still zero because no candidate paths were generated for those cell types upstream — not a UI bug. `empty_deg_celltypes` payload field flags them so the heatmap renders them distinctly.
- **Strict tier may be empty for some contrasts.** Observed: 61 rows total at `paper p<0.01`. Heatmap stays informative because `all` and `p05` tiers always populate; user can step down through tiers to find signal.
- **Client-side slider performance.** Heaviest pair shard is 4,124 paths × 9 contrasts = 37,116 rows. JS filtering at that size is tens of ms — acceptable.

## Out of scope

- Editing tier definitions in-browser (the 4 tiers are baked at build time; user can refine within a tier via sliders).
- Cross-pathway gene-level views or kinase ↔ pathway joins.
- BH-adjusted q-values (deliberately omitted per data/incytr_factorial_outputs/README.md "Why raw p, not BH").

## Updating the pathway data

```bash
pixi run incytr-factorial   # regenerates receiver_cache/ (post-Phase-0 SigProb)
pixi run viewer             # rebuilds shards + payload (auto-sources receiver_cache/)
```

Changing tier definitions requires editing `_INCYTR_HEATMAP_TIERS` at the top of `alz/build_unified_viewer.py` and re-running `pixi run viewer`.
