# Incytr viewer — current schema (baseline, pre-refactor)

Ground-truth audit of the two shared Incytr tabs as they exist today. Source of truth, not
recollection. Both tabs live in **`alz/viewer_shared/template/js/tabs/`** (shared by the unified
*and* t-cell viewers — edits propagate to both). State is shared between them via `IncytrFilter`;
a heatmap-cell click seeds the table's filters and switches tabs.

Data contract: `ViewerPayload.incytr()` → one `block`. Score-column set is locked in
`alz/viewer/shared/incytr_index.py` (`_INCYTR_SCORE_COLS`) and advertised per-block as
`block.score_columns` (JS falls back to the 5-col default).

---

## 1. Pathways tab (`incytr_pathways.js`) — table

### Columns (the actual `cols` array, in order)
| col | notes |
|---|---|
| Sender | `_sender` — WMB cell-type class emitting the ligand |
| Receiver | `_receiver` |
| Path | redundant: `Ligand\|Receptor\|EM\|Target` joined |
| **Ligand** | + DEG/prG evidence badge (`Ligand_label`) |
| **Receptor** | + badge |
| **EM** | + badge — *(omitted in the recollection; it is a real column — the path is 4-node L-R-**EM**-T)* |
| **Target** | + badge |
| contrast | `<disease>_<timepoint>` per row (e.g. `App_4mo`) — one row **per contrast** |
| pvalue | Wald t-test on the contrast coeff (sci notation) |
| PDS | composite Pathway Disturbance Score, colored ↑red / ↓blue |
| TPDS, PPDS, PhPDS_ps, PhPDS_py, SiK_score | the 5 `_INCYTR_SCORE_COLS` |
| trajectory | badges; only when payload `version >= 3` |

### Evidence badge (DEG / prG) — the per-node provenance the user flagged
Each of Ligand/Receptor/EM/Target carries a `<node>_label` ∈ {**DEG**, **prG**}:
- **DEG** (blue) — single-cell differentially-expressed gene (transcript evidence)
- **prG** (green) — proteomics-significant gene (bulk-protein evidence)

Vocab fixed at `_INCYTR_LABEL_VOCAB = ("DEG", "prG")`.

### Two view modes (`ip-mode`)
- **Top** (`top`, default) — ranks the *whole pathway universe* via `IncytrGlobalIndex`, capped at
  `topLimit ∈ {500, 1000, 5000}`, ranked by `abs(PDS)`. Sender/Receiver/Recur multiselects are
  **hidden**; only the top-limit shows.
- **Cell Type** (`pair`) — loads one (sender,receiver) shard at a time (per-shard pagination, 100
  rows/page). Sender/Receiver/Recur multiselects show; top-limit hides.

### Filters (toolbar)
mode · top-limit (500/1000/5000) · **Sender** · **Receiver** · **Disease** · **Timepoint** ·
**Recur in** (multiselect, AND-gate: keep paths present in *all* selected disease contrasts) ·
**Trend** (`always-up / always-down / monotonic-up / monotonic-down / mixed`; gated on traj version) ·
**pvalue <** · **|PDS| ≥** · **PDS sign** (both/up/down) · **per-score |≥** gates for each of
TPDS/PPDS/PhPDS_ps/PhPDS_py/SiK · **gene search** (substring AND; pair-mode exact-symbol uses the
on-demand `gene_node_index_shard`) · **sparse-cell** toggle (low-signal include/exclude; shown only
when QC flags exist) · **Tissue** (5xFAD only) · Reset · Export CSV.

### Row detail (expander)
- **Evidence** sub-tab (default): 4 nodes × 4 omics layers via `EvidencePanel`.
- **Scores** sub-tab: per-path score trajectories across timepoints (Plotly; PDS + the 5 scores,
  line/bar/both, per-group).

---

## 2. Heatmap tab (`incytr_heatmap.js`) — sender×receiver path-count grid

Cell = candidate-path count for the active contrast; click seeds the table. Backed by a dense
`block.heatmap_counts` tensor (senders × receivers × contrasts × pvalue-thresholds × |PDS|-thresholds),
plus `heatmap_counts_signed` adding a sign axis.

### Filters
**Tissue** (5xFAD only) · **Disease** · **Timepoint** · **pvalue <** · **|PDS| ≥** · **PDS sign**
(both/positive/negative) · **axis limit** (top-N senders/receivers by burden) · **color scale**
(linear / log1p) · **sparse-cell** (low-signal) · **timeline scrubber** (steps the active axis
through its timepoints or diseases) · Reset.

Below the grid: a QC scatter (`celltype_pathway_qc`) — median n_cells vs receiver-pathway burden,
flagging median-n≤3 cell types.

---

## 3. Payload block keys (`ViewerPayload.incytr()`)
`senders` · `receivers` · `contrasts` · `diseases` · `timepoints` · `score_columns` · `label_nodes` ·
`recur_index` (pathStr → disease[]) · `slice_index.present` (present pairs) · `heatmap_counts` (+`_signed`,
`thresholds`, `abs_pds_thresholds`, `total_by_threshold`) · `gene_node_index_shard` (~15 MB, on-demand) ·
`celltype_pathway_qc` · `empty_deg_celltypes` · `version`. Per-pair rows: `SliceCache.loadIncytrShard()`.
Top mode: `IncytrGlobalIndex` (complete-universe typed-array index, `rank_by = abs(PDS)`).

---

## 4. Gaps relevant to the refactor (findings, not yet fixed)

1. **Acet/Ubiq are scored but NOT surfaced.** The R driver (`incytr_commandline.R`) scores the
   `Ack` (acetylation) and `KGG` (ubiquitination) channels end-to-end → emits `Ack_score`,
   `KGG_score` (and `Rme1_score`, methylation) columns. But `_INCYTR_SCORE_COLS` is hardcoded to the
   5 phospho/transcript/protein scores and excludes them, so they never reach the table or the
   score-filter set. Standardizing the surfaced result = extend the score-column contract to carry
   `Ack_score`/`KGG_score`(/`Rme1_score`) when present (5xFAD-only; self-gates empty for Song/t-cell).

2. **Tissue filter gating** — intended to be 5xFAD-only via `_syncFivexfadTissueToggle`
   (`show = SHOW_FIVEXFAD_INCYTR && HAS_FIVEXFAD_INCYTR && mode==="fivexfad"`, sets `wrap.hidden`).
   User reports it visible under Song; reproduce against the live build before refactoring (visual
   feedback is authoritative). Present in **both** tabs (`ip-tissue` / `ih-tissue`).

3. **Filter surface is piecemeal.** Pathways and heatmap maintain overlapping-but-divergent filter
   sets (e.g. PDS sign is `up/down` in pathways, `positive/negative` in heatmap; pvalue/|PDS| are
   free inputs in pathways, snapped-to-grid in heatmap). The refactor consolidates to one uniform
   surface.

4. **Backbone dimension is absent.** The Wave-1 `backbone_rem_t.parquet` (R-EM-T) and its sibling
   groupings (L-R-EM, R-EM) are not wired into either tab. The desired filter — *restrict to
   pathways/backbones present at all 3 timepoints, then compare PDS* — does not exist yet. The
   closest current proxies: `Trend = always-up/always-down` (requires all 3 timepoints present) and
   the `Recur in` disease-AND-gate (recurrence across *diseases*, not *timepoints*). A genuine
   **n-timepoints-present** filter at a selectable backbone grain (L-R-EM / R-EM-T / R-EM) is new.
