# Plan: Standardized CSV export across all unified-viewer tables

## 1. Current state — table inventory and export status

### Tabs in the unified viewer (TAB_MANIFEST in `alz/viewer/template/js/02_ui_chrome.js`)

| Tab key | Label | HTML table element | Has CSV export today? | Has Markdown export today? |
|---|---|---|---|---|
| `kinase` | Kinase (Mouse) | `#ke-table` (`<table>`) | No | Yes — `wireExportButtons()` injects "Export view" (Markdown) into `#tab-kinase .ke-toolbar`. Download is `.md`, not CSV. |
| `kinasehuman` | Kinase (Human) | `#kh-table` (`<table>`) | No | No |
| `temporalv2` | Temporal v2 | No tabular data — SVG/canvas bar charts only | N/A | No |
| `crosstable` | Crosstable | `#kx-table` (injected into `#kx-table-wrap`) | No | No |
| `fivexfadkinase` | 5xFAD Kinase | `#f5-table` (`<table>`) | No | No |
| `incytrheatmap` | Incytr Heatmap | No tabular data — Plotly heatmap | N/A | No |
| `incytrpathways` | Incytr Pathways | `#ip-table-wrap` (rendered by `incytr_pathways.js`) | **Yes** — `#ip-export` button with full client-side CSV, reflects active filters | No |
| `methods` | Methods | Static iframe — `pipeline_overview.html` | N/A | No |

Audit tables (in the kinase detail / audit panel, rendered by `AuditTable` class in `alz/viewer/template/js/01_state.js`): already have "Export filtered" and "Export full source" CSV buttons per-table instance. These are internal to the drilldown panel; they are already correct CSV and do not need to change.

**Tables needing CSV export added:** `kinase` (Mouse), `kinasehuman`, `crosstable`, `fivexfadkinase`.
**Incytr Pathways:** already emits CSV; needs alignment to the shared utility.
**Temporal v2, Incytr Heatmap, Methods:** no table, no export needed.

---

## 2. Existing partial implementation — how it works

### Incytr Pathways (`alz/viewer_shared/template/js/tabs/incytr_pathways.js`)

The only tab with real CSV today. Three private helpers:

```js
function _ipCsvEscape(v) { ... }           // RFC-4180: wrap in "" if v contains , " \r \n
function _ipCsvColumns(rows) { ... }       // Dynamic column list: fixed set + conditional score/rank/traj cols
function _ipRowsToCsv(rows) { ... }        // Header row + data rows, \n-joined, trailing \n
function _ipDownloadCsv(rows, filename) { ... }  // Blob → anchor click → revokeObjectURL
async function _ipExportCurrentView() { ... }    // Resolves rows for "top" or "pair" mode, then calls _ipDownloadCsv
```

The export button `#ip-export` is in `alz/viewer/template/body.html:366`. It triggers `_ipExportCurrentView()`. Rows reflect **current filter state** (not raw), with all columns present in the data object.

### Markdown "Export view" (`alz/viewer_shared/template/js/02_ui_chrome_common.js`)

`wireExportButtons()` injects a button only for the `kinase` tab (line 237). It reads the DOM via `_exportTableFromDom("ke-table")` — strips sort-arrow suffixes from `<th>` text, reads `<td>` `.textContent`. Output is Markdown, not CSV. Format is `.md`.

This is a chatbot-paste helper, not a data export. It emits visible-text content, including rendered HTML badges stripped to their text. **It is not a substitute for CSV export** — the Markdown pipeline does not produce clean data.

### AuditTable (in `alz/viewer/template/js/01_state.js`)

`AuditTable.exportRows(rows, cleanHeaders)` + `AuditTable.downloadCsv(rows, label, cleanHeaders)`:
- RFC-4180 quoting inline: `,` `"` `\n` trigger quoting
- Headers: raw column names by default; with `cleanHeaders=true`, uses `_auditColMeta()` display labels
- Always exports the **filtered** set (search + sort, not paginated)
- Filename: `${tableKey}_filtered.csv`

This is a good local precedent for the shared utility.

---

## 3. Shared utility design

### Location

Add a new file: `alz/viewer_shared/template/js/06_export_csv.js`

Include it in `alz/viewer_shared/template/index.html.j2` after `05_header.js`, before the tab files (line 17 position). The `_raw()` loader picks it from `viewer_shared` automatically.

### Signature

```js
/**
 * Escape one cell value for RFC-4180 CSV.
 * null/undefined → ""; non-finite numbers → ""; wrap in "" if value contains , " \r \n
 */
function csvEscape(v) { ... }

/**
 * Serialize rows to CSV string.
 * @param {string[]} headers   - Column display names for the header row.
 * @param {string[]} keys      - Parallel array of data-object keys to read per row.
 * @param {Object[]} rows      - Data rows (plain JS objects).
 * @returns {string}           - Complete CSV text, UTF-8, trailing newline.
 */
function csvSerialize(headers, keys, rows) { ... }

/**
 * Trigger a browser file download of a CSV blob.
 * @param {string} csv         - Output of csvSerialize().
 * @param {string} filename    - Filename including .csv extension.
 */
function csvDownload(csv, filename) { ... }
```

`csvEscape` is identical in logic to `_ipCsvEscape` (incytr_pathways) and to the inline lambda in `AuditTable.exportRows`. Unify here; the others call this one.

### CSV rules
- RFC-4180 quoting: if the stringified value contains `,`, `"`, `\r`, or `\n`, wrap in `""` and double any internal `"`.
- `null`, `undefined`, non-finite `Number` → empty string.
- Header row uses the **display names** (same as the `<th>` labels in the HTML, not raw object keys), so the exported file is self-explanatory.
- All rows reflect the **current filtered + sorted state** (matching what the user sees), not the full unfiltered dataset. Filename encodes the tab key and ISO date so downloads don't collide.
- Filename convention: `{tab_key}_{YYYY-MM-DD}.csv` — e.g. `kinase_2026-06-18.csv`, `kinase_human_2026-06-18.csv`, `crosstable_2026-06-18.csv`, `fivexfad_kinase_2026-06-18.csv`.

---

## 4. Per-table wiring

### 4a. Kinase (Mouse) — `kinase_explorer.js`

**Data access:** `renderKinaseExplorer()` builds the `visible` array (filtered + sorted JS objects in `_keRows`). Expose a module-level reference `_keVisible` that is set at the end of the render loop before `tbody.innerHTML = parts.join("")`. This avoids re-running the filter logic.

**Columns to export** (maps `<th data-col>` → object keys already present on each `_keRows` item):

| Header | Key |
|---|---|
| Kinase | `name` |
| Gene | `gene_symbol` |
| Family | `family` |
| Residue | `residue_type` |
| n_sig | computed `scopedSig` (store on row during render, or recompute in export) |
| peak_NES | `peakAbsNes` (recompute in export pass, same as render pass) |
| song_topShare | `song.topShare` |
| song_topCell | `song.topCell` |
| wmb_max_tier | from `_kineMaxWmbTierScoped` |
| conf | highest tier string |

The NES profile and agreement profile columns are SVG/visual-only; omit from CSV (they carry no text data that isn't encoded in n_sig + peak_NES).

**Wiring:** Add `function exportKinaseCsv()` at the bottom of `kinase_explorer.js`. Call `csvSerialize(headers, keys, _keVisible)`. Wire in `wireKinaseTable()`:
```js
const btn = document.getElementById("ke-export");
if (btn) btn.addEventListener("click", exportKinaseCsv);
```

**HTML:** Add export button to `#ke-unified-toolbar` in `body.html`, after `#ke-filter-reset`:
```html
<button id="ke-export" class="ke-filter-reset" title="Export current kinase table as CSV">Export CSV</button>
```

### 4b. Kinase (Human) — `kinase_human.js`

**Data access:** `renderKinaseHuman()` builds `rowsAll = _khFilter(_khAllRows())` then `_khSort(rowsAll)`. Store the sorted visible array in `_khVisible` at the point where the table is written.

**Columns:**

| Header | Key |
|---|---|
| Kinase | `name` |
| Gene | `gene_symbol` |
| Family | `family` |
| Residue | `residue_type` |
| median_NES_sig | `median_nes_sig_only` |
| n_donors_sig | `n_donors_sig` |
| n_up | `n_donors_up` |
| n_down | `n_donors_down` |
| n_ctrl_sig | `n_ctrl_sig` |
| conf | resolved from `_khAttributionSummary(r).conf` |
| location_tier | resolved from `_khAttributionSummary(r).maxTierRank` |
| n_cell_types | resolved from `_khAttributionSummary(r).count` |

NES profile strip is visual-only; omit.

**Wiring:** Add `function exportKinaseHumanCsv()` at bottom of `kinase_human.js`. Wire in `wireKinaseHuman()`:
```js
const btn = document.getElementById("kh-export");
if (btn) btn.addEventListener("click", exportKinaseHumanCsv);
```

**HTML:** Add to `#kh-toolbar` in `body.html`, after `#kh-filter-reset`:
```html
<button id="kh-export" class="ke-filter-reset" title="Export current human kinase table as CSV">Export CSV</button>
```

### 4c. Crosstable — `kinase_crosstable.js`

**Data access:** `_kxRenderTable()` calls `_kxFilteredRows(s)` → `_kxComputeAgreement(rows, ...)` → filters by comparison-state → `_kxSortRows(shown, s)`. Store `shown` in `_kxVisible` after sort.

**Columns:** The crosstable rows are JS objects with keys `name`, `gene`, `residue`, `family`, `_mNes` (mouse median NES), `_hNes` (human median NES), `_f5Nes` (5xFAD aggregate NES), `_agreeCategory` (string), plus resolved `song`, `seaad`, `wmb` from `_kxResolveSpec`. The visual glyph cells (colored arrows) collapse to their text label equivalents.

| Header | Key / computation |
|---|---|
| Kinase | `name` |
| Gene | `gene` |
| Residue | `residue` |
| Family | `family` |
| Mouse_med_NES | `_mNes` |
| Human_med_NES | `_hNes` |
| 5xFAD_med_NES | `_f5Nes` |
| Crossplay | `_agreeCategory` |
| Song_fold | resolved `sp.song.topShare / _KX_SONG_UNIFORM` (or null) |
| WMB_tier | resolved `sp.wmb` |
| SEAAD_log2 | resolved `sp.seaad` |

**Wiring:** Add `function exportCrosstableCsv()` at bottom of `kinase_crosstable.js`. Wire in `wireKinaseCrosstable()`:
```js
const btn = document.getElementById("kx-export");
if (btn) btn.addEventListener("click", exportCrosstableCsv);
```

**HTML:** Add to `#kx-toolbar` in `body.html`, after `#kx-reset`:
```html
<button id="kx-export" class="ke-filter-reset" title="Export current crosstable as CSV">Export CSV</button>
```

### 4d. 5xFAD Kinase — `kinase_fivexfad.js`

**Data access:** `renderFiveXFADKinase()` calls `_f5FilteredRows()` which returns sorted rows. Store return value in `_f5Visible`.

**Columns:**

| Header | Key |
|---|---|
| Kinase | `kinase` |
| Gene | `gene_symbol` |
| Family | `family` |
| Tissue | `tissue` |
| Residue | `residue_type` |
| n_sig | `sigCount` |
| peak_NES | `peakAbsNes` |
| 5xFAD_snrna | from `_f5BestAttr(r).fivexfad_concentration` or null |
| WMB_tier | from `_f5BestAttr(r)` WMB tier |
| Conf | from `_f5BestAttr(r).confidence_tier` or null |

NES profile and agreement profile are visual-only; omit.

**Wiring:** Add `function exportFiveXFADCsv()` at bottom of `kinase_fivexfad.js`. Wire in `wireFiveXFADKinase()`:
```js
const btn = document.getElementById("f5-export");
if (btn) btn.addEventListener("click", exportFiveXFADCsv);
```

**HTML:** Add to `#f5-toolbar` in `body.html`, after `#f5-filter-reset`:
```html
<button id="f5-export" class="ke-filter-reset" title="Export current 5xFAD kinase table as CSV">Export CSV</button>
```

### 4e. Incytr Pathways — align to shared utility

The existing `_ipCsvEscape` / `_ipDownloadCsv` / `_ipRowsToCsv` in `incytr_pathways.js` are private. **Replace** all three with calls to `csvEscape`, `csvSerialize`, `csvDownload` from `06_export_csv.js`. The column set (`_ipCsvColumns`) and data-object shape stay unchanged. This collapses ~30 lines of private implementation into calls to the shared helpers.

`_ipExportCurrentView()` calls `_ipDownloadCsv(rows, filename)` which should become `csvDownload(csvSerialize(headers, keys, rows), filename)`. Because Incytr pathways row objects carry string-keyed values directly (not computed properties), `headers` = the column display names and `keys` = the same column identifiers — same as what `_ipCsvColumns()` returns.

---

## 5. Preserving the PAYLOAD / Store / TAB_MANIFEST contract

- No new PAYLOAD keys. No TAB_MANIFEST changes. No Store state changes.
- `06_export_csv.js` is pure utilities (three functions, no globals).
- Each per-tab export function is added to the same file that already owns that tab's render logic; the wiring call is added to the existing `wireXxx()` function.
- The Markdown "Export view" button that currently exists on the `kinase` tab is **not removed** — it serves a different purpose (chatbot pasting). The new CSV button is additive, placed in the same toolbar.
- `AuditTable.exportRows` / `AuditTable.downloadCsv` in `01_state.js` could optionally delegate to `csvEscape` + `csvDownload` (one-line change each), but since they work correctly today and their code is stable, leave them as-is to minimize blast radius.
- No changes to `build_unified_viewer.py` except adding `js/06_export_csv.js` to the `index.html.j2` include order (done by adding a `raw()` call in the shared template, not in the Python builder).

---

## 6. File change summary

| File | Change |
|---|---|
| `alz/viewer_shared/template/js/06_export_csv.js` | **New** — `csvEscape`, `csvSerialize`, `csvDownload` |
| `alz/viewer_shared/template/index.html.j2` | Add `{{ raw('js/06_export_csv.js') }}` after line 16 (`05_header.js`) |
| `alz/viewer/template/body.html` | Add `<button id="ke-export">`, `<button id="kh-export">`, `<button id="kx-export">`, `<button id="f5-export">` in their respective toolbars |
| `alz/viewer/template/js/tabs/kinase_explorer.js` | Add `_keVisible`, `exportKinaseCsv()`, wire in `wireKinaseTable()` |
| `alz/viewer/template/js/tabs/kinase_human.js` | Add `_khVisible`, `exportKinaseHumanCsv()`, wire in `wireKinaseHuman()` |
| `alz/viewer/template/js/tabs/kinase_crosstable.js` | Add `_kxVisible`, `exportCrosstableCsv()`, wire in `wireKinaseCrosstable()` |
| `alz/viewer/template/js/tabs/kinase_fivexfad.js` | Add `_f5Visible`, `exportFiveXFADCsv()`, wire in `wireFiveXFADKinase()` |
| `alz/viewer_shared/template/js/tabs/incytr_pathways.js` | Replace `_ipCsvEscape` / `_ipRowsToCsv` / `_ipDownloadCsv` bodies with calls to shared utility |

Total: 1 new file, 7 edits. No Python changes, no payload schema changes.

---

## 7. Verification

Build and inspect:

```bash
pixi run viewer                # or: python alz/build_unified_viewer.py
```

Then in a browser (hard-refresh Ctrl+Shift+R to bypass cached HTML):

1. **Kinase tab** — set a filter (e.g. Disease = App), click "Export CSV". Open the file: confirm header row matches the column list above, confirm row count matches the count shown in `#ke-count`, confirm no HTML entities or badge markup appears in cells.
2. **Human Kinase tab** — set n_sig ≥ 2, click "Export CSV". Confirm `n_donors_sig` column values are all ≥ 2.
3. **Crosstable tab** — set Compare = Song vs Mukesh, click "Export CSV". Confirm `Crossplay` column is populated (not empty).
4. **5xFAD Kinase tab** — set Tissue = cortex, click "Export CSV". Confirm all rows have `tissue = cortex`.
5. **Incytr Pathways tab** — set |PDS| ≥ 0.5, click "Export CSV". Confirm row count matches `#ip-count`. Open CSV and spot-check that `PDS` column values are all ≥ 0.5 or ≤ -0.5.
6. **Temporal v2, Incytr Heatmap, Methods** — confirm no export button appears (none was added).

For each exported file: `python3 -c "import csv,sys; list(csv.DictReader(open(sys.argv[1])))" <file.csv>` to confirm valid RFC-4180 parse without error.
