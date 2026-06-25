# C2 Audit — Cohort-name display surfaces

Read-only recon (2026-06-24). Complete worklist of **user-facing** surfaces emitting a cohort name. Scope per `_contracts.md §C2`: rename display layer only (`song`→`MouseC1`, `fivexfad`/`5xFAD`→`MouseC2`, `mukesh`→`HumanC1`); internal keys, payload keys, JS/DOM names, module dirs, data paths UNCHANGED. T-cell + reference atlases (SEA-AD/WMB/NSCLC) out of scope.

**Key finding:** there is **no existing `COHORT_DISPLAY` map**. Display names are scattered inline. JS display strings are baked into source (not payload-driven) and most are prose with the name mid-sentence → a runtime map is only feasible for *programmatic construction sites* (CSV export, Python context labels); the rest is a literal find-replace.

## Surface inventory

### A. `alz/viewer/template/body.html`
| Line | String | Surface |
|---|---|---|
| 5–7 | `Mouse (Song)` / `Mouse (5xFAD)` / `Human (Mukesh)` | mode-toggle buttons |
| 51–53 | `Song` label + `aria-label`/`title` on `#ke-filter-song` | kinase-explorer filter |
| 92 | `title="Song location evidence…"` | kinase table `<th>` |
| 398 | `title="Song location evidence…"` (`#kx-mspec`) | crosstable filter |
| 431–434 | `Song vs Mukesh` / `Song vs 5xFAD` / `Mukesh vs 5xFAD` / `5xFAD tissue split` | compare dropdown options |

### B. `alz/viewer/template/js/01_state.js` — TAB_GUIDE prose
Lines 266, 268, 272, 280, 293, 296 — how-to drawer passages naming Song/Mukesh/5xFAD. Literal find-replace.

### C. `alz/viewer/template/js/02_ui_chrome.js`
Line 143 `_tissueLabel()` regex `/^5xFAD\s+/i` strips the prefix off context labels to yield "Cortex"/"Hippocampus". **Must update to `/^MouseC2\s+/i`** once context labels are renamed (B-side dependency on the Python context label).

### D. `alz/viewer/template/js/tabs/kinase_crosstable.js` (heaviest)
- 90–96 `_KX_COMPARE_LABELS` dict values (`"Song vs Mukesh"` etc.)
- 572, 579, 591–592, 804–806 — column header `Song`, in-panel prose, badge tooltips
- 1003 `"Song LFC"`, 1045 `"5xFAD snRNA LFC"`, 1056 fallback msg
- 1159 `"5xFAD · cortex and hippocampus"`, 1181–1190 `TH(...)` rendered column headers + tooltips (`Mouse`/`Human`/`5xFAD cortex`/`5xFAD hip`, "Song location evidence")
- 1401 not-measured msgs, 1467–1469 verdict lines, 1480–1481 detail headings

### E. `alz/viewer/template/js/tabs/kinase_audit.js`
576 tooltip, 589–596 `ATTR_VERDICT_COLS` labels (`Song`, `Song LFC`), 792 explainer row, 800–802 confidence-tier prose, 849 section heading, 1052/1059/1066/1076 shard messages.

### F. `alz/viewer/template/js/tabs/kinase_explorer.js`
658–659 cell-type pill tooltip (`Song ${fold}×`, `Song n/a`).

### G. `alz/viewer/template/js/tabs/kinase_fivexfad.js`
281 col label `5xFAD snRNA`, 1671–1672 supergroup tooltips, 1718/1731/1735 attribution prose. (`fivexfad_lfc` renders as label "snRNA LFC" — not a cohort-name surface.)

### H. `alz/build_unified_viewer.py` (programmatic — map-driven)
| Line | String | Surface |
|---|---|---|
| 171, 173 | `"5xFAD sample manifest"`, `"5xFAD dataset index"` | audit-tab download table labels |
| 983 | `"label": "Song AD"` | song context label (→ C1 splits this later) |
| 1046–1049 | `_5xfad_incytr_ctx_labels` → `"5xFAD Cortex"`, `"5xFAD Hippocampus"` | Incytr tissue-toggle context labels |

### I. CSV export — **DEFERRED to F2** (per contract; F2 owns the export boundary, reads C2's map)
`kinase_explorer.js:881` (`song_topShare/topCell`), `kinase_fivexfad.js:759/761` (`5xFAD_snrna`, `fivexfad_kinase_*.csv`), `kinase_crosstable.js:1618/1620` (`Mouse/Human/5xFAD_med_NES`, `Song_fold`). C2 does NOT touch these; it provides the map F2 consumes in Wave 4.

### J. Out of scope (confirmed disjoint)
- `build_tcell_viewer.py` + `alz/tcell_viewer/` — no Song/5xFAD/Mukesh display names (only internal `song_*` concordance keys).
- Provenance stamps `cohort="5xFAD"` (`fivexfad/celltype_mea.py:389`, `fivexfad/ingest.py:834`) — the literal model name is correct provenance; leave UNLESS the Audit tab renders the field as a visible label (verify in plan Stage 0).
- All `song_*` / `fivexfad_*` / `mukesh_*` payload keys, `cohort_id=`, `data/datasets/song/`, module dirs.

## Ambiguous → resolutions
- `"cohort": "5xFAD"` in `fivexfad.py:1590` payload block — internal unless JS renders `block.cohort` as text (scan shows it doesn't). Treat internal; verify in Stage 0.
- CSV `song_topShare` headers — user-visible in Excel but deferred to F2 (export boundary).
