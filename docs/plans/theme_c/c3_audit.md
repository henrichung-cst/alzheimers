# C3 Audit — Disease-direction focus (up/down ranking, early markers, biomarkers)

Read-only recon (2026-06-25). C3 is a **unified-viewer-only** lens for the AD (Song = `MouseC1`) cohort. No tcell, no MEA rerun. The two data files it needs are on disk and bounded (`site_level_ols.csv` 10.7 MB, `hpa_secretome.tsv` tiny) → **no heavy compute, no gate.** Contract: `_contracts.md §C1` (consumer side). TODO source: `TODO.md §C3` (lines 68-74).

**C3 is two entities:** kinase-level (sub-features 1,2 = NES per genotype) + protein/phosphosite-level (3,4,5 = biomarker hunting). That split drives the two-panel structure.

## A. Where the panel lives
- Viewer assembled by `alz/build_unified_viewer.py`; tabs registered in `TAB_MANIFEST` (`alz/viewer/template/js/02_ui_chrome.js:13-98`). Adding a tab = 4 steps (`alz/viewer/template/MANIFEST.md:8-19`): drop `js/tabs/<name>.js`, add `{{ raw(...) }}` to `index.html.j2`, add `<div id="tab-<name>">` to `body.html`, add a `TAB_MANIFEST` entry.
- **No disease-direction / biomarker tab exists today.** Closest structural matches: `crosstable` (sub-features 1,2 after C1's 3 per-genotype med-NES cols), `temporalv2` (NES trajectories — sub-feature 3), `kinase` Explorer `setWhitelist` (`temporal_v2.js:708-759` — sub-feature 5).

## B. Sub-features 1 & 2 (directional + ranking) — buildable from C1 payload
Fully buildable. After C1, `PAYLOAD.kinases` carries signed `peak_NES_{App,Tau,ApTt}`, `peak_contrast_{g}`, `n_sig_{g}`, `trajectory_{g}` + the 9 raw `NES_{c}`/`FDR_{c}` (`song.py:124-155`). Reusable: `_kxMedNesCell` (signed red/blue cell, `kinase_crosstable.js:859-863`), `_kxMouseGlyphCell` (3×3 all-contrast glyph, `:883`). Ranking key = `songOverallPeak(row).nes` (the transient C1 introduces, `c1_plan.md` Stage 4), signed per F1.

## C. Sub-feature 3 (early-change phosphosites) — two layers
- **Kinase-level (free, exists):** `_classify_trajectory` (`recover.py:204-243`) already emits `"early"` (sig only at 2mo); becomes `trajectory_{g}` per genotype after C1. Pure JS flag.
- **Site-level (the literal TODO ask, MISSING as a derived column):** `outputs/reports/kinase_attribution/site_level_ols.csv` (10.7 MB) has per-site `stoich_lfc_{c}`/`stoich_pval_{c}`/`stoich_fdr_{c}` + `raw_*` for all 9 contrasts. Already mirrored to `outputs/reports/unified_viewer/audit_sources/site_level_ols.csv` and loaded by the audit tab (`AuditDataStore.load("site_level_ols")`, `kinase_audit.js:11-12`). The per-site "early" classification is **not computed anywhere** — needs a new bounded build step.

## D. Sub-feature 4 (secretability) — HPA secretome on disk
- `outputs/reports/kinase_attribution_human/ctrl_audit/hpa_secretome.tsv` — 1,902 genes; cols `Gene`, `Secretome location` ("Secreted to blood" 785, "Secreted in other tissues" 286, "…extracellular matrix" 236, "…unknown" 116, "Secreted in brain" 76, …), `Secretome function`. Currently loaded only by the one-off `ctrl_outlier_suspect_lfc_table.py:38-42` (`load_secretome()`). Human Protein Atlas "Predicted secreted proteins" (`:6,20`). **Human-origin — which is the intended semantics (treatment target is human).** Lives in `outputs/`, not a tracked data input.
- UniProt `canonical_map.json` (`data/datasets/mukesh/analysis_cache/uniprot/`, 26 MB) is **sequences only** — no subcellular location. Incytr `Ligand` set is not a clean secretome proxy (mouse-DB-specific, requires reading an `.rda`).

## E. Sub-feature 5 (gene-list lookup) — input + LFC source
- No free-text/paste-a-list control exists. `setWhitelist()` (`temporal_v2.js:708-759`) takes kinase IDs programmatically; `ke-search` (`kinase_wiring.js:37-43`) is single-line over name+gene_symbol. A textarea is new.
- LFC sources: `top_celltype_1_song_lfc` (kinase-level scalar, in payload, `song.py:151` ← `recover.py:340-343`); audit verdict `song_lfc` (per kinase×contrast×cell_type, lazy, `kinase_audit.js:595,705`); `song_concordance` shards (per-gene parquet `gene_symbol,cell_type,contrast,song_lfc,…`, `song.py:498`, `04_slice_cache.js:120-140`). Join key = uppercased gene symbol.

## F. Collision class — unified-viewer-only
| File | Change | Overlaps C1? |
|---|---|---|
| `alz/viewer/template/js/tabs/kinase_disease_direction.js` (NEW) | new tab | no |
| `alz/viewer/template/body.html` | add `<div id="tab-...">` | no |
| `alz/viewer/template/js/02_ui_chrome.js` | `TAB_MANIFEST` entry | no |
| `alz/viewer/template/js/01_state.js` | `TAB_GUIDE` entry | no |
| `alz/viewer/cohorts/song.py` | new payload column(s) (secretable, early flags) | **YES — C1 Stage 2 edits same fn**; C1 Wave 1 first, C3 on top |
| `alz/build_unified_viewer.py` | secretome join + (deferrable) site-level shard | no (C1 doesn't touch it) |
| `alz/viewer_shared/template/js/06_export_csv.js` | seed `numCmp` if C3 ships before F1 | shared with F1 (Wave 4) — serialize |

No heavy compute → C3 is a normal Wave-2/3 fan-out theme.

## G. Open decisions (resolved in the 2026-06-25 grill)
- **G1 structure** → new "Disease Direction" tab, two stacked panels (kinase directional-ranking / biomarker table). Not a crosstable extension.
- **G2 early-change** → site-level is the ask; built from `site_level_ols.csv` as a **self-contained deferrable stage** (rest of C3 independent). Kinase-level `trajectory_{g}` flag stays in core (free).
- **G3 secretability** → **HPA, "secreted in human" semantics (the intended filter)**; promote file to a tracked `data/` path; surface the **category string** (not boolean); annotation only (never drops rows).
- **G4 gene-list** → self-contained mini-table (textarea → filter C3 table + LFC col); `setWhitelist` handoff is secondary.
- **G5 ranking** → `songOverallPeak()` signed default; header-click per-genotype.
- **G6 F1 ordering** → C3 seeds `numCmp` in `06_export_csv.js`; F1 adopts. P4 serialization note.
- **C4 tie-in (optional, veto on review)** → human-specificity column in the biomarker panel (h_spec already in payload), since C4 enforcement bites exactly there and the goal is human treatment.
