# Theme C3 — Disease-direction focus

**Status:** done — folded into Kinase Explorer (2026-06-29). **Contract:** `_contracts.md §C1` (consumer side). **Audit:** `c3_audit.md`. **Wave:** 2/3 (consumer; unified-viewer-only, no heavy compute). **Prereqs:** C1 (per-genotype NES split), C2 (`MouseC1` display name). **Collision class:** unified viewer; shares `song.py` with C1 (C1 first) and `06_export_csv.js` with F1 (serialize).

## As built — folded into Kinase Explorer (no separate tab)

C3 originally shipped as a standalone `diseasedirection` tab (kinase directional-ranking + candidate-biomarker table). It was a strict subset of the Kinase Explorer plus a gene-list textarea duplicating the Explorer's own whitelist/search; it was folded into the Explorer and deleted:

- **Per-genotype direction → Explorer main table.** The App / Tau / ApTt columns in the `ke-table` carry each genotype's NES trend. (The first fold-in shipped a `peak_NES` + trajectory-classification badge; both were dropped in the **kinase trend refactor** — peak NES and the trajectory labels were judged worthless and replaced by a single NES **trend pill** classified client-side from the per-genotype NES vector. See `kinase_trend_refactor.md` for the authoritative current design.)
- **Translational annotations → Explorer detail pane.** Selecting a kinase shows a `kx-annot-strip` with **Secreted (human, HPA)** (`secretome_location`), **LFC (top cell type)** (`top_celltype_1_song_lfc`), and **SEA-AD expr** (`h_spec` = max `seaad_location_score` from `PAYLOAD.attribution_index`). Lives in the **shared** `kinase_detail.js` (`_renderKinaseTranslation`, both render branches); each field self-gates, so it renders only where present — Song only (t-cell / 5xFAD lack the fields → empty strip → nothing drawn). **This is the part of C3 that survives unchanged.**
- **Gene-list lookup dropped, not ported.** It duplicated the Explorer's whitelist/search and has no meaning in a single-kinase detail pane.
- **Tab deleted.** `kinase_disease_direction.js`, its `TAB_MANIFEST` entry (`02_ui_chrome.js`), `body.html` panel, `01_state.js` `TAB_GUIDE`, the `build_unified_viewer.py` include, the `MANIFEST.md` row, and the dead `dd-*` CSS — all removed.

Both viewers build clean; all edited JS passes `node --check`. Browser confirmation (the three detail annotations for a Song kinase) is the outstanding human gate.

## Decisions that carried through

- **Secretability = HPA secretome, "secreted in human" semantics** (the intended filter — treatment target is human; a mouse-secretome flag would be true-but-unhelpful). `hpa_secretome.tsv` joined on uppercased gene symbol; surfaces the **category string** ("Secreted to blood" / "Secreted in brain" / blank), **not a boolean**; annotation only, never drops rows; labeled **"Secreted (human, HPA)"** so semantics are unambiguous.
- **Ranking:** `songOverallPeak().nes` signed (F1) is the Explorer's default sort; header-click ranks by a chosen genotype (App / Tau / ApTt).
- **F1 ordering:** C3 shipped before F1 → C3 seeded `numCmp` into `06_export_csv.js`; F1 later *adopts* it. No inline shim.
- **C4 tie-in:** the detail-pane `h_spec` annotation is the human-specificity surface C4's guard bites on.

## Stage 1 — Secretome ingestion (build-time, bounded) — built

- `hpa_secretome.tsv` promoted to a tracked input (provenance: HPA "Predicted secreted proteins"); **not committed** (data-file rule) — fetched/copied by the runner + MANIFEST.
- `build_unified_viewer.py` loads it (small read), builds `{GENE_UPPER → Secretome location}`, attaches `secretome_location` to Song kinases (`song.py:_build_kinases_slice`). Blank where absent.
- `numCmp(av,bv,dir)` (signed, null-last regardless of dir) seeded into `06_export_csv.js`.

## Stage 4 — Site-level early-change (DEFERRABLE) — not built

The genuinely sharp "early *and* secretable" diagnostic view: per-phosphosite early-change joined to the secretome flag. Self-contained and deferrable; never built. As shipped, the actionable early signal is the **kinase-level** `trajectory_{g}=="early"` badge in the Explorer main table, one resolution coarser than per-site.

If built: a step (`build_unified_viewer.py` or a small `alz/.../site_early_change.py`) reads `site_level_ols.csv` (bounded ~10.7 MB — DuckDB / chunked, never whole-file if it grows), classifies each `site_id` per genotype **early in g** iff `stoich_fdr_{g}_2mo < FDR_THRESH` AND not `< FDR_THRESH` at `{g}_4mo`/`{g}_6mo` (`MEA_FDR_THRESH` from config), and emits a per-gene shard (`gene_symbol → early_sites[]` with genotype + 2mo LFC). The detail-pane annotation strip would gain an **"early sites"** entry — the actual early-secretable candidate view.

## Verification

- **Browser (human, authoritative):** App/Tau/ApTt peak-NES cells show the trajectory badge; a known 2mo-only kinase shows the `early` badge. Detail pane shows the Secreted / LFC / SEA-AD-expr strip for a Song kinase — secretome category for a known secreted gene, **blank (not "no"/false) where HPA lacks the gene**; no fabricated LFC. The Disease Direction tab is gone.
- Payload: `secretome_location` present on Song kinases; values ∈ HPA categories ∪ blank. `node --check` passes on the edited JS; both viewers build exit 0; built unified HTML carries zero `diseasedirection`/`_ddRender` refs.

## Out of scope

The C1 split itself (consumed), C2 naming internals (consumed), the F1 sweep (C3 only seeds `numCmp`), tcell viewer (the detail strip self-gates off there), any MEA/pipeline rerun, protein-LFC layer for non-kinase non-site genes (not in payload — shown blank honestly), UniProt subcellular fetch (HPA chosen).
