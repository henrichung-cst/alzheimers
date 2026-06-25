# Theme C1 — Song genotype split (de-aggregation)

**Contract:** `_contracts.md §C1`. **Audit:** `c1_audit.md`. **Wave:** 1 (contract producer). **Prereq:** C2 (uses the `MouseC1` label). **Consumers:** C3 (directional side-by-side), B2 (per-genotype sankeys). **Collision class:** bulk_mea pipeline + unified-viewer builder + JS — gate runs a bulk_mea recompute.

## Decisions (locked, P3 grill 2026-06-24)
1. C1 includes the pipeline/payload change (`recover.py` + `song.py`), not viewer-only.
2. `trajectory_label` splits per-genotype.
3. `top_celltype_1_song_lfc` is OUT (not a genotype pooling).
4. Crosstable renders **3 always-visible per-genotype med-NES columns** (4a), replacing the single "M med".

Split key: genotype = `contrast.split("_")[0]` ∈ `config.DISEASE_GROUPS = ["App","Tau","ApTt"]`. Per-genotype timepoint reduction mirrors the current pooled logic, applied within each genotype's ≤3 timepoints (median of sig-passing NES for `_mNes`; max-\|NES\| for peak; count for n_sig; `_classify_trajectory` over the genotype's timepoints).

## Scope boundary — C1 owns the crosstable agreement rework (plan-time call, veto on review)
The agreement/crossplay machinery (`_kxComputeAgreement` → `song.sig`/`song.dir`/`_agreeCategory`/`_agreeScore`, concordance badges, `_kxComparisonState`) currently keys on the **pooled** Song source. Anti-shim forbids a pooled value anywhere, and C1 ships to its gate **before** C3 — so C1 cannot leave agreement on a pooled remnant. **C1 reworks agreement to consume the 3 per-genotype Song sources.** Definition (honest, labeled — not pooled): the kinase's crossplay category = the category of its **most-concordant genotype** (argmax agreement with Human/5xFAD), and the badge/`_agreeScore` is **labeled with that genotype** (e.g. "ApTt concordant-up"), exactly as `peak_contrast` names the peak's contrast. A single `_agreeScore` remains for sorting. **C3 builds the dedicated side-by-side directional panel on top of the per-genotype sources C1 exposes — it does not fix a pooled remnant (C1 leaves none).**

---

## Stage 1 — Pipeline: per-genotype scalars (`alz/bulk_mea/recover.py`)
In `_build_kinase_activity_matrix` (lines ~204–271):
- For each `g in DISEASE_GROUPS`, over that genotype's contrasts (`[c for c in CONTRASTS if c.split("_")[0]==g]`):
  - `peak_NES_{g}` = max-\|NES\|; `peak_contrast_{g}` = its contrast.
  - `n_sig_{g}` = `(fdr[g_contrasts] < MEA_FDR_THRESH).sum()`.
  - `trajectory_{g}` = `_classify_trajectory` restricted to the genotype's timepoints (refactor `_classify_trajectory` to take a contrast subset).
- **Remove** the pooled `peak_NES`, `peak_contrast`, `n_sig_contrasts`, `trajectory_label` (anti-shim — no stored pooled value). Update the column writer + any in-module references.

## Stage 2 — Payload (`alz/viewer/cohorts/song.py`)
`_build_kinases_slice` (124–156): emit `peak_NES_{g}`, `peak_contrast_{g}`, `n_sig_{g}`, `trajectory_{g}` for the 3 genotypes; drop the pooled columns. Keep all 9 `NES_{c}`/`FDR_{c}` (unchanged). Keep `top_celltype_1_song_lfc` (out of scope). Specificity columns untouched.

## Stage 3 — Crosstable (`alz/viewer/template/js/tabs/kinase_crosstable.js`)
- `_kxComputeAgreement` (823–853): build `songVals` **3×**, one per genotype (`CONTRASTS.filter(c => c.split("_")[0]===g)`); produce `_mNes_App/_Tau/_ApTt` and `_songSource_{g}` (sig/dir/nSig each).
- Replace the single "M med" column (`_kxMedNesCell(r._mNes)` at 1230; header 1181/1185; sort key 1284) with **3 columns** `MouseC1_App / _Tau / _ApTt` med-NES (C2 display names), each via `_kxMedNesCell`. Sort: each column sorts on its own `_mNes_{g}` (signed, per F1).
- Agreement: `_agreeCategory` = category of the most-concordant genotype; `_agreeScore` single (labeled with that genotype); `_kxComparisonState`/`_kxComparePair`/`_kxThreeWayState` consume per-genotype sources. Detail header verdict lines (1446, 1467–1469, 1480–1481) show the 3 genotypes.
- Remove reliance on the deleted pooled `peak_NES` for the default sort fallback (1291) → use the `songOverallPeak` helper (Stage 4).

## Stage 4 — Other consumers of the removed pooled scalars
Shared transient helper `songOverallPeak(row)` → `{nes, contrast, genotype}` = the max-\|NES\| across the 3 per-genotype peaks (computed on the fly, **not stored**; always displayed genotype-labeled — `peak_contrast_{g}` already names the genotype):
- `kinase_explorer.js` (256–257, 535, 864): headline peak cell + sort use `songOverallPeak`; display unchanged (it already prints the contrast name = genotype-explicit).
- `kinase_audit.js:7`: default-contrast pick uses `songOverallPeak(...).contrast`.

---

## Verification
- **Pipeline:** re-run bulk_mea (`pixi run` the recover step) under the memory cap; `python alz/bulk_mea/summary.py` passes; spot-check a kinase's `peak_NES_{g}` = max-\|NES\| of that genotype's contrasts.
- **Viewer:** `pixi run viewer` → hard-refresh. `PAYLOAD.kinases` carries `peak_NES_App/Tau/ApTt`, `n_sig_*`, `trajectory_*`; **no** `peak_NES`/`n_sig_contrasts`/`trajectory_label` keys remain (grep the payload).
- **Browser click-through (human, authoritative):** crosstable shows 3 `MouseC1_*` med-NES columns (no single "M med"); each sorts signed; crossplay badge is genotype-labeled; detail panel shows 3 genotype rows; kinase-explorer peak cell still names its contrast.
- `command grep -rn 'peak_NES\b\|n_sig_contrasts\|trajectory_label\|_mNes\b' alz/` → only the helper/per-genotype forms remain; no bare pooled identifier.

## Out of scope
`top_celltype_1_song_lfc`, audit-verdict `song_lfc` column (contrast-scoped, correct), 3×3 mouse glyph (already per-contrast), specificity (unified), C2 naming internals, the dedicated directional disease-view panel (C3).
