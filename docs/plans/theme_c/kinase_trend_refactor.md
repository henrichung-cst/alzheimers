# Kinase trend refactor — drop trajectory-classification + peak-NES, adopt trend pills

**Status:** done — implemented all cohorts (2026-06-29). **Supersedes:** the trajectory/peak-NES portions of `c3_plan.md` and the C1 per-genotype peak scalars. **Trigger:** the kinase trajectory labels (`peaked/progressive/declining/sustained/single_contrast/early/late/none/mixed`) are confusing and partly dead code (`early`/`late` are unreachable; the badge only ever styled `sustained`); "peak NES" is a worthless display scalar. **Locked decision:** one trend vocabulary repo-wide — `always_up / always_down / monotonic_up / monotonic_down / mixed` (the existing incytr set). `single-peak` / `single-trough` are dropped.

## Two systems — keep one, delete the other

| | System 1 — **KEEP** | System 2 — **DELETE** |
|---|---|---|
| What | incytr pathway **trend** (sign-vector) | kinase **trajectory classification** + peak NES |
| Code | `alz/viewer/shared/trajectory.py` `annotate_trajectory_columns`, `traj_labels`/`sign_vec`, `_SIGN_VEC_LABELS`; `_5xfad_annotate_trajectory_columns` (delegates to it) | `alz/bulk_mea/recover.py` `_classify_trajectory`; `trajectory_{g}`, `peak_NES_{g}`, `peak_contrast_{g}`, `n_sig_{g}`; JS `songOverallPeak`, `_keTrajBadge` |
| Vocabulary | always-up/always-down/monotonic-up/monotonic-down/mixed | peaked/progressive/declining/sustained/single_contrast/early/late/none/mixed |
| Verdict | the valuable trend; already the 5-label set | the worthless labels; gone with prejudice |

The two share only the word "trajectory" and the token "mixed" — but System-1 `mixed` = a real NES sign reversal across timepoints (kept), System-2 `mixed` = a catch-all (deleted). The 5xFAD `f5-mea-trajectory` element is a **time-series line chart** (`_f5RenderTrajectory`), not a label — kept, though we may rename it to "NES over age" in a later pass (not in scope).

## The replacement — `TrendFilter.classify(values)`

`TrendFilter.vectorMatches` (`trend_filter.js`) is a *matcher*; the pills need a *classifier*. Add `classify(values)` returning a single label from an ordered numeric NES vector, priority-ordered so the non-exclusive patterns resolve deterministically:

1. `always_up` (all > 0) — refine to `monotonic_up` if strictly increasing
2. `always_down` (all < 0) — refine to `monotonic_down` if strictly decreasing
3. `monotonic_up` / `monotonic_down` (sign-mixed but strictly ordered)
4. `mixed` (sign reversal, not monotonic)
5. `null` when < 2 finite values (insufficient data → muted "—" pill, never a fabricated label)

`classify` and `vectorMatches` share one definition each of up/down/monotonic/mixed (no divergent logic). Same input the pills and the filter both use: each genotype's ordered `NES_{2mo,4mo,6mo}` vector — already in the payload, already what `_kineTrendMatches` reads (`r._nes`). **No new payload column; no peak column.**

`TrendFilter`: collapse `KINASE_VALUES` into `INCYTR_VALUES` (one list of 5); delete the `peak`/`trough` branches from `vectorMatches` and the `peak`/`trough`/`single peak`/`single trough` entries from `LABELS`/`options`.

## Removal inventory

### Pipeline — `alz/bulk_mea/recover.py`
- Delete `_classify_trajectory` (incl. the dead `early`/`late` branches and `_SUSTAINED_RATIO_THRESH`).
- In `_build_kinase_activity_matrix`: drop the per-genotype loop that emits `peak_NES_{g}`/`peak_contrast_{g}`/`n_sig_{g}`/`trajectory_{g}`.
- In `_build_kinase_hypothesis_table`: the t3 sort currently keys on `_max_n_sig` then `_abs_peak` (peak_NES). Replace with a live sort over the raw NES/FDR columns already in t1 — total significant contrasts (`FDR < MEA_FDR_THRESH` across all 9) then max |NES| across all 9. Drop the per-genotype-scalar passthrough block.
- `print_summary`: drop the "Trajectory distribution" block.
- **Schema change → regenerate `kinase_activity_matrix.csv`** (bounded, ~240 kinases / 181 KB) via the attribution-recovery task before rebuilding viewers.

### Song payload — `alz/viewer/cohorts/song.py`
- Drop the `peak_NES_{g}`/`peak_contrast_{g}`/`n_sig_{g}`/`trajectory_{g}` columns from `_build_kinases_slice` (the `NES_{c}`/`FDR_{c}` vectors stay — pills compute from them).

### Song viewer — `kinase_explorer.js` / `body.html` / `styles.css` / `01_state.js`
- Replace `_kePeakNesCell` + the App/Tau/ApTt `peak_NES_{g}` cells with a per-genotype **trend pill** (`TrendFilter.classify` on each genotype's NES vector). Delete `_keTrajBadge` and the `.ke-traj*` CSS; add `.ke-trend-pill*` styling.
- Delete `songOverallPeak` (`kinase_explorer.js`). Its consumers:
  - `kinase_audit.js` `_selectedAuditContrast` — replace the "overall peak contrast" fallback with a live max-|NES| contrast from `K.NES_{c}` (no stored peak).
  - `kinase_crosstable.js` default sort fallback — replace with live max-|NES| across the NES vector.
- Sort on the App/Tau/ApTt columns becomes **categorical** (group by trend label in a fixed order: always_up → monotonic_up → mixed → monotonic_down → always_down → "—"). Header tooltips updated.
- `body.html`: App/Tau/ApTt `<th>` titles drop the "signed peak NES" wording → "trend across timepoints". The **Trend filter** dropdown (`ke-filter-pattern`) now renders `TrendFilter.optionsHtml("incytr")` (the 5-set); the per-disease matching in `_kineTrendMatches` is unchanged.
- `01_state.js` `kinase` TAB_GUIDE: rewrite the trajectory-badge bullet + the peak-NES wording to describe the trend pills; drop the dead-label text I added earlier.
- CSV export (`exportKinaseCsv`, lines 961-962): replace `peak_NES_App/Tau/ApTt` with `trend_App/Tau/ApTt` (the classified label string).

### Detail pane — `kinase_detail.js` (shared)
- The translational-annotation strip stays. No peak/trajectory there now; nothing to remove except confirm no `peak_NES` reference crept in (it didn't).

### t-cell — `slices_kinase.py` / `tcell .../kinase_explorer.js`
- `slices_kinase.py`: drop `trajectory` (dead — always `""`), `peak_contrast`, `peak_NES`. Keep `n_sig_contrasts` (feeds the n_sig column).
- `tcell kinase_explorer.js`: replace the single `peak_NES` column (display line ~759, sort key ~390, build lines 157-159) with **one trend pill** from `r._nes`; align the Trend filter options to the 5-set; CSV header (~776) `peak_NES` → `trend`.

### 5xFAD — `fivexfad.py` / `kinase_fivexfad.js`
- `fivexfad.py`: **no change to `_5xfad_annotate_trajectory_columns`** (System 1, kept).
- `kinase_fivexfad.js`: `_f5TrendMatches` already uses `TrendFilter` — just inherits the 5-set. Replace the single-TG `peak_NES` in the CSV export (line 682) and any peak display with a trend pill. Keep `_f5RenderTrajectory` (the chart).

### Docs
- `c3_plan.md`: replace the trajectory-badge / peak-NES sections with a pointer-free rewrite (the trend-pill design); keep the secretome/h_spec detail-pane content.
- `p4_dag.md`, `_contracts.md §C1`: the C1 contract currently promises `peak_NES_{g}`/`trajectory_{g}`/`songOverallPeak` to consumers — rewrite to the trend-pill / NES-vector contract. `c3_audit.md` left as dated history.
- `MANIFEST.md` / payload schema notes naming these columns — updated.

## Scope question for approval

The directive named the Song **App/Tau/ApTt** columns, but also "drop the terms throughout this entire repo" and "peak NES is completely worthless." This plan therefore applies the **vocabulary + peak removal to all three cohorts** and replaces every kinase view's peak column with a trend pill (Song = 3 per-genotype pills; t-cell / 5xFAD = 1 pill), consistent with the shared-viewer contract. If you'd rather scope the *column replacement* to Song only and merely delete the dead t-cell `trajectory` field, say so — the System-2 deletion in `recover.py`/`song.py` is Song-specific regardless.

## Verification & sequencing

1. Regenerate `kinase_activity_matrix.csv` (attribution-recovery task) — schema no longer carries peak/trajectory.
2. Rebuild: `pixi run viewer` (unified) + `pixi run python alz/build_tcell_viewer.py` (t-cell), both exit 0.
3. `node --check` on every edited JS.
4. **sce4 parity unaffected** — no scoring-path or incytr edit; `verify-incytr-sce4` not required but harmless to spot-run.
5. Built-HTML greps: zero `peak_NES`, zero `trajectory_App/Tau/ApTt`, zero `songOverallPeak`, zero `_keTrajBadge`; `TrendFilter` `LABELS` carries no `peak`/`trough`.
6. **Browser (authoritative):** App/Tau/ApTt show trend pills matching each genotype's NES arc; the Trend filter offers exactly the 5; t-cell/5xFAD kinase views show a single trend pill; no peak-NES column anywhere; sce4/incytr trend (System 1) still renders always-up/…/mixed.

## Out of scope
The 5xFAD `_f5RenderTrajectory` chart (kept; possible "NES over age" rename later), the incytr System-1 trend (kept verbatim), any MEA rerun beyond the bounded attribution-recovery regen, F2 CSV-export format (only the column set changes, not the serializer).
