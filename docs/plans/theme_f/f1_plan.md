# Theme F1 — Signed-sort convention

**Contract:** `_contracts.md §F1`. **Audit:** `f_audit.md`. **Wave:** 4 (cross-cutting sweep, after table-adding themes). **Collision class:** both viewers' JS table tabs — applied as a sweep, never concurrently with C1/C3/B2 table edits.

## Decisions (locked, P3 grill 2026-06-25)
- Single shared comparator `numCmp(av, bv, dir)` in `alz/viewer_shared/template/js/06_export_csv.js` (the shared-util file both viewers load before tabs): signed numeric compare, **nulls/NaN always last regardless of `dir`** (`dir=+1` desc default, `-1` asc). Replaces every per-tab `Math.abs` numeric branch and the scattered `-Infinity`/`return 1` null logic.
- **Profile/peak columns sort by the signed NES of the peak-magnitude contrast** — keep selecting the kinase's headline contrast (max `|NES|`), rank by its sign. Descending = strongest-up on top; asc toggle → strongest-down tail.
- **Scope = user-facing sortable table columns only.** Leave magnitude best-pick selection and drill-down detail orderings (magnitude is correct there).

## Stage 1 — Shared comparator
Add to `06_export_csv.js`:
```js
function numCmp(av, bv, dir) {            // dir: +1 desc (default), -1 asc
  const an = (av == null || av !== av), bn = (bv == null || bv !== bv);
  if (an && bn) return 0;
  if (an) return 1;                        // nulls/NaN ALWAYS last
  if (bn) return -1;
  return dir > 0 ? (bv - av) : (av - bv);
}
```
(String columns keep `localeCompare` with direction — F1 is about *numeric* magnitude; the helper covers numeric branches.)

## Stage 2 — Single-value offenders (drop Math.abs)
- **crosstable `_kxSortRows`** (`kinase_crosstable.js:1297-1299`): replace `Math.abs(av)/abs(bv)` magnitude branch with `numCmp(av, bv, s.sortDir)` for m_med/h_med/f5_med/wmb/h_spec/m_spec. (agree_score + string cols already signed — route numerics through `numCmp` for null-last uniformity.)
- **human `_khSort`** (`kinase_human.js:278-279` + tie-breaks 303-320): `median_nes_sig_only` → signed via `numCmp`; remove the `Math.abs` tie-break magnitude.

## Stage 3 — Profile/peak offenders (signed peak)
Compute the signed NES at the argmax-`|NES|` contrast (the existing peak), then sort by `numCmp`:
- **`_kineMaxAbsNesScoped`** (`kinase_explorer.js:783`, tcell `:627`): add/return the **signed** value at the max-abs position (e.g. new `_kineSignedPeakNesScoped`), used by `_makeKeCompare` for nes_profile + peak_NES. Remove the `-Infinity` sentinel (numCmp handles null-last — fixes the ascending nulls-first bug).
- **human `_khSort` nes_profile** (`:275-276`): replace `Math.max(...map(abs))` with the signed value at max-abs element.
- **5xFAD** (`kinase_fivexfad.js`): the stored `peakAbsNes` (set in the row builder) gains a signed companion `peakNes` (signed value at the peak); `_f5FilterSort` (`:626`) sorts the profile column on `peakNes` via `numCmp`. (`peakAbsNes` may stay for any display that needs magnitude — but the SORT uses signed.)

## Stage 4 — Adopt the helper in already-signed comparators (uniformity)
Route the numeric branches of `_attrVerdictCmp` (both viewers), `_f5AttrCmp`, `_ipFilterRows`, `_selectTopK` through `numCmp` so null-last + direction are identical everywhere. Behavior-preserving for these (already signed) — consolidation, not a fix.

## Out of F1 scope (do NOT change)
- Best-pick selection: `kinase_crosstable.js:1016` (song-LFC), `:1041` (5xFAD-LFC) — magnitude picks the strongest evidence; correct.
- Drill-down orderings: `kinase_fivexfad.js:1842` (per-cell β), `kinase_human.js:565` (per-site δ) — detail tables, not the primary sortable grid.
- String comparators, AuditTable debug-table fallback.

## Verification (browser, human — authoritative)
- Every numeric column: default click = descending with **largest positive on top**; second click ascending = **largest negative on top**; a row with +3 ranks above −3 in descending and below it in ascending.
- Nulls/NaN sit at the bottom in **both** directions (specifically re-check KE + human — the old `-Infinity` bug put them on top when ascending).
- Profile columns (KE/5xFAD/human nes_profile): a kinase whose peak is strongly negative sinks under descending, rises to top under ascending.
- `command grep -rn "Math.abs" alz/viewer/template/js alz/tcell_viewer/template/js` → remaining hits are only best-pick/drill-down (Stage-out-of-scope list), never a user-facing sort key.
