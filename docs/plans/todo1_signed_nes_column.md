# Plan: Display signed NES in the kinase table `|NES|` column

## Summary

The `|NES|` column in the kinase explorer table currently renders and labels the
_absolute_ value of peak NES. This plan changes it to display the **signed** value
while keeping magnitude-based sorting intact.

The change is **render-time only** in two JS files and two HTML template files.
No Python payload changes are needed — `peak_NES` in the payload already carries the
signed value (see [Data-build audit](#data-build-audit) below).

---

## Scope

Three viewer surfaces share the same kinase-table layout:

| Viewer | HTML template | JS renderer |
|--------|---------------|-------------|
| Unified (mouse + human + 5xFAD) | `alz/viewer/template/body.html` | `alz/viewer/template/js/tabs/kinase_explorer.js` |
| Unified 5xFAD tab | `alz/viewer/template/body.html` | `alz/viewer/template/js/tabs/kinase_fivexfad.js` |
| T-cell | `alz/tcell_viewer/template/body.html` | `alz/tcell_viewer/template/js/tabs/kinase_explorer.js` |

All three must be updated so the column is consistent across tabs.

---

## Data-build audit

### Unified viewer (mouse / Song cohort)

`alz/bulk_mea/recover.py` line 266–267:

```python
t1["peak_NES"] = t1.apply(
    lambda r: max((r[c] for c in present_nes), key=abs), axis=1)
```

`max(..., key=abs)` picks the element with the largest absolute value but returns
the **signed** original value. `peak_NES` in the payload is therefore already signed.

`alz/viewer/cohorts/song.py` line 215 passes it through verbatim:

```python
cols["peak_NES"].append(_get(ka_row, "peak_NES"))
```

### T-cell viewer

`alz/build_tcell_viewer.py` line 681–683:

```python
i_peak = max(finite, key=lambda t: abs(t[1]))[0]
cols["peak_NES"].append(nes_vec[i_peak])
```

Again the index of max-|NES| is found, but the stored value is `nes_vec[i_peak]`
(signed). Payload is signed.

**Conclusion:** the absolute value is applied _only_ at render time in JS. The
payload already carries signed `peak_NES` and per-contrast `NES_*` columns.

---

## Exact changes

### 1. Column header label: `body.html` (unified viewer)

**File:** `alz/viewer/template/body.html`  
**Line 88** (the `<th>` for `peak_NES`):

```html
<!-- BEFORE -->
<th data-col="peak_NES" data-metric="peakNes" title="Peak |NES| — max absolute NES across the active disease/timepoint scope (or all 9 contrasts if no scope). Click to sort.">|NES|</th>

<!-- AFTER -->
<th data-col="peak_NES" data-metric="peakNes" title="Peak NES — signed NES at the contrast with the largest |NES| in the active disease/timepoint scope (or all 9 contrasts if no scope). Click to sort by |NES|.">NES</th>
```

### 2. Cell value rendering: `kinase_explorer.js` (unified viewer)

**File:** `alz/viewer/template/js/tabs/kinase_explorer.js`  
**Line 856** (inside the `parts.push(...)` loop in `renderKinaseExplorer`):

```js
// BEFORE
`<td class="attr-num">${peakAbsNes != null ? peakAbsNes.toFixed(2) : '<span class="muted">—</span>'}</td>` +

// AFTER — r.peak_NES carries the signed value from the payload
`<td class="attr-num">${r.peak_NES != null && isFinite(r.peak_NES) ? (r.peak_NES > 0 ? "+" : "") + r.peak_NES.toFixed(2) : '<span class="muted">—</span>'}</td>` +
```

Note: `r.peak_NES` is loaded in `_ensureKinaseIndexes` at line 257 as
`peak_NES: K.peak_NES[i]`. It is the signed payload value. The local variable
`peakAbsNes` (set at line 815 via `_kineMaxAbsNesScoped`) is still used for the
**sort key** (see §Sorting below) — it must not be touched.

### 3. Column header label: `body.html` (t-cell viewer)

**File:** `alz/tcell_viewer/template/body.html`  
**Line 73**:

```html
<!-- BEFORE -->
<th data-col="peak_NES" data-metric="peakNes" title="Peak |NES| — max absolute NES across timepoint contrasts. Click to sort.">|NES|</th>

<!-- AFTER -->
<th data-col="peak_NES" data-metric="peakNes" title="Peak NES — signed NES at the contrast with the largest |NES| across timepoint contrasts. Click to sort by |NES|.">NES</th>
```

### 4. Cell value rendering: `kinase_explorer.js` (t-cell viewer)

**File:** `alz/tcell_viewer/template/js/tabs/kinase_explorer.js`  
**Line 678**:

```js
// BEFORE
`<td class="attr-num">${peakAbsNes != null ? peakAbsNes.toFixed(2) : '<span class="muted">—</span>'}</td>` +

// AFTER
`<td class="attr-num">${r.peak_NES != null && isFinite(r.peak_NES) ? (r.peak_NES > 0 ? "+" : "") + r.peak_NES.toFixed(2) : '<span class="muted">—</span>'}</td>` +
```

The t-cell `kinase_explorer.js` loads `peak_NES` at line 141:
`peak_NES: K.peak_NES[i]` — same pattern as the unified viewer, signed payload value.

### 5. 5xFAD tab

**File:** `alz/viewer/template/body.html`  
**Line 551** (`<th data-col="peakAbsNes">`):

```html
<!-- BEFORE -->
<th data-col="peakAbsNes" title="Largest absolute finite NES across the active age scope.">|NES|</th>

<!-- AFTER -->
<th data-col="peakAbsNes" title="Signed NES at the age with the largest |NES| in the active age scope. Sorted by |NES|.">NES</th>
```

**File:** `alz/viewer/template/js/tabs/kinase_fivexfad.js`  
**Line 796** (inside the row-rendering return template):

```js
// BEFORE
<td class="attr-num">${r.peakAbsNes == null ? '<span class="muted">—</span>' : r.peakAbsNes.toFixed(2)}</td>

// AFTER
```

The 5xFAD tab computes `peakAbsNes` locally in `_f5Metric` (lines 516–530) and
stores only the absolute value in `r.peakAbsNes`. It does not carry the sign. Two
options:

**Option A (preferred — minimal change):** Store `peakSignedNes` alongside
`peakAbsNes` in `_f5Metric`:

In `alz/viewer/template/js/tabs/kinase_fivexfad.js`, `_f5Metric` (lines 515–530):

```js
// BEFORE
function _f5Metric(group, ages) {
  let peakAbsNes = null;
  let sigCount = 0;
  const fdr = Store.state.filters.fdr;
  for (const age of ages) {
    const row = group.rows.get(Number(age));
    if (!row) continue;
    const nes = _f5Num(row.NES);
    if (nes != null) {
      const abs = Math.abs(nes);
      if (peakAbsNes == null || abs > peakAbsNes) peakAbsNes = abs;
    }
    const q = _f5Num(row.FDR);
    if (q != null && q < fdr) sigCount += 1;
  }
  return {peakAbsNes, sigCount};
}

// AFTER
function _f5Metric(group, ages) {
  let peakAbsNes = null;
  let peakSignedNes = null;
  let sigCount = 0;
  const fdr = Store.state.filters.fdr;
  for (const age of ages) {
    const row = group.rows.get(Number(age));
    if (!row) continue;
    const nes = _f5Num(row.NES);
    if (nes != null) {
      const abs = Math.abs(nes);
      if (peakAbsNes == null || abs > peakAbsNes) {
        peakAbsNes = abs;
        peakSignedNes = nes;
      }
    }
    const q = _f5Num(row.FDR);
    if (q != null && q < fdr) sigCount += 1;
  }
  return {peakAbsNes, peakSignedNes, sigCount};
}
```

Then in the row object built from `_f5Metric` (search for the `peakAbsNes` property
assignment, around line 42/563/720 depending on context), propagate `peakSignedNes`
alongside `peakAbsNes`. The sort key (`peakAbsNes`) is unchanged.

Line 796 render:

```js
// AFTER
<td class="attr-num">${r.peakSignedNes == null ? '<span class="muted">—</span>' : (r.peakSignedNes > 0 ? "+" : "") + r.peakSignedNes.toFixed(2)}</td>
```

---

## Sorting — no change needed

Sorting on the `peak_NES` / `peakAbsNes` column already uses `_kineMaxAbsNesScoped`
(unified, line 537–540) and `r.peakAbsNes` (5xFAD, line 563). These are magnitude
comparisons. They must remain unchanged: "sort by |NES|" is the correct ranking
behaviour even when displaying signed NES. The `data-col` attribute value on the
`<th>` does not change, so `wireKinaseTable`'s click handler in
`alz/viewer/template/js/tabs/kinase_wiring.js` is unaffected.

---

## Sign convention

Canonical: `+NES` = up in disease (`+` = up in disease is the project-wide LFC/NES
sign convention per CLAUDE.md). The payload `NES_<contrast>` columns carry this
sign. Displaying signed peak NES directly from `r.peak_NES` therefore propagates the
canonical sign with no extra transformation.

---

## Downstream consumers — no impact

- **NES profile heatmap** (`_renderNesProfile`) uses `r._nes[ci]` per-contrast, not
  `peak_NES`. It already handles sign (red = positive, blue = negative). Unchanged.
- **NES strip chart in audit panel** uses per-contrast `stoichNes` / `rawNes`
  directly. Unchanged.
- **`recover.py` column `peak_NES`** and **`build_tcell_viewer.py`**: payload
  already signed; no Python edits.
- **Sort comparators** (`_makeKeCompare`, `_f5Metric`): use `_kineMaxAbsNesScoped`
  / `peakAbsNes`; unchanged.

---

## Files changed (summary)

| File | Change |
|------|--------|
| `alz/viewer/template/body.html` | Line 88: `th` label/title → signed wording; Line 551: `th` label/title → signed wording |
| `alz/viewer/template/js/tabs/kinase_explorer.js` | Line 856: render `r.peak_NES` (signed) instead of `peakAbsNes` |
| `alz/viewer/template/js/tabs/kinase_fivexfad.js` | `_f5Metric`: track `peakSignedNes`; propagate through row object; Line 796: render signed |
| `alz/tcell_viewer/template/body.html` | Line 73: `th` label/title → signed wording |
| `alz/tcell_viewer/template/js/tabs/kinase_explorer.js` | Line 678: render `r.peak_NES` (signed) instead of `peakAbsNes` |

---

## Verification

1. Rebuild the viewer: `pixi run viewer` (or `python alz/build_unified_viewer.py`).
2. Hard-refresh the output in the browser (Ctrl+Shift+R / Cmd+Shift+R) — the viewer
   inlines payload as a `<script>` block; a normal refresh will serve the cached HTML.
3. On the kinase tab:
   - The column header now reads **NES** (not `|NES|`).
   - For a kinase known to be down-regulated (e.g. one with a negative NES), the
     cell should show a negative value with a leading `−` sign.
   - For a kinase known to be up-regulated, the cell should show `+X.XX`.
   - Clicking the column header still sorts by largest magnitude first (the arrow
     appears descending by default), so a strong `-2.1` outranks a weak `+0.3`.
4. Repeat on the 5xFAD tab and the T-cell viewer.
5. Check `DevTools → Network` to confirm `PAYLOAD.meta.generated_at` reflects the
   new build (stale cache symptom: header still shows `|NES|`).
