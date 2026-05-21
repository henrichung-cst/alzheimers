# Plan — Kinase library motif logo + KL score in Measurement Trace

Date: 2026-05-21
Scope: mouse Kinase Audit and human Kinase viewers, Measurement Trace tab only.

## Goal

Above the per-site Measurement Trace table in both viewers, render a single sequence logo: the kinase's preference pattern from the `kinase_library` package's PSSM (9 positions: −5..−1 and +1..+4; center handled via `kinase.st_fav` for ST kinases).

Add a `kl_score` column to the Measurement Trace site rows so each row carries the percentile-rank value the library used to decide substrate-set membership.

## Data-side changes

### 1. Stamp `kl_score` into mouse `mea_substrate_sets.csv`
File: `alz/kinase_enrich.py:340-352`

After the existing `create_kin_sub_sets` call, look up each `(kinase, motif)` row in `rpd.data_kl_values` and write the percentile rank into a new `kl_score` column. New schema:

```
kinase, contrast, motif, residue_type, track, kl_score
```

Hard-fail if any (kinase, motif) row is missing from `data_kl_values` — that would mean set construction and scoring drifted.

### 2. Stamp `kl_score` into human `mea_substrate_sets.csv`
File: `alz/ingest_mukesh_perdonor.py`

Mirror change. Human keying is `(donor, kinase, motif)`.

### 3. Per-kinase PSSM payload
File: `alz/build_unified_viewer.py`

For each kinase referenced by either viewer, emit:

- `matrix`: 23×9 (ST) or 20×9 (Y) `kl.get_matrix(kinase, mat_type='norm')` values
- `st_fav`: `{'S': ..., 'T': ...}` for ST kinases, `null` for Y
- `kin_type`: `"ser_thr"` or `"tyrosine"`
- `positions`: `[-5, -4, -3, -2, -1, 1, 2, 3, 4]`

Inline into PAYLOAD under `payload.kinase_motifs[kinase_id]`. Estimated size ~300 KB total.

## Viewer-side changes

### 4. Sequence-logo widget
New file: `alz/viewer/template/js/widgets/sequence_logo.js`

Pure SVG renderer. Input: `{matrix, positions, st_fav, kin_type}`. For each position, stack amino-acid glyphs with heights scaled by `probability × information_content` (`info = log2(20) − entropy`). Standard Lesk amino-acid chemistry palette. Position 0 (the phosphoacceptor) rendered as a fixed glyph between positions −1 and +1: `S/T` split by `st_fav` proportions for ST kinases, fixed lowercase `y` for Y kinases.

### 5. Mouse Measurement Trace
File: `alz/viewer/template/js/tabs/kinase_audit.js:1279-1286`

Insert a header section above the table:

```html
<section class="audit-panel">
  <h4>Substrate motif (kinase library)</h4>
  <div id="audit-trace-logo"></div>
  <p class="kinase-stage-note muted">
    Position-specific amino-acid preferences from the kinase library PSSM
    for ${kinase_name}. Center (0) is the phosphoacceptor.
  </p>
</section>
```

Render via the new widget from `payload.kinase_motifs[ctx.kinase_id]`. Add `kl_score` to the Measurement Trace column list at line 1285. Source: join `mea_substrate_sets.csv` rows by `(kinase, contrast, motif)` and carry `kl_score` into the per-site lookup map.

### 6. Human Measurement Trace
File: `alz/viewer/template/js/tabs/kinase_human.js`, `_khRenderTrace` (line 506-562).

Same header pattern. Add `kl_score` column to the trace table head (line 548-551) and body (line 555 onward). Source: human-side `mea_substrate_sets.csv` from `ingest_mukesh_perdonor.py` (step 2) keyed by `(donor, kinase, motif)`.

### 7. Styles
File: `alz/viewer/template/styles.css`

`.kinase-motif-logo` — centered, max-width 540px, height ~150px. `.sequence-logo svg text { font-family: monospace; }`.

## Verification

1. **KL stamp round-trip.** For 5 random `(kinase, motif)` rows pulled from `mea_substrate_sets.csv`, recompute the score via `Substrate(motif).score([kinase])` and confirm agreement with the stamped value.
2. **Library logo matches the package's own renderer.** Pick one kinase (e.g. AAK1), call `Kinase("AAK1").seq_logo()` and visually compare to our viewer's SVG. Shape must agree.
3. **Same library logo on both viewers.** Open CDK5 in mouse Kinase Audit and human Kinase — the rendered library logo SVG must be visually identical.
4. **`kl_score` populated.** Both viewers' Measurement Trace tables show finite numeric values in the new column.

## Non-goals

- No empirical / data-derived logo. Only the kinase library PSSM is rendered.
- No logo anywhere outside Measurement Trace.
- No re-introduction of per-(kinase, site) score elsewhere.
