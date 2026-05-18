# Change request 03 — Human MEA: control donors side-by-side (LOO)

## Goal

In the Human (Mukesh / NBB) kinase tab, render each control donor's
MEA next to the AD donor MEAs so we can spot cases where the
"AD-vs-CTRL" enrichment is being washed out by an outlier control.

## Locked decisions

- Each control donor scored leave-one-out:
  `delta_CTRL_i = stoich(CTRL_i) − nanmean(stoich(CTRL_{−i}))`.
- Same residue tracks (`st`, `py`) and same MEA wrapper as treatment
  donors — control NES values land on a directly comparable scale.
- Control columns are visually delineated from treatment columns in
  the viewer (a divider + a "CTRL" group header) so they're not
  mistaken for disease cases.

## Surface map

- `alz/ingest_mukesh_perdonor.py` — builds donor deltas (`_build_donor_deltas`
  at line 79) and runs MEA. Today only iterates AD donors.
- `alz/seaad_human_agreement.py:50` — reads `recurrence*.csv` from
  `perdonor/` to compute SEA-AD agreement. Recurrence stays
  AD-only; SEA-AD agreement is unaffected.
- `alz/viewer/template/js/tabs/kinase_human.js` — donor iteration,
  column rendering, detail panel.
- `alz/build_unified_viewer.py` — packs `PAYLOAD.human` with `donors`,
  `donors_all`, `ctrl_donors`, and the `NES_<donor>_vs_CTRLmean`
  columns.

## Implementation steps

### A. Pipeline — emit control MEA

1. **`alz/ingest_mukesh_perdonor.py`**
   - Replace `_build_donor_deltas` so it returns deltas for *both*
     groups:
     ```
     # AD: stoich(AD_i) - nanmean(stoich(all CTRL))
     # CTRL: stoich(CTRL_i) - nanmean(stoich(CTRL_{-i}))  # LOO
     ```
     The AD construction is unchanged. CTRL is new and uses a small
     helper that, for each i, masks column i out of `ctrl_block`
     before computing the column-wise mean.
   - Result key naming: keep `<donor>_vs_CTRLmean` for both groups
     (so the existing viewer regex still works). Group membership is
     carried separately, see step 3.
   - For symmetry against the existing AD path (mean over *all*
     controls), document that AD uses `mean(all 8 CTRL)` while CTRL
     uses `mean(7 other CTRL)`. This is an asymmetry on purpose: an
     AD donor is a new observation against the full CTRL reference;
     a CTRL donor is a leftover-one against the remaining reference.

2. Recurrence table (`recurrence{,_raw}{suffix}.csv`) stays AD-only —
   it's the "AD donors with significant kinase" summary feeding
   `seaad_human_agreement.py`. Don't pollute it with CTRL rows.
   Instead emit a sibling table `recurrence_ctrl{,_raw}{suffix}.csv`
   with the same shape so we can show CTRL recurrence in the viewer
   if desired.

3. **Sample-mapping export** — write a small JSON sidecar at
   `perdonor/donor_groups.json`:
   ```json
   {"ad": ["...","..."], "ctrl": ["...","..."]}
   ```
   So the viewer build doesn't have to re-parse `sample_mapping.csv`.

### B. Viewer build (`alz/build_unified_viewer.py`)

- Extend `PAYLOAD.human` with:
  - `donors`: AD donors (unchanged; today this is the rendering axis)
  - `ctrl_donors`: NEW, list of control donors
  - `donors_all`: AD + CTRL (existing; used for raw lookups)
- Pack `NES_<ctrl_id>_vs_CTRLmean` columns into the same kinase block
  as AD donors. Same key shape so the wide loaders pick it up
  without schema changes.
- Bump the payload version flag so older viewers fail loudly.

### C. Viewer JS (`alz/viewer/template/js/tabs/kinase_human.js`)

1. **Column model**: instead of a flat `donors` axis, render two
   groups side-by-side with a thin spacer column between them:
   ```
   | AD donor 1 ... AD donor N | spacer | CTRL donor 1 ... CTRL donor M |
   ```
   Add a `donor_kind: "ad" | "ctrl"` derived from `_KH.ctrl_donors`.
2. **Filtering**: donor multiselect already exists. Add a "Show
   controls" toggle (on by default). When off, CTRL columns are
   hidden — restores the current view.
3. **Recurrence (`n_donors_sig`, `n_donors_up`, `n_donors_down`,
   `n_donors_tested`)**: keep these computed over AD only. CTRL has
   no clinical meaning for "recurrence." Add a separate
   `n_ctrl_sig` column hidden by default; useful when investigating
   wash-out.
4. **Detail panel** (`audit-mea-trajectory`, etc.):
   - When the user clicks a kinase row, the per-donor NES bar already
     plots all donors. Just split it visually by group and color CTRL
     bars in a muted palette.
   - Add a small annotation: "CTRL spread = ±X NES" (1 SD over the
     CTRL group) so the outlier-vs-band comparison is immediate.

### D. SEA-AD agreement (`alz/seaad_human_agreement.py`)

- No change. `recurrence` is still AD-only.

## Runtime

- Per-donor MEA is fast; adding ~8 more donors per track is +~30s
  total. No infra changes.

## Risks

- LOO mean reuses the same site coverage mask. If a control donor has
  unique NaN sites the mean-of-others may shift; OK, the rank list
  drops the site anyway.
- Visual real-estate: 17 donor columns may overflow the table on
  narrow viewports. Implement column-group collapse (one click to
  collapse all CTRL into a single summary chip) before shipping.

## Open question

- Should the *spacer column* between AD and CTRL include a "CTRL
  spread" sparkline summary by default (1 SD band), or only inside
  the kinase detail panel? Default I'd ship: detail panel only,
  table stays clean.
