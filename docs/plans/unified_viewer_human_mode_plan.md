# Unified Viewer — Human (Mukesh / NBB) mode

Plan for adding a Human cohort mode to `outputs/reports/unified_viewer/` alongside the existing Mouse (Song) view. Single HTML deliverable, top-level mode toggle, near-identical kinase explorers in each mode for visual parity.

## Goal

A reviewer should be able to open the unified viewer and flip between Mouse and Human cohorts with a single chip in the header, landing in a kinase explorer with the same look, feel, and interaction primitives in either mode. Mouse-mode functionality is unchanged; Human mode is additive.

## Scope

Strict scope for this lift:

- **Mouse mode** — exactly the current viewer. Tabs: Kinase, Temporal, Incytr Heatmap, Incytr Pathways, Methods. No changes to mouse behavior.
- **Human mode** — single tab: Kinase. No temporal axis (NBB is cross-sectional). No Incytr. No cell-type attribution. No methods page.

Out of scope:

- Cross-cohort comparison views (scatter, concordance counters, joined heatmaps). The "comparison" affordance for now is the mode toggle — users compare by flipping modes on the same kinase.
- Human Incytr integration.
- Human cell-type attribution (the NBB pipeline does not produce attribution; explicit in `conf/human_nbb/parameters.yml`).
- Methods page for the human pipeline.
- AD-15 special handling. All donors are treated identically; AD-15 appears in the donor multiselect like any other donor.

## Architecture

```
┌─ header ──────────────────────────────────────────────────┐
│ Kinase + Pathway Viewer   [Mouse | Human]   FDR< [_]      │
└───────────────────────────────────────────────────────────┘
[Kinase] [Temporal] [Incytr Heatmap] [Incytr Paths] [Methods]   ← Mouse mode
[Kinase]                                                        ← Human mode
```

- One HTML file. One payload.
- `PAYLOAD.human = {kinases, contrasts:[10 donor IDs], perdonor_index}` lives alongside the existing mouse slices (`PAYLOAD.kinases`, `PAYLOAD.celltypes`, `PAYLOAD.attribution_index`, …). Mouse-side payload structure is unchanged.
- If `outputs/reports/kinase_attribution_human/perdonor/` is missing at build time, the human slice is omitted and the mode toggle is hidden — the artifact is byte-equivalent to today's mouse-only viewer for repos without human outputs.
- `state.view.mode ∈ {"mouse","human"}`. Selection state is mode-scoped so switching modes never carries a mouse-kinase selection into the human panel.
- URL hash includes `mode=` for bookmarkability.

## Build side (`alz/build_unified_viewer.py`)

New function `build_human_slice()` reading `outputs/reports/kinase_attribution_human/perdonor/`:

- `mea_perdonor{,_pY}.csv` → long form keyed on (kinase, donor, residue_type), used to populate `NES_<donor>` / `FDR_<donor>` columns and the per-donor leading-edge substrates.
- `kinase_donor_nes{,_pY}.csv` and `kinase_donor_fdr{,_pY}.csv` → already-wide kinase×donor matrices, the primary source for the human kinase slice.
- `recurrence{,_pY}.csv` → per-kinase summary columns (`n_donors_sig`, `n_donors_up`, `n_donors_down`, `n_donors_tested`, `median_nes`, `median_nes_sig_only`).
- `mea_global_shift{,_pY}.csv` and `winsorized_sites{,_pY}.csv` → diagnostics shown in the human detail panel.

`PAYLOAD.human` columnar shape (mirrors the existing `PAYLOAD.kinases` pattern):

```
human.kinases  = {id, name, gene_symbol, residue_type,
                  n_donors_sig, n_donors_up, n_donors_down,
                  n_donors_tested, median_nes, median_nes_sig_only,
                  NES_<donor>, FDR_<donor>  (one pair per donor)}
human.contrasts = ["AD-01_vs_CTRLmean", …, "AD-15_vs_CTRLmean"]
human.perdonor_index = {kinase_id, donor_id, NES, FDR,
                        leading_substrates}   (long form, sparse)
human.global_shift   = {donor_id, median_shift, pct_pos_before, pct_pos_after}
human.winsorized_sites = pointer to per-donor counts (full table copied beside
                        HTML, not inlined)
```

Kinase ID vocabulary for the human slice is its own — built from the union of ST + pY recurrence tables — and is independent of the mouse kinase vocabulary. There is no cross-cohort kinase join in this lift; that becomes relevant only when comparison views are added later.

Estimated builder lift: ~150 lines, one new function + a `--human-cohort` flag, wired into the existing `main()` so the default build picks it up automatically when the perdonor directory exists.

## JS side (`alz/viewer/template/js/`)

Changes confined to:

1. **Mode toggle** in `body.html` header chrome (~10 lines HTML).
2. **State** (`01_state.js`) — add `state.view.mode`, `SET_MODE` action in the reducer (~15 lines). Mode-scoped selection.
3. **URL hash** (`03_filters_hash.js`) — round-trip `mode=` (~15 lines).
4. **Tab manifest gating** (`02_ui_chrome.js`) — each `TAB_MANIFEST` entry gets `modes: ["mouse"]` or `modes: ["human"]`. Tab bar rebuilds on `SET_MODE`. New human kinase tab entry (~25 lines).
5. **Shared primitive factoring** — extract from `kinase_explorer.js` / `kinase_wiring.js`: NES heatmap-strip cell renderer, multiselect chip wiring, sort/filter table chrome, FDR gate hookup. Both explorers consume the same primitives so visual parity is enforced by construction (~100 lines moved, no new behavior).
6. **Human kinase explorer** (`tabs/kinase_human.js`, new) — table + filter chips + detail panel (~200 lines). Donor multiselect (all 10 selected by default). Default sort: `n_donors_sig` desc, ties broken by `|median_nes_sig_only|` desc. Columns: Kinase, Gene, NES Profile (10-cell donor strip), `|median_nes|`, `n_donors_sig`, `n_donors_up`, `n_donors_down`. FDR slider in header applies; "Donors sig ≥" preset chips (≥1 / ≥5 / ≥10).
7. **Human detail panel** — opens on row click, replaces the panel area entirely (no shared sub-tabs with mouse). Content:
   - 10-donor signed-NES bar, cells colored by per-donor FDR.
   - Recurrence summary line (`n_donors_sig / up / down / median_nes_sig_only`).
   - Per-donor leading-edge substrates (from `mea_perdonor.csv` "Lead subs").
   - Per-donor global-shift row (so AD-15's pre-centering shift is visible in context, but as data, not as a special-cased flag).
   - Site-level delta vector (`stoich_AD_i − mean(stoich_CTRL)`) for the leading-edge sites of the selected kinase — the human analog of measurement trace, constructed from `stoichiometry_matrix.csv` on the human pipeline outputs.

Estimated JS lift: ~350 lines total.

## File touch list

New:

- `alz/viewer/template/js/tabs/kinase_human.js`

Modified:

- `alz/build_unified_viewer.py` — add `build_human_slice()` + payload merge.
- `alz/viewer/template/body.html` — mode toggle in header, new `<div id="tab-kinase-human">` panel.
- `alz/viewer/template/js/01_state.js` — `view.mode`, `SET_MODE`.
- `alz/viewer/template/js/02_ui_chrome.js` — `TAB_MANIFEST` entries gain `modes`, tab bar rebuilds on mode change.
- `alz/viewer/template/js/03_filters_hash.js` — `mode=` hash round-trip.
- `alz/viewer/template/js/tabs/kinase_explorer.js` + `kinase_wiring.js` — factor out shared primitives (no behavior change for mouse).
- `alz/viewer/template/index.html.j2` — `{{ raw('js/tabs/kinase_human.js') }}`.
- `alz/viewer/template/MANIFEST.md` — append entry for `kinase_human.js`.

Untouched:

- `alz/viewer/template/js/tabs/kinase_audit.js` — mouse-only, gated by mode.
- `alz/viewer/template/js/tabs/temporal_v2.js` — mouse-only, gated by mode.
- All `tabs/incytr_*.js` — mouse-only, gated by mode.
- `alz/viewer/template/styles.css` — existing classes (`.data-table`, `.ke-toolbar`, `.detail-card`) cover the human explorer.

## Verification

After implementation:

- `pixi run python alz/viewer/verify_template.py` must still pass (the Jinja-rendered HTML still byte-equivalent to the legacy template once the human additions are mirrored).
- Build on a repo without `outputs/reports/kinase_attribution_human/perdonor/` → mode toggle hidden, mouse viewer byte-equivalent to today's.
- Build with the human directory present → mode toggle visible, Human mode shows the kinase tab populated from the 311 ST + 78 pY kinases.
- Smoke check: clicking a kinase in Human mode opens the human detail panel; switching to Mouse mode shows no carried-over selection.

## Locked decisions

- One HTML deliverable, mode toggle in header chrome.
- Mouse mode: unchanged (5 tabs: Kinase, Temporal, Incytr Heatmap, Incytr Pathways, Methods).
- Human mode: Kinase tab only.
- Kinase IDs are not joined across cohorts in this lift.
- All donors treated identically in the human view; plain donor multiselect, no AD-15 special-casing.
- Default human sort: `n_donors_sig` desc, secondary `|median_nes_sig_only|` desc.
- Mouse-side payload structure unchanged; human slice is additive under `PAYLOAD.human`.
- Build is graceful when human outputs are absent (toggle hidden, artifact unchanged).

## Out-of-scope follow-ups (not part of this lift)

- Cross-cohort comparison tab (kinase scatter / concordance counter).
- Human methods page.
- Human Incytr integration.
- Human cell-type attribution.
