# T-cell viewer: lift `alz/viewer/` verbatim, shape T-cell data to its contract

**Date:** 2026-05-29
**Status:** Plan — awaiting approval before any code edits

## Why this plan exists

Three attempts. The first two greenfielded thin viewers; the third (this session) deleted 11 of the 14 shared modules. All three violated the standing rule (`feedback_no_reimplementing_shared_viewer.md`): **port means lift verbatim — never rewrite to dodge the PAYLOAD/Store/TAB_MANIFEST contract**. The recurring failure mode is the same: I look at T-cell data, conclude "the mouse modules don't fit", and rewrite. The correct response is to make the **data** fit the contract, not rewrite the contract.

This plan does not invent anything. It defines:
1. The lift (what files copy verbatim).
2. The data shaping (how T-cell artifacts become the same PAYLOAD keys the modules already read).
3. The two real divergences from mouse (donor toggle replaces genotype toggle; donor2 has no kinase MEA, gate accordingly).

## State of the world (verified 2026-05-29)

### Source modules — `alz/viewer/` (intact, untouched by recent work)
```
alz/viewer/template/
  index.html.j2                          (32 lines, jinja2 concat)
  body.html                              (349 lines)
  js/01_state.js  02_ui_chrome.js  03_filters_hash.js
  js/04_slice_cache.js  05_header.js  boot.js
  js/widgets/multiselect.js  evidence_row.js  transcript_trace.js  sequence_logo.js
  js/tabs/temporal_v2.js  kinase_explorer.js  kinase_audit.js  kinase_detail.js
  js/tabs/kinase_wiring.js  kinase_human.js  kinase_crosstable.js
  js/tabs/incytr_heatmap.js  incytr_pathways.js  incytr_state.js
alz/build_unified_viewer.py              (3056 lines — payload assembler)
```

### Source build emits these PAYLOAD top-level keys
```
kinases, kinase_motifs, celltypes, audit_tables,
edge_slice_ref, incytr_pathways, meta
```

### T-cell data inventory

| Artifact | Donor1 | Donor2 |
|---|---|---|
| Bulk MEA (`kinase_attribution_tcells/<d>/mea/*`) | **full** (NES/FDR/timecourse/substrate_sets, both pSTY & pY) | manifest only — **no kinase outputs** |
| Pair-mode Incytr (`incytr_pair_mode_tcells/<d>/wide/*.parquet`) | d13_d2, d17_d2, d20_d2 | d5_d2, d7_d2, d9_d2, d11_d2 |
| Per-pair Incytr shards (`tcell_viewer/edge_slices/incytr_pathways/`) | yes (donor1__ prefix) | yes (donor2__ prefix) |
| Cell-type decomposition | **none** (cohort memory: "MEA on bulk only, never deconvoluted") | none |
| scRNA / ProjecTILs annotations | yes (separate plan) | yes (separate plan) |
| Human kinase data | n/a | n/a |

### Implications (load-bearing)

- **Kinase tabs are donor1-only.** Donor2 has no MEA outputs. Tabs gate via `requires:[]` on payload presence + donor selection.
- **No `celltypes` payload.** No T-cell deconvolution exists or is planned. `kinase_crosstable.js` (kinase × celltype) has no data and must be excluded from the T-cell `TAB_MANIFEST` — not deleted from the module set, just not registered.
- **No `kinase_human` tab.** No human T-cell phospho data on this box. Same handling: not registered in T-cell manifest.
- **`block.version` stays at 1** unless we explicitly emit v3-shaped contrast metadata. Trajectory chips in `incytr_pathways.js` are gated on `>=3`; leaving them gated off is honest.

## Scope: what the T-cell viewer ships

Tabs registered in T-cell `TAB_MANIFEST` (subset of mouse, none invented):

| Tab id | Module | Donor1 | Donor2 | Gate |
|---|---|---|---|---|
| `temporalv2` | `temporal_v2.js` | yes | empty | `PAYLOAD.kinases` present **and** `selection.donor === "donor1"` |
| `kinaseexplorer` | `kinase_explorer.js` | yes | empty | same |
| `kinaseaudit` | `kinase_audit.js` | yes | empty | same |
| `kinasewiring` | `kinase_wiring.js` | yes | empty | same |
| `incytrheatmap` | `incytr_heatmap.js` | yes | yes | `PAYLOAD.incytr_pathways` present |
| `incytrpathways` | `incytr_pathways.js` | yes | yes | always |

Tabs NOT registered (modules still ship, just no manifest entry):
- `kinase_crosstable.js` — no celltypes payload
- `kinase_human.js` — no human data
- `kinase_detail.js` — drilldown invoked by `kinase_explorer`, not its own tab

Donor2's kinase tabs render the empty-state message via `requires` failing — mouse code path already handles this (it's how mouse-mode hid human-only tabs when no human payload existed).

## The lift (Step 1 — mechanical copy)

```
cp -r alz/viewer/template/js/tabs/*     alz/tcell_viewer/template/js/tabs/
cp -r alz/viewer/template/js/widgets/*  alz/tcell_viewer/template/js/widgets/
cp    alz/viewer/template/js/01_state.js     alz/tcell_viewer/template/js/
cp    alz/viewer/template/js/02_ui_chrome.js alz/tcell_viewer/template/js/
cp    alz/viewer/template/js/03_filters_hash.js alz/tcell_viewer/template/js/
cp    alz/viewer/template/js/04_slice_cache.js  alz/tcell_viewer/template/js/
cp    alz/viewer/template/js/05_header.js       alz/tcell_viewer/template/js/
cp    alz/viewer/template/js/boot.js            alz/tcell_viewer/template/js/
cp    alz/viewer/template/index.html.j2         alz/tcell_viewer/template/
cp    alz/viewer/template/body.html             alz/tcell_viewer/template/
```

After this, `alz/tcell_viewer/template/` is byte-identical to `alz/viewer/template/`. No edits yet.

## The targeted edits (Step 2 — only what genuinely differs)

These are the *only* places T-cell deviates from the verbatim lift. Each is justified by a real data-shape difference, not aesthetics.

### Edit A — `template/body.html`: donor toggle replaces mode toggle
- The mouse mode-toggle (App/Tau/ApTt) is meaningless here. Replace the `<div id="mode-toggle">` block with `<div id="donor-toggle">` carrying two buttons (donor1/donor2). Keep the same CSS class (`mode-btn`) so the existing visual treatment applies.
- Title: "T-cell Pathway Viewer".
- Everything else (filter bar, glossary, tab panels, all `tab-*` divs) **kept**. Tabs not in the T-cell manifest are simply not built into the tab bar; their panels sit dormant — same mechanism mouse uses to hide human tabs without a human payload.

### Edit B — `template/js/05_header.js`: wire donor toggle
- Add `_syncDonorToggle()` calling pattern (same shape as the existing `_syncModeToggle`).
- Keep mode-toggle code path; gate its DOM lookup on `document.getElementById("mode-toggle")` returning null → no-op. This is not a shim — it's the same defensive null-check pattern the source already uses for human tab buttons.

### Edit C — `template/js/01_state.js`: state slice
- Add `selection.donor: "donor1"` default.
- Keep `selection.mode` — never read in T-cell because the T-cell `TAB_MANIFEST` doesn't gate on it.
- Add `SET_SELECTION` handler for `donor` (same shape as existing `mode` handler).

### Edit D — `template/js/02_ui_chrome.js`: T-cell `TAB_MANIFEST`
- Replace `TAB_MANIFEST` with the 6-tab table above.
- Add `requires` entries:
  - Kinase tabs: `["kinases", "donor:donor1"]` (extend the existing `requires` resolver to recognize `donor:` prefix — one new branch, ~5 lines).
  - Incytr tabs: `["incytr_pathways"]` (already supported).
- `TAB_GROUP_ORDER`: `["kinase", "incytr"]`.

### Edit E — `template/js/03_filters_hash.js`: hash codec
- Add `d` (donor) to hash schema. `t` (tab) and `m` (mode) already there; leave `m` since the codec is mode-agnostic.

### Edit F — `template/js/04_slice_cache.js`: per-donor shard prefix
- The T-cell shard naming is `{donor}__{sender}__{receiver}.parquet` (already on disk). Update `loadIncytrShard` to prepend `Store.state.selection.donor` and read `slice_index.by_donor.<d>.present` for the gate.
- All other cache buckets (backbone, decomp_ols, song concordance, human substrate) remain — they just won't be exercised because the T-cell payload doesn't populate `edge_slice_ref` for them. No code deletion.

### Edit G — `incytr_pathways.js`: dynamic disease/timepoint vocab
- Replace constant `_IP_DISEASES = ["App","Tau","ApTt"]` with `_ipDiseases(block)` deriving from `block.by_donor[d].contrasts` (mouse keeps `_IP_DISEASES` if it falls back to constants when `block.by_donor` absent).
- Same for `_IP_TIMEPOINTS`.
- Palette: small qualitative palette derived from contrast count (existing `_mousePalette` kept as fallback for mouse).

**Total surface area of edits B–G:** roughly 80–120 lines across 6 files, all narrow and isolated.

## The build script — `alz/build_tcell_viewer.py` (Step 3)

Rewrite to follow `build_unified_viewer.py`'s payload shape exactly. The PAYLOAD contract is the contract; we emit the same keys.

```
PAYLOAD = {
  "kinases":          [donor1 kinase list] if donor1 MEA present else [],
  "kinase_motifs":    {donor1 motifs} else {},
  "celltypes":        [],                          # no T-cell decomposition
  "audit_tables":     {donor1 audit manifest} else {},
  "edge_slice_ref": {
    "incytr_pathways_url": "edge_slices/incytr_pathways/",
    # NO backbone / decomp_ols / human_perdonor_substrate URLs — donor2 N/A
  },
  "incytr_pathways": {
    "block": {
      "version": 1,
      "by_donor": {
        "donor1": {"contrasts": ["d13_d2","d17_d2","d20_d2"], ...},
        "donor2": {"contrasts": ["d5_d2","d7_d2","d9_d2","d11_d2"], ...},
      },
    },
    "slice_index": {"by_donor": {"donor1": {"present":[...]}, "donor2": {...}}},
    "ranked_rows": [...],
  },
  "meta": {
    "cohort": "tcell",
    "donors": ["donor1","donor2"],
    "days_by_donor": {"donor1":["d13","d17","d20"], "donor2":["d5","d7","d9","d11"]},
    "baseline": "d2",
    "tracks": ["pSTY","pY"],          # donor1 only; rendered tabs gate on this
    "generated_at": "<ISO timestamp>",
  },
}
```

Sources for each key:
- `kinases`: `outputs/reports/incytr_pair_mode_tcells/donor1/wide/kinase_timepoint_nes.csv` column header.
- `kinase_motifs`: kinase-library catalog filtered to donor1 kinases (same path as mouse).
- `audit_tables`: `mea_substrate_sets.csv`, `mea_global_shift.csv`, `mea_manifest.json` from donor1.
- `incytr_pathways.ranked_rows`: union of donor1+donor2 wide parquets, top-N by |PDS| per donor (whatever cap mouse uses).
- Slice shards already at `outputs/reports/tcell_viewer/edge_slices/incytr_pathways/{donor}__{s}__{r}.parquet` — no regeneration needed.

The mouse build's `build_audit_manifest`, `kinase_activity`, motif lookup, ranked-rows, and slice-index helpers should be **imported or copy-paste-with-attribution** from `build_unified_viewer.py`, not re-derived. Where mouse code reads three genotypes, T-cell reads one donor — that's a narrow loop variable change, not a rewrite.

## What this plan explicitly does NOT do

- Does not invent a "T-cell trajectory chart". Trajectory chips are gated on `block.version >= 3`; we ship v1 and let them stay hidden until a v3 schema exists.
- Does not delete `kinase_crosstable.js`, `kinase_human.js`, `kinase_detail.js`, or any widget. They stay on disk, unused, exactly as the source ships them. Anti-shim does not require deleting files that the manifest simply doesn't reference — those are dead routes in *this* viewer, not dead code in the shared module set.
- Does not add a "donor2 kinase coming soon" placeholder anywhere. The kinase tabs simply unmet-prerequisite-render for donor2, using the existing `renderUnmetPrerequisite` path with message "no kinase MEA for donor2".
- Does not modify `alz/viewer/` source modules. The lift is one-way.

## Verification (read in order)

1. `diff -r alz/viewer/template/js/tabs/ alz/tcell_viewer/template/js/tabs/` → empty (verbatim).
2. `diff -r alz/viewer/template/js/widgets/ alz/tcell_viewer/template/js/widgets/` → empty.
3. `pixi run tcell-viewer` → exits 0; emits `outputs/reports/tcell_viewer/index.html` and payload JSON.
4. `node --check` over the concatenated JS bundle → no syntax errors (existing harness in `verify_template.py`).
5. Browser open + hard refresh:
   - donor1 selected: all 6 tabs visible. Click each, no console errors.
   - Switch to donor2: kinase tabs show "no kinase MEA for donor2" via `renderUnmetPrerequisite`. Incytr tabs render with d5/d7/d9/d11 contrasts.
   - URL hash carries `d=donor2` when switching; reload restores.
6. PAYLOAD spot-check in devtools: `PAYLOAD.meta.cohort === "tcell"`, `Object.keys(PAYLOAD).sort()` matches mouse keys set.

## Order of operations

1. Approve this plan.
2. `cp -r` lift (Step 1). One commit: `feat(tcell-viewer): lift unified_viewer modules verbatim`.
3. Edits A–G (Step 2). One commit per edit, or one combined commit `feat(tcell-viewer): wire donor toggle + dynamic contrast vocab`.
4. Rewrite `build_tcell_viewer.py` against PAYLOAD contract (Step 3). One commit: `feat(tcell-viewer): emit payload matching unified contract`.
5. Verification pass (above). If any step fails, fix root cause — do not gate behind a flag.

## Why this won't be attempt #4

The failure mode of the prior three attempts was: I treated "this data doesn't fit" as a signal to rewrite the viewer. This plan inverts the question: the viewer is fixed (it's the existing source under `alz/viewer/`), and the build script's job is to make T-cell data look like the payload the viewer expects. Donor toggle, donor2's missing kinase data, and the day-vs-timepoint vocab are the only three real divergences; everything else stays verbatim. If during implementation I notice "the existing mouse code path doesn't quite work for T-cells" the answer is to **emit the right data shape**, not edit the viewer module.
