# T-cell: two columns — cell-type specificity and state enrichment

**Status:** proposed, no code touched. Supersedes the relabel-only
`tcell_cell_state_vs_cell_type.md` (that plan honestly relabels the state columns
and adds an NSCLC cross-lineage column, but computes no new metric and does not
fix the erroneous kinases found in the audit). This plan keeps that relabeling and
adds the substance: a real within-cohort **cell-type** specificity metric and
**guards** on the state metric.

## Why

The T-cell cohort currently has **one** localization number per kinase —
`tcell_state_enrichment` = expression in a state ÷ the kinase's median state,
over a flattened set of the 14 ProjecTILs labels. Two problems, both confirmed
against a true-positive exhaustion-kinase audit:

1. **It conflates two different questions.** The 14 ProjecTILs labels are a grid
   of **cell type** (CD8 / CD4 / Treg) × **state** (naive / memory / effector /
   exhausted). A single flattened metric can answer neither cleanly. There is no
   within-cohort cell-type specificity at all today.
2. **The state number is unguarded, so it reports noise as signal.** Measured on
   well-characterized exhaustion kinases:
   - **SYK** → "6.9× enriched in exhausted CD4" while being **undetected in every
     T-cell state** (the fold divides near-zero by near-zero).
   - **AKT3, ZAP70, FYN, MAP4K1, JAK1** → top enrichment reported in **CD8.MAIT**,
     a state with **8 cells** in donor1 — an unreliable average.

Cross-cohort context: **Song** and **5xFAD** both compute cell-type specificity
with the shared `specificity.compute()` (concentration / `concentration_tier` /
`effective_n`), with the broad cell-type class as the curated unit, and both feed
the confidence pill a **unit-level** `effective_n` (`song_unit_effective_n`,
`fivexfad_unit_effective_n`). 5xFAD explicitly *retired* its bespoke fold
localizer ("one metric, every cohort"). T-cell is the lone cohort that (a) never
adopted the shared cell-type metric and (b) feeds the pill the **raw** per-state
`effective_n` — the exact "raw eff is wrong" mistake that was bug-fixed in 5xFAD.

## Outcome — two columns

| Column | Question | How computed | Cross-cohort status |
|---|---|---|---|
| **Cell-type specificity** | Is the kinase concentrated in CD8 / CD4 / Treg? | Collapse the 14 states to cell types; shared `concentration_tier` + `effective_n` over cell types (cell type = the curated unit, exactly Song/5xFAD) | **Same shared formula** — no divergence |
| **State enrichment** | Is the kinase higher in exhausted cells than in a typical state? | Existing fold-over-median, but computed only over **reliable** states (detected + enough cells) | T-cell-only axis (brain cohorts have no "state"); nothing to diverge from |

NSCLC stays as the **cross-tissue** corroborator ("is this a T-cell kinase at all,
vs B / myeloid / epithelial?") — the relabel-only plan's NSCLC "Cell types" column
is retained as that separate layer.

## Decisions (confirmed)

1. **Cell-type map** — `CD8.* → CD8`; `CD4.{NaiveLike, Tfh, Th17, CTL_EOMES,
   CTL_GNLY, CTL_Exh} → CD4`; `CD4.Treg → Treg`. Three cell types; CD8.MAIT (n=8)
   folds into CD8 and so can never stand alone.
2. **Reliability threshold for the state metric** — a state needs **≥ 50 cells**
   to be eligible for the state-enrichment median and the top-state pick. Drops
   only CD8.MAIT. Residual: AKT3 (CD4.Tfh, 311) and FYN (CD8.Naive, 102) still
   show ~2× / ~1.9× on the state column; the cell-type column reads "broad" for
   both, so the conclusion holds. `MIN_STATE_CELLS = 50` as a named constant.
3. **Detection gate on the state metric** — a state must have
   `fraction_cells_expressing ≥ 0.10` (the repo-standard gate) to count toward the
   median or be eligible as the top state. This is what kills the SYK phantom.

## Build — pipeline (`alz/cross_reference/tcell_within_cohort.py`)

The state-resolution `specificity.compute()` call (≈ line 191) stays — it still
produces per-state `detected`, `fraction_cells_expressing`, `concentration`, and
`linear_expression`. Add two things:

1. **Cell-type specificity (new).** Map each state to its cell type (decision 1),
   then collapse the per-state `concentration` shares onto cell types and compute
   `effective_n` over cell types — the same mechanism Song/5xFAD use via
   `alz/bulk_mea/specificity_class.py::unit_concentration_shares` /
   `unit_effective_n`. That helper currently hard-codes the brain unit map
   (`config.load_specificity_unit_map()`); generalize its signature to accept a
   `unit_map` argument (default = the brain map, so Song/5xFAD are byte-unchanged)
   and pass the T-cell cell-type map. Emit per kinase:
   - `tcell_celltype_concentration_tier` — fold of the top cell type's share over
     the even `1/N_celltypes` share (the standard tier bins 1/2/5/10×).
   - `tcell_celltype_effective_n` — `1 / Σ celltype_share²`.
   - `tcell_top_celltype` — the dominant cell type (CD8 / CD4 / Treg). **This
     replaces** the current `tcell_top_celltype`, which is today a *state*
     argmax (a category error — it names a state in a column called "top
     celltype").
2. **Guard the state enrichment (replace).** Recompute `tcell_state_enrichment`
   over the eligible set only — states with `detected == True` **and**
   `n_cells ≥ <threshold>` (decisions 2–3). The median is taken over eligible
   states; ineligible states get `tcell_state_enrichment = NaN` (rendered as no
   badge). A kinase with no eligible state has no state enrichment. The current
   unguarded computation (lines ≈ 219–221) is removed, not kept alongside.

Output columns of `tcell_specificity.csv` / `unified_attribution_tcells.csv` gain
`tcell_celltype_concentration_tier`, `tcell_celltype_effective_n`; the existing
`tcell_state_enrichment` semantics change (now guarded); `tcell_top_celltype`
changes meaning (now a cell type, not a state).

## Build — viewer assembly (`alz/build_tcell_viewer.py`)

- **Confidence pill** (`kinase_meta` / `_tcell_exclusivity_tier`, ≈ lines 635–661):
  feed `meta["eff"]` from **`tcell_celltype_effective_n`**, not raw
  `tcell_effective_n`. This is the 5xFAD `unit_effective_n` fix. The NSCLC
  corroboration crosswalk keys off the cell-type home (CD8/CD4/Treg all map to
  NSCLC `T_NK`), simplifying `_TCELL_STATE_TO_NSCLC` to a cell-type → `T_NK` map.
- **Per-kinase slice** (`_build_donor_kinases_slice`, ≈ lines 835–900): the
  current sort/headline on `tcell_state_enrichment` (≈ line 892) keeps the
  **guarded** state enrichment as the "State specificity" headline; add the
  cell-type columns (`tcell_celltype_concentration_tier`, top cell type) as a
  parallel per-kinase slice for the new Explorer column.

## Build — templates / JS

Fold in the honest relabeling from `tcell_cell_state_vs_cell_type.md` (Cell type →
Cell state, Specificity → State specificity on the state-derived columns), then:

- **Kinase Explorer** (`body.html` + `js/tabs/kinase_explorer.js`): add a
  **Cell-type specificity** column (top cell type badge + tier) sourced from the
  new slice fields. Keep the separate NSCLC cross-lineage column from the
  relabel plan if that is also wanted (decision: two cell-type-ish columns —
  within-cohort CD8/CD4/Treg and cross-tissue NSCLC — or just the within-cohort
  one).
- **Attribution detail** (`js/tabs/attribution_manifest_tcell.js`): the
  `_tcellEnrichCell` badge (≈ lines 28–39) must **gate on detection** — render
  "—"/muted for a row whose `tcell_detected` is false, regardless of the fold
  value. (Today it renders the fold unconditionally, which is how SYK got a green
  "6.9× strong" badge on an undetected row.) Add a per-kinase cell-type-specificity
  line to the §0 verdict.

## What is removed (anti-shim)

- The unguarded `tcell_state_enrichment` computation — replaced by the guarded one,
  not kept behind a flag.
- The raw `tcell_effective_n` feed into the confidence pill — replaced by
  `tcell_celltype_effective_n`.
- `tcell_top_celltype` as a *state* argmax — replaced by the cell-type argmax.
  (The state a kinase is most enriched in is still available from the per-state
  table; it is no longer mislabeled "top celltype".)

## Verification — the audit IS the acceptance test

Re-run the true-positive exhaustion-kinase check and confirm the before/after:

| Kinase | Old (broken) | New cell-type col | New state col (guarded) |
|---|---|---|---|
| SYK | 6.88× "strong", undetected | CD4, broad | **none — not detected** |
| AKT3 | CD8.MAIT(8) 3.33× | CD8, broad | CD4.Tfh(311) ~2.3× |
| ZAP70 | CD8.MAIT(8) 2.60× | CD8, broad | CD8.TEX 1.47× |
| FYN | CD8.MAIT(8) 2.59× | CD8, broad | CD8.Naive(102) 1.89× |
| MAP4K1 | CD8.MAIT(8) 2.14× | CD8, broad | CD8.TEX 1.31× |
| JAK1 | CD8.MAIT(8) 1.68× | CD8, broad | CD4.CTL_Exh 1.24× |
| GSK3B / ITK / LCK | 1.3–1.7× (already fine) | broad | unchanged |

Acceptance: SYK shows no state badge; no kinase's headline state is CD8.MAIT;
every pan-T kinase reads "broad" on the cell-type column (`effective_n` ≈ 2–2.5
over 3 cell types, tier 1).

Then: `ast.parse` the edited Python, `node --check` the edited JS, rebuild under a
memory cap (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 pixi
run tcell-viewer`), hard-refresh, and eyeball the two columns in the Kinase
Explorer.

## Out of scope

- The NSCLC corroboration modality-mismatch (low-transcript ST kinases like CDK8 /
  MTOR / PDPK1 / SGK fall below the 10% transcript gate in NSCLC `T_NK` despite
  strong activity signal, so the pill caps at "moderate"). Independent finding;
  not touched here.
- Song / 5xFAD outputs — unchanged. The only shared-code edit is adding an
  optional `unit_map` argument to `unit_concentration_shares` with the brain map
  as default, so their results are byte-identical.
