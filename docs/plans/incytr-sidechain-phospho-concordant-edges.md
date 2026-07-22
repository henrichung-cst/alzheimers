# Terminal edges from direct phosphorylation changes (t-cells)

**Status:** proposal, awaiting review.
**Scope:** t-cell cohort only (`tcells_donor1`) as the pilot. AD cohorts (song,
5xFAD) untouched this pass — see *Non-goals*.

## Problem

The current terminal edge `kinase → pathway-node` exists whenever the kinase's
motif matches a **detected** phosphosite on the node's gene
(`kinase_incytr_bridge.build_substrate_bridge`, floored at
`kl_percentile >= 99`). `kl_percentile` is PSSM match quality, **not** phospho
change. Verified on `tcells_donor1`: a kinase's floor-99 site set is ~85% stable
across the 5 contrasts, so edge existence and `n_sites` are contrast-independent
motif bookkeeping. The only place the measured phospho change enters is the flat,
kinase-level `NES` (GSEA over the fold-change ranking). Per edge, we currently
draw *could-phosphorylate*, not *did-change*.

## Goal

Redefine the terminal edge so it exists only when the kinase's motif-matched
site(s) on the node gene show a **direct, measured, direction-concordant phospho
change** in the contrast. The motif stays as the kinase *attributor* (a moved
site alone cannot name a kinase — motifs are many-to-one); MEA stays as the
kinase *eligibility* gate; the measured per-site change becomes the edge
*existence and strength* criterion.

## Correctness guardrail (why MEA is retained)

A single moved site matches many kinases' PSSMs. "Motif matches a site that
moved" without a kinase-activity gate explodes into promiscuous edges. Concordance
alone does **not** brake this: a kinase with a near-zero, randomly-signed NES that
happens to match a moved site passes the concordance check ~50% of the time.

**Verified (blocker resolved):** the current pipeline does **not** restrict graph
nodes to MEA-significant kinases — the bridge is explicit ("not gated by MEA FDR",
`kinase_incytr_bridge.py:196`) and the viewer only *sizes* nodes by `|NES|`, never
filters them (`incytr_sidechains.js:499`). So the MEA gate is not currently a real
brake. **Fix:** in the new bridge step, make edge eligibility require the kinase's
`(contrast, channel)` MEA `FDR < MEA_FDR_THRESH` (`= 0.25`, the existing canonical
constant in `alz/shared/config.py:87` — no new tunable). Only then does
"concordant" mean "concordant with a real activity call" rather than with noise.

## Threshold method (researched — see rationale below)

Per-site change is called by **Significance B (Cox & Mann 2008)** — the
field-standard replicate-free ratio-outlier test — computed on the MEA-normalized
`stoichiometry_matrix` values, the same scale MEA ranks on. Computed **per
channel** (ST and pY nulls estimated separately — never pooled).

Empirical basis (`tcells_donor1`, Δ = timepoint − d2, log2):
- Samples are already per-timepoint centered (means ≈ −0.1…−0.2) → global shift
  normalized out; a relative null is correct here and stays consistent with NES.
- Robust σ of the null Δ ≈ 1.4 (right) / 1.9 (left) → a fixed LFC cutoff (e.g. 1)
  is inside 1σ of noise, i.e. meaningless. Data-driven cutoff is mandatory.
- Noise is intensity-dependent (mean|Δ| 1.74 low-abundance vs 1.03
  high-abundance tercile) → the **intensity-binned** variant (Sig B), not a
  single global null (Sig A).

Per contrast, per channel, per intensity bin: median `m`, right σ = `P84.13 − m`,
left σ = `m − P15.87`; site `z = (Δ − m)/σ_side`. Significance is the **two-sided**
tail probability `erfc(z/√2)` (test magnitude only — the side is picked by the
site's own Δ, so a one-sided `0.5·erfc` would be data-chosen and run ~2×
anti-conservative through BH). Benjamini-Hochberg across sites within a
contrast×channel; changed at corrected α (default 0.05). **Concordance is applied
as a separate filter after significance**, not folded into the tail choice.

### Timecourse — correlated corroboration, not replication

At n=1 with σ≈1.5, a single-contrast call is weak. The 5 baseline contrasts
(d13/15/17/19/20 vs d2) corroborate a call, but they are **not** independent
replicates: they share the same d2 reference (common-mode error) and the dX states
are genuinely different biological conditions, so a site can truly move at one
timepoint and not another. Store, per site, in how many of the 5 contrasts it is a
significant concordant outlier (`timecourse_consistency`, 0–5) as a credibility
signal. It is **stored for analysis, not gated** this pass.

### Concordance (sign convention resolved)

Verified against the MEA source: the enrichment ranks on `log2_fold_change`
(`enrich.py:308`), which is `value(dX) − value(d2)` with `+ = up at the later day`
(`alz/cohorts/tcells/mea.py:12-13`, `_build_timepoint_deltas`), fed to
kinase-library's prerank MEA. So **`NES > 0` = the kinase's substrate motifs are
enriched among sites up at dX vs d2 = kinase inferred more active at dX.** Δ is
defined the same way (`dX − d2`). Therefore:

`concordant = sign(site Δ) == sign(signed_nes)` — no ambiguity, no flip.

## Data flow changes

```
kinase_incytr_bridge.py   →  kinase_kinase_edges.py     →  payload_helpers.py       →  incytr_sidechains.js
(per-site_id Δ, Sig B,        (edge exists iff             (+ edge_delta,               (edge_delta drives width;
 concordance, timecourse       n_significant_concordant≥1;   n_significant_concordant)     panel: Δ, sig, concordance,
 — per channel)                strength = edge_delta)                                     timecourse)
```

### Site grain — carry `site_id` through (decision)
The bridge currently keys the motif→gene map on the 13-mer motif, deduped on
`(motif_key, gene_symbol)` (`kinase_incytr_bridge.py:234`), and `sites` records
carry no `site_id`. One motif can map to several physical phosphosites with
different Δ, so per-site Δ is undefined at motif grain. **Fix:** thread `site_id`
+ `site_position` through the motif→gene map so `sites` records are
**per-physical-site**. `n_sites` becomes the distinct-`site_id` count (was:
distinct-motif count); this changes existing values but **preserves
`len(sites) == n_sites`** — the load-bearing invariant enforced at
`kinase_incytr_bridge.py:289`, `kinase_kinase_edges.py:220`,
`incytr_sidechains.js:529`, and foundation doc L227. Do not break it.

### 1. `alz/cross_reference/kinase_incytr_bridge.py`
- Load abundance columns: the stoich matrix is read with a pruned `usecols`
  (`site_id, gene_symbol, motif`); **add `D1_d2 … D1_d20`** so Δ can be computed.
- Motif→gene map carries `site_id`, `site_position` (per-site grain, above).
- New step after `build_substrate_bridge`, before `gene_node_hits`: for each
  `(kinase, contrast, gene_symbol, site_id)` compute `delta = dX − d2` from the
  two contrast columns.
- Significance B **per contrast × channel**: intensity bins over site mean
  abundance, robust asymmetric null, **two-sided** tail, BH within
  contrast×channel. Attach `site_significance`. ST and pY nulls never pooled.
- Edge eligibility gate: keep only kinases whose `(contrast, channel)` MEA
  `FDR < MEA_FDR_THRESH` (0.25). This is the promiscuity brake (see guardrail).
- `concordant` per site = `sign(delta) == sign(signed_nes)` (convention resolved).
- `timecourse_consistency` per site across the 5 contrasts (bridge already loads
  the full substrate-set file for all contrasts).
- Per-site `sites` JSON entries gain `site_id`, `site_position`, `delta`,
  `site_significance`, `concordant`; keep `motif`, `residue_type`, `kl_percentile`.
- **`n_sites` stays = `len(sites)`** (all matched physical sites). Add a new field
  `n_significant_concordant`. Emit a row only when `n_significant_concordant ≥ 1`;
  non-moving / discordant sites stay in `sites` with their flags — stored, not
  dropped.

### 2. `alz/cross_reference/kinase_kinase_edges.py` (`load_motif_edges`,
`build_terminal_map`)
- `ROW_NUMBER()` max-|NES| pick unchanged (one row per
  kinase×gene×role×contrast×owning_cluster); `abs(signed_nes) == best_abs_nes`
  preserved.
- Carry `n_significant_concordant` through, and add one edge-level aggregate
  **`edge_delta`** = aggregate Δ over the significant-concordant sites (mean vs
  max: decide in impl). `min_site_significance` / `max_timecourse_consistency`
  stay inside the per-site `sites` JSON only — **not** promoted to columns (every
  survivor is significant by construction; timecourse is not gated). Fewer
  schema-guard columns.
- Interactome builder untouched (`n_sites` still terminal-only).

### 3. `alz/viewer/shared/payload_helpers.py`
- Append **`n_significant_concordant`, `edge_delta`** to
  `_INCYTR_SIDECHAIN_TERMINAL_COLUMNS` (schema tuple guard forces regeneration —
  anti-shim, same pass). Interactome tuple unchanged.

### 4. `alz/viewer_shared/template/js/tabs/incytr_sidechains.js`
- Node size unchanged (`_isEmphasis(|NES|)`). **Terminal edge width driver moves
  from `n_sites` to `edge_delta` magnitude** (decision), with a specified
  transform (reuse the `_isEmphasis` clamp/γ family; pin lo/hi anchors from the
  Δ distribution). The static legend + the foundation encodings table must be
  updated in the same pass — width now means "how much the sites moved", not
  motif multiplicity.
- Terminal edge detail panel + node-relation table: add measured Δ, per-site
  significance, concordance, and timecourse-consistency rows. Per-site evidence
  table gains Δ / significance / concordant / site_position columns.

### 5. `docs/foundation/kinase_sidechain_incytr_graph.md`
- Invert the documented edge-creation invariant in the same pass:
  "motif-created, PSP-corroborated" → "measured-change-created, motif-attributed,
  MEA-gated, PSP-corroborated". Keep the `len(sites) == n_sites` invariant (L227)
  — it still holds. Update the edge-strength model (width = `edge_delta`) and the
  encodings table.

### 6. `tests/test_kinase_sidechain_weighting.py`
- `len(sites) == n_sites` assertion stays. Add: Significance B robust-null +
  two-sided BH unit cases (seeded synthetic Δ), concordance sign test, edge
  dropped when `n_significant_concordant == 0`, and width-from-`edge_delta`
  emphasis math.

## Non-goals / anti-shim

- **No dual mode.** The could-phosphorylate edge definition is *replaced*, not
  kept behind a flag. No "motif-only fallback".
- **AD cohorts (song, 5xFAD) this pass.** Their stoich matrices carry condition
  columns too, and 5xFAD has replicates (→ a *stronger*, properly-tested DE call,
  not Sig B). Uniform cross-cohort criterion is a follow-up decision, not a
  silent per-cohort branch. T-cells prove the model first.
- Timecourse consistency is **stored, not gated** this pass.

## Impl-time parameters (not blocking)
- `edge_delta` aggregation over significant-concordant sites: mean vs max.
- Significance B α (default 0.05) and intensity-bin count.
- Width transform anchors (lo/hi) for `edge_delta` emphasis.

## Verify
```
pixi run python -m pytest tests/test_kinase_sidechain_weighting.py tests/test_motif_peer_narrowing.py -q
pixi run python -m alz.cross_reference.kinase_incytr_bridge --cohort tcells
pixi run python -m alz.cross_reference.kinase_kinase_edges  --cohort tcells
pixi run python -m alz.build_tcell_viewer --html
pixi run python -m alz.viewer.verify_payload_contract \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json
```
Then preview the edge-count drop (per-kinase fan sizes before/after) before
committing — a large collapse is an expected, reportable result, not a bug.
