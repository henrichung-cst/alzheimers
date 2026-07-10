# AD gene.use — un-pin from sce4's frozen per-pair node sets

**Status:** DECIDED (2026-07-10) — **un-pin**. AD switches to the derived `DEG∪prG`
recipe, the same path t-cells use. Surfaced 2026-07-07 when sce4 parity was
declared a closed non-goal and the `verify-incytr-sce4` gate was removed; the pin
is the last piece of parity machinery still shaping production AD output.
Production-output-changing — requires a full AD re-run + viewer rebuild after the
switch. Implementation is the "Touch points" list below.

## The pin

The AD cohort does **not** derive its Incytr `gene.use`. It reads sce4's own
reconstructed per-(sender,receiver) node sets from
`data/incytr_frozen/sce4_geneuse/<c1>_<c2>.csv`, consumed via the
`SCE4_GENEUSE_DIR` env var (set only by the AD runner `alz/incytr_pair/run_pair_mode.sh`;
the t-cell runner leaves it unset and derives). Artifacts are built by
`alz/incytr_pair/extract_sce4_geneuse.R` from sce4's pre-cap pairwise RDS.

The driver branch is `use_frozen_geneuse` in `alz/incytr_pair/incytr_commandline.R`
(~L343). When unset, the same driver derives `gene.use = DEG ∪ prG` per contrast
(DEG at `avg_log2FC > 1 & p_val < 1e-4`, prG = `proteomics_gene(style="aFC", cutoff=1, strict=TRUE)`),
which is exactly the t-cell path.

## Why this is now in question

The pin exists **only** to reproduce sce4's exact enumeration — it constrains AD
output to sce4's gene selection. That is parity maintenance. With parity now a
non-goal (baseline was verified, improvements shipped on top, byte-for-byte
sce4 reproduction is no longer a target), the pin is the remaining piece of
parity machinery still shaping production AD results.

The narrowing rule that gets sce4 from `DEG(>1) ∪ prG` to its 182-ligand universe
was never reconstructable from on-disk artifacts (the CellChat ligand-category
candidate was refuted). So the frozen sets are sce4's *effective* gene.use lifted
losslessly from its RDS — not a rule we can re-derive. Un-pinning means AD falls
back to the `DEG ∪ prG` derivation, which is **broader** than sce4's sets
(engine-monotonic: subset ⇒ fewer paths), so AD path counts will grow.

## Rationale

Un-pin is consistent with "parity is a non-goal" — a single gene.use recipe across
cohorts. AD enumeration widens (more candidate paths; the downstream SigProb/|PDS|
filter still applies). The frozen sets are sce4's *effective* gene.use lifted from
its RDS, not a re-derivable rule, so keeping them would only be defensible as a
better selection on independent merits — which is not the case here.

## Touch points

- `alz/incytr_pair/run_pair_mode.sh` — stop exporting `SCE4_GENEUSE_DIR`; drop the
  extract-artifacts block (L51–63).
- `alz/incytr_pair/incytr_commandline.R` — the `use_frozen_geneuse` branch (~L343)
  becomes dead; remove it (anti-shim: no flag left behind).
- `alz/incytr_pair/extract_sce4_geneuse.R` — becomes orphaned; delete.
- `data/incytr_frozen/sce4_geneuse/` — frozen inputs may stay on disk as record.
- CLAUDE.md — the "gene.use selection is cohort-dependent" invariant collapses to
  one recipe; update.
