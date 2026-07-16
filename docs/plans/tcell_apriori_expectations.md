# Plan — T-cell exhaustion a priori expectations (validation anchor)

**Status:** plan, awaiting approval before the anchor doc is written.
**Produces:** `docs/reference/tcell_apriori_expectations.md` — a dated, blind, literature-grounded
prediction set committed *before* inspecting current pipeline outputs, used to score whether the
analysis recovers known T-cell exhaustion biology. Feeds `/check-controls`-style sanity checks.

## Discipline (non-negotiable)

- **Strictly literature-grounded, blind.** Every prediction stated qualitatively — direction,
  relative ordering, presence/absence — and justified from published exhaustion biology only.
  No pipeline numbers are cited. The TEX percentages already visible in the summary doc are
  treated as unseen.
- **Pinned.** Every prediction names the pipeline artifact it will be scored against. A prediction
  with no observable surface is cut.
- **Model-honest framing.** This is an **in vitro, chronically stimulated** T-cell exhaustion time
  course with **proliferation induced in silico** — not in vivo tumor/chronic-infection exhaustion.
  Expect canonical *directional* signatures, not necessarily the full terminal-exhaustion
  transcriptional lockdown. Cell cycle is excluded by design, so proliferation does not confound
  the state axis.
- **Scoring is out of scope for this doc.** The anchor states predictions only; agree/off/borderline
  verdicts happen later against outputs.

## Surfaces in scope

Composition, within-cohort attribution (kinase controls folded in), Incytr signaling axes.
Standalone kinase-MEA NES trajectories are **out** — kinase expectations live inside attribution
(which kinase *localizes to* which state).

## Structural limits that constrain what can be predicted (must appear in the doc)

- **Composition is relative, not absolute** — label count / all retained cells (same donor+day).
  No expansion/contraction claims in absolute cell numbers.
- **No per-state significance/FDR** — one donor, one library per day; per-cell tests would be
  pseudoreplication. Predictions are directional/ordinal, not inferential.
- **CD4 substates are near-chance** in this cohort's per-cell resolution (cytotoxic/exhaustion axes
  resolve; fine CD4-helper states do not). Composition predictions are scoped to robust axes only.
- **Incytr checkpoint receptors reconstruct only as Target**, not as receptors — a T-cell-only
  cohort cannot rebuild the receptor role. A naive "PD-L1→PD-1 pathway lights up" prediction would
  fail for a data-structure reason, not biology. Checkpoint predictions are phrased around what is
  actually reconstructable.
- **Detection ≠ activity.** Kinase attribution predictions are about transcript **detection
  localization**; concordance is context, never a validator.
- **Donor 2 has no kinase surface** (no IMAC). Its absence is expected, not a miss.

## Committed predictions

### A. Cell-state composition — pinned to per-cell label composition tables

- **C1.** Exhausted (TEX) relative fraction increases across the time course (late days > d2).
  *Basis:* chronic antigen stimulation drives progressive exhaustion (Wherry & Kurachi 2015; Blank
  et al. 2019 consensus nomenclature).
- **C2.** Activated/effector states hold the largest relative share early (near d2) and decline in
  share as TEX expands. *Basis:* acute activation precedes exhaustion differentiation.
- **C3.** A resting/memory compartment persists as a distinct late population rather than every cell
  converting to TEX. *Basis:* memory/precursor (TCF7+ TPEX-like) persistence under chronic stim;
  here it maps to the resting/memory label (TPEX not separately reported).
- **C4.** CD8 exhaustion is more pronounced than CD4 at matched late days (CD8 TEX share > CD4 TEX
  share). *Basis:* canonical exhaustion is CD8-centric; CD4 exhaustion is weaker/less defined.
- **C5.** Both donors show the same *direction* (TEX up over time) despite differing rate/magnitude.
  *Basis:* directional reproducibility of the exhaustion program across individuals.

### B. Within-cohort attribution (donor 1 only) — pinned to `unified_attribution_tcells.csv` detection/evidence

- **A1.** TCR-proximal tyrosine kinases — LCK, ZAP70, ITK, TEC family — detected across T cells and
  their activity localizing to activated/effector rather than exhausted states. *Basis:* proximal
  TCR signaling peaks in activated effectors and is attenuated in exhaustion.
- **A2.** Cytokine/JAK kinases (JAK1/2/3, TYK2) detected broadly and localizing to activated/effector
  (IL-2 / common-γ-chain, IFN response); blunted in TEX. *Basis:* exhausted cells have reduced
  cytokine responsiveness.
- **A3.** Internal-consistency control — the TEX marker program (PDCD1, HAVCR2/TIM-3, LAG3,
  ENTPD1/CD39, TOX, NR4A1) is detected preferentially in the TEX state vs activated. *Basis:*
  definitional; scored as a coherence check on detection localization, not a discovery.

### C. Incytr signaling axes — pinned to incytr pair-mode wide outputs (both donors)

- **I1.** Co-inhibitory / checkpoint ligand and receptor genes are **present as reconstructable
  nodes** (as ligand/EM/Target roles the cohort can build), and their pathway PDS trends upward over
  the time course. Phrased around reconstructable roles because receptors surface only as Target.
  *Basis:* exhaustion shifts the signaling balance toward coinhibition.
- **I2.** Costimulatory and effector-cytokine signaling (IL-2, common-γ-chain, IFN-γ autocrine)
  present and strongest early, ceding relative PDS to inhibitory axes late. *Basis:* costim→coinhib
  shift over exhaustion.
- **I3.** Cross-donor: the same qualitative axis directions hold wherever both donors carry the genes.

## Definition of done for the anchor doc

Each prediction has: a stable id, a one-line literature basis, the pinned artifact, and a plain-text
scoreable statement. The doc is dated, and carries the structural-limits section verbatim so a future
scorer does not re-derive the caveats.
