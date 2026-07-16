# T-cell Kinase-MEA — A Priori Activity Expectations

**Predictions committed:** 2026-07-13, before reading any NES/FDR value on the kinase-MEA surface.
**Scoring:** deferred — this doc states predictions only. Verdicts are assigned later against the
pinned artifacts.

This is the validation anchor for the T-cell **kinase-activity** surface (motif-MEA on the bulk
phosphoproteome). It is distinct from `tcell_apriori_expectations.md`, which covers composition,
attribution, and Incytr and touched kinases only via a post-hoc re-pin of A1/A2 to the pY track.
That doc's closing note said a blind re-run should commit the corrected "declines-from-d2" direction
*up front* — this doc does exactly that, and extends coverage to the Ser/Thr (`st`) track the prior
anchor never used.

## Discipline

- **Literature-grounded, direction/ordering only.** Each prediction is a sign (up/down vs d2),
  ordering, or presence/absence claim from published T-cell signaling biology. No fabricated
  significance.
- **Pinned.** Every prediction names the artifact it scores against.
- **Blindness — stated honestly.**
  - **ST track: fully blind.** No NES/FDR value on `kinase_timepoint_nes*.csv` (Ser/Thr) has been
    read. This is the substantive new prediction set.
  - **pY track: partially exposed.** The exhaustion anchor already reported pY-track directions
    (proximal TKs negative, JAK/TYK2 positive, ITK/JAK1/TYK2 FDR values). pY predictions below are
    therefore **confirmatory, not blind** — flagged per-row. They are restated here only to commit
    the corrected d2-baseline framing up front.
- **Model-honest.** In-vitro, chronically stimulated T-cell exhaustion time course; donor 1 only for
  kinase activity.

## Surface facts that constrain what is testable (do not re-derive — from `alz/cohorts/tcells/mea.py`)

- **Two tracks.** `py` = tyrosine kinases (substrate = pTyr motifs); `st` = Ser/Thr kinases (IMAC,
  substrate = pSer/pThr motifs). A tyrosine kinase can only score on `py`; a Ser/Thr kinase only on
  `st`. ERK (MAPK1/3) has a dual pT-**E**-p**Y** activation loop and may surface partially on either.
- **Baseline = d2 acute-activation peak.** Contrast is `value(day) − value(d2)`, sign `+ = up at the
  later day`. **An activation-driven kinase is expected to score NEGATIVE later** (it was highest at
  the d2 peak). This is the framing A1 originally got wrong; it is committed up front here.
- **5 contrasts, donor 1 only:** `D1_d13, D1_d15, D1_d17, D1_d19, D1_d20` vs d2, on both tracks, for
  stoich (primary) and raw (sensitivity). **Donor 2 has no kinase surface** (no IMAC → no `st`; pY
  export carries no flanking motif → 0 motifs). Its absence is expected, not a miss.
- **Bulk, whole-population phospho — NOT cycle-excluded.** The "proliferation induced in silico /
  cell cycle excluded" caveat applies to the **scRNA per-cell labeling**, not to this surface. The
  MEA runs on the bulk self-normalized phosphoproteome, so proliferation-kinase activity here is
  **real signal**, not a leak.
- **Motif-family degeneracy is the analog of "detection ≠ activity."** MEA scores substrate-motif
  enrichment. Kinases sharing a motif (basophilic AGC/CAMK; acidic CK2) are **not individually
  resolvable** — those are family-level calls only, not per-kinase.
- **stoich vs raw:** direction should agree across the two kinds; stoich isolates phospho-occupancy
  from protein-abundance drift. A stoich/raw sign flip is itself a finding.
- **Scoreable form:** for a directional call, sign consistent in ≥3/5 contrasts, with FDR < 0.25
  (`MEA_FDR_THRESH`) in ≥1; **strong form** = monotone across all 5. Ordering claims scored on the
  NES matrix directly.

## Committed predictions

### pY track — pinned to `.../mea/kinase_timepoint_nes_pY.csv` + `..._fdr_pY.csv` (+ `_raw_pY` sensitivity)

- **KY1 — Proximal TCR tyrosine kinases decline from d2 (negative NES later).** Src-family **LCK**,
  **FYN**; Syk-family **ZAP70** (and **SYK**); Tec-family **ITK**, **TEC**, **TXK/RLK**; and the
  negative regulator **CSK** (which tracks the proximal module). *Basis:* LCK/FYN phosphorylate
  CD3/ζ ITAMs; ZAP70 docks and fires LAT/SLP76; ITK/TEC drive PLCγ1. Proximal-TCR-signaling
  attenuation is a canonical exhaustion hallmark (Wherry & Kurachi 2015). *Blindness:* confirmatory
  (exposed via A1).
- **KY2 — Cytokine/JAK tyrosine kinases persist or rise vs d2 (positive NES later).** **JAK1**
  (IL-2Rβ / IFN / γc), **JAK3** (γc: IL-2/7/15/21), **JAK2** (IFN-γ, IL-12 partner), **TYK2**
  (type-I IFN, IL-12/23). *Basis:* autocrine IL-2/IFN loops accumulate over the chronic-stim culture,
  so JAK/STAT activity rises relative to the d2 acute peak. *Tension logged:* terminal in-vivo
  exhaustion reduces cytokine responsiveness; in this in-vitro chronic-stim model cytokine signaling
  is expected to persist — directional call is **up**. *Blindness:* confirmatory (exposed via A2).
- **KY-null — non-T-central receptor/other TKs are not predicted.** Any strong enrichment of a
  receptor TK with no T-cell role is off-target motif enrichment, not biology; explicitly out of the
  prediction set.

### ST track — pinned to `.../mea/kinase_timepoint_nes.csv` + `..._fdr.csv` (+ `_raw` sensitivity) — FULLY BLIND

- **KS1 — Anabolic/activation Ser/Thr program declines from d2 (negative NES later).** The strongest
  ST prediction:
  - **PI3K→AKT→mTOR→S6K axis:** **AKT1/2** (PKB), **PDPK1/PDK1**, **MTOR**, **RPS6KB1** (p70S6K),
    **RPS6KA** (RSK). *Basis:* CD28 costimulation drives PI3K-AKT-mTOR; PD-1 engagement suppresses
    PI3K/AKT; exhausted cells are metabolically insufficient with blunted mTOR (Patsoukis 2015;
    Bengsch 2016).
  - **PKCθ (PRKCQ):** immune-synapse kinase, TCR→NF-κB/AP-1; peaks at activation. (Basophilic motif —
    interpret at AGC/PKC-family level.)
  - **MAPK/ERK cascade:** **MAP2K1/2** (MEK), **MAPK1/3** (ERK). Ras→ERK downstream of the TCR.
  - **CaMK:** **CAMK2**, **CAMK4** — Ca²⁺/calmodulin downstream of PLCγ1, tracks proximal signaling.
  - **Canonical NF-κB kinases:** **CHUK/IKKα**, **IKBKB/IKKβ** — TCR/costim-driven canonical NF-κB.
- **KS2 — Stress/energy Ser/Thr kinases rise vs d2 (positive NES later; weaker/uncertain).**
  - **AMPK (PRKAA1/2):** energy-stress sensor; metabolically stressed exhausted cells → up.
  - **Stress MAPKs:** **p38 (MAPK14)**, **JNK (MAPK8/9)** — chronic stress/DNA-damage → possibly up.
  - **GSK3B:** constitutively active, inhibited by AKT-pSer9. *Internal consistency check:* if KS1's
    AKT decline holds, GSK3B is disinhibited → **up**. AKT-down ⇒ GSK3B-up is a coupled pair.
- **KS3 — Proliferation kinases present but NOT exhaustion-informative (no sign predicted).**
  **CDK1/CDK2**, **AURKA/AURKB**, **PLK1**, and CDK substrate motifs. Because this surface is
  whole-population bulk phospho (not cycle-excluded), active proliferation makes these real; scored
  as a context/control, not a discovery. Proliferation rate vs d2 is not a clean exhaustion axis, so
  no direction is committed.
- **KS-caution — degenerate-motif families.** **CK2 (CSNK2)** and other high-constitutive basophilic
  kinases carry heavy substrate loads and promiscuous motifs; any enrichment is treated as
  low-confidence family-level context, not a per-kinase activity call.

### Cross-track / cross-kind

- **KX1 — stoich and raw agree in direction** for every KY/KS call above; a stoich/raw sign flip is
  reported as a mechanism finding (occupancy vs abundance), not averaged away.
- **KX2 — trends are directional across the 5 contrasts**, strongest at the late timepoints
  (d19/d20), consistent with progressive exhaustion; no per-contrast significance is required for a
  directional verdict.

## Pinned artifacts

- pY activity/FDR — `outputs/reports/kinase_attribution_tcells/donor1/mea/kinase_timepoint_nes_pY.csv`,
  `kinase_timepoint_fdr_pY.csv` (+ `_raw_pY`)
- ST activity/FDR — `.../mea/kinase_timepoint_nes.csv`, `kinase_timepoint_fdr.csv` (+ `_raw`)
- Recurrence — `.../mea/recurrence{,_raw}{,_pY}.csv`
- Provenance — `.../mea/mea_manifest.json` (records donor2 skip numerically)

## Scorecard — scored 2026-07-13 against donor1 NES/FDR (stoich primary, raw sensitivity)

**32/38 kinase-level directional calls agree, 5 off, 1 weak.** Rule: predicted sign in ≥3/5
contrasts AND FDR < 0.25 in ≥1 (stoich). ST track was scored blind; pY was confirmatory. No metric
was relaxed — the 5 offs are reported straight.

| Block | Prediction | Verdict | Evidence (NES vs d2, d13→d20) |
|---|---|---|---|
| KY1 | proximal TCR TKs decline | **8/8 agree** | LCK/FYN/SYK/ITK/TEC/TXK/CSK negative 5/5 (ITK −1.5, FDR 0.048; TXK sig 5/5); ZAP70 flips +→− (down d17–d20, 3/5) |
| KY2 | JAK/TYK2 rise | **4/4 agree** | JAK1 +1.68 / TYK2 +1.62 (FDR 0.001), JAK2/JAK3 positive 5/5 |
| KS1 | anabolic Ser/Thr program declines | **15/19** | AKT1/2/3 −2.3…−2.5, PDK1 −1.6, P70S6K −2.0, RSK2 −2.2, all FDR 0.000; PKCθ −1.3, IKKα/β −1.5, CaMK2A/B/D + CaMK4 −1.3…−2.0 all agree; MEK1/2 agree on stoich but raw flips (low conf) |
| KS1 offs | — | **off: MTOR ↑, ERK1/2 ↑; weak: CAMK2G** | MTOR +1.82 (0/5); ERK2 +1.39 (0/5), ERK1 +0.73; CAMK2G mean +0.12 (oscillates) |
| KS2 | stress/energy kinases rise | **5/7** | p38A +0.74, JNK1/2/3 +0.65…+0.72, GSK3B +1.11 (FDR 0.012) agree |
| KS2 offs | — | **off: AMPKA1/2 ↓** | AMPKA1 −0.61, AMPKA2 −0.56 (predicted up) |
| KS3 | proliferation present, no dir | context (as predicted) | CDK1 +2.82 / CDK2 +2.75 (FDR 0.000) — validates real bulk proliferation signal; PLK1 +1.26; AURA/AURB −2.1 |
| KSc | CK2 degenerate-motif | context (as predicted) | CK2A1/A2 oscillate (−1.1 d13 → +1.4 d20); treated as low-confidence |
| KX1 | stoich/raw agree | mostly holds | raw sign agrees except MEK1/2 (flip) — the two noisiest KS1 rows |
| KX2 | trends strongest late | holds | AKT/JAK arms deepen/rise toward d19–d20 |

### The 5 offs, interpreted (not explained away)

- **AMPKA1/2 (KS2) — falsified.** AMPK substrate phosphorylation *declines* from the d2 peak, not
  rises. This was the pre-flagged "weaker/uncertain" bucket; it is simply wrong. AMPK tracks *with*
  the anabolic program here, not reciprocally against it.
- **ERK1/2 (KS1) — off at the kinase, confirmed at the cascade.** The direct upstream **MEK1/2 score
  down (agree)**, so the Ras→MEK read confirms. ERK's *positive* ST-track score is proline-directed
  motif bleed from the dominant CDK1/2 signal (+2.8) — the exact KS-caution / KS3 motif-degeneracy
  pre-registered here. The ERK rows are off; the MAPK cascade is not.
- **MTOR (KS1) — genuine intra-axis split.** AKT/PDK1/S6K/RSK all decline (the PD-1-suppressed arm
  confirms strongly), but mTOR itself scores up on its distinct substrate-motif pool. Reported as a
  real partial miss, not rationalized.

### Confirmed internal-consistency check

**AKT ↓ ⇒ GSK3B ↑** (KS1/KS2 coupled pair): AKT-pSer9 disinhibition of GSK3B. Both scored
independently — AKT1/2/3 down at FDR 0.000, GSK3B up at FDR 0.012. The mechanistic coupling holds.

### Headline

The activation→exhaustion kinase axis is recovered on both tracks: **proximal TCR tyrosine kinases
attenuate** (KY1) while **cytokine JAKs rise** (KY2) on pY; the **CD28/PI3K→AKT→S6K anabolic Ser/Thr
program collapses** (KS1, FDR 0.000) — the strongest blind result — with **stress MAPKs and GSK3B
rising** (KS2). Bulk proliferation (CDK1/2) is real and dominant, which confounds proline-directed ST
scores (ERK) exactly as pre-registered. The clean falsifications are **AMPK** (declines, not rises)
and **mTOR** (splits from its own AKT/S6K arm).
