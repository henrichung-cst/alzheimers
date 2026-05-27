# Human CTRL outlier audit — FINDINGS

Date: 2026-05-25
Plan: [`human_ctrl_outlier_audit_2026-05-25.md`](human_ctrl_outlier_audit_2026-05-25.md)
Scripts: `alz/cross_reference/ctrl_outlier_audit.py` (A–C), `ctrl_outlier_audit_kinases.py` (D)
Figures/tables: `outputs/reports/kinase_attribution_human/ctrl_audit/`

## Verdict

**The AD-like signal in the last 3 sequential controls is genuine, not a technical artifact.**

| Sample | Verdict | Strength |
|---|---|---|
| **CTRL-08** | Genuine AD-like | Strong — embeds inside the AD cluster on every measure |
| **CTRL-10** | Genuine AD-like | Strong — embeds inside the AD cluster on every measure |
| **CTRL-07** | AD-leaning, genuine but weaker | Moderate — clearly AD-ward (3× any clean control, survives coverage control) but the least AD-like of the three, and carries an independent low-coverage caveat |

The phospho-omic evidence that motivates the kinase MEA is genuinely shared between these
controls and the AD donors: the same phosphosites move the same direction, so the MEA NES is
similar because the underlying data is similar — not because of a method/coverage/normalization
artifact. This supports the statement: *"I know these last 3 controls look unusual, but after
triple-checking the underlying data and method, it is genuine."*

Groups used: AD = 10 donors; **clean controls** = CTRL-01/02/03/04; **suspicious controls** =
CTRL-07/08/10.

## Evidence

### Phase A — structure on the underlying stoichiometry data (`phaseA_pca_st.png`, `phaseA_corr_heatmap_st.png`)
PCA on complete-case IMAC sites: **CTRL-08 and CTRL-10 fall inside the AD cluster**; clean
controls separate along PC2. CTRL-07 sits apart (it is also a general outlier, see below).
Global Pearson is uninformative (all samples ~0.9, shared structure) — the discrimination axis
below is the right lens.

### Phase C — alignment with the AD-vs-clean axis (`phaseC_alignment_st.png`)
Correlation of each sample's deviation-from-clean-controls with the AD-vs-clean discrimination
vector, over the top 500 discriminating IMAC sites (+1 = behaves like AD, 0 = like clean controls):

| sample | alignment (IMAC) | alignment (pY) |
|---|---|---|
| CTRL-01..04 (clean) | −0.085 … +0.157 | −0.125 … +0.137 |
| **CTRL-07** | **+0.466** | +0.156 |
| **CTRL-08** | **+0.864** | +0.344 |
| **CTRL-10** | **+0.782** | +0.298 |
| AD donors (range) | +0.543 … +0.948 | +0.374 … +0.737 |

CTRL-08 and CTRL-10 land **inside the AD range**; CTRL-07 is intermediate but ~3× any clean
control. pY is directionally identical, weaker (only 1,133 sites — underpowered).

### Phase B — artifact controls (all negative)
- **Coverage:** AD-correlation is unchanged restricting to the 9,237/9,554 sites quantified in
  all 17 samples (CTRL-07 0.827→0.826, CTRL-08 0.959→0.959, CTRL-10 0.939→0.940). All three
  suspicious controls have ~99% finite sites — site count is not the issue. CTRL-07's flagged
  "low coverage" is in the *total-proteome denominator* (`median_log2_protein` 8.534), not the
  phosphosite count.
- **Normalization:** on raw phospho with **no protein denominator**, CTRL-08 (+0.923) and
  CTRL-10 (+0.892) are still the **most AD-correlated of all 7 controls** (clean controls
  0.64–0.83). Not a stoichiometry-denominator artifact.
- **Run order:** acquisition columns are ID-sorted (all AD, then all CTRL; shared `053124`
  date) so injection order is not separable from sample ID in metadata. But the signal is
  *site-specific* — concentrated at AD-discriminating substrate motifs (Phase D), not a global
  drift — which a run-order/batch artifact would not produce.

### Phase D — per-kinase leading-edge proof (`phaseD_leading_edge_proof.png`, `phaseD_kinase_table.csv`)
The granular proof, kinase by kinase, that the **motif signal MEA consumes is the same** in the
suspicious controls and AD. 8 kinases significant + concordant in both AD and the suspicious
controls (5 up, 3 down; families CK2/GRK/PLK/CAMK2/ACVR + HIPK/MOK/BUB1):

| metric (IMAC) | up-kinases | down-kinases | reading |
|---|---|---|---|
| median NES — AD | +2.5 … +2.8 | −1.6 … −1.8 | active in AD |
| median NES — **suspicious** | **+2.6 … +3.1** | **−1.7 … −2.0** | **same as AD** |
| median NES — clean | −1.8 … −2.0 | +0.3 … +1.1 | opposite (mirror) |
| substrate-site LFC corr, AD vs suspicious | +0.62 … +0.71 | +0.44 | sites move together |
| **clean-baseline deviation corr, AD vs suspicious** | **+0.86 … +0.90** | **+0.78** | **same deviation from real controls** |
| leading-edge Jaccard, AD ∩ suspicious | 0.34 … 0.44 | 0.17 | share driver sites |
| leading-edge Jaccard, AD ∩ clean | 0.01 … 0.03 | 0.02 … 0.03 | clean shares almost none |

The clean-baseline deviation correlation (0.78–0.90) is the decisive number: at each kinase's
substrate sites, *how AD departs from legitimate controls* and *how the suspicious controls
depart from legitimate controls* are nearly the same vector. The running-enrichment curves for
AD and the suspicious controls overlay; the clean curve is the mirror image.

## Consequence for the analysis (gates Concern 1)

The per-donor human contrast (`alz/ingest/mukesh_perdonor.py:_build_donor_deltas`) builds every
donor's LFC against `nanmean(all 7 CTRL)`. With **3 of 7 controls genuinely AD-like, the control
reference mean is pulled toward the AD phenotype.** The signature of this contamination is visible
above: the clean controls show **significant NES in the *anti-AD* direction** (−2.0 for up-kinases,
+1.0 for down-kinases) purely because they sit below a contaminated mean. This attenuates every
AD-vs-CTRL effect and mislabels the legitimate controls as the outliers.

→ Concern 1's single AD-vs-CTRL kinase metric must be computed against the **clean control set
(CTRL-01/02/03/04)**, not all 7. Whether to also exclude CTRL-07/08/10 from the cohort or treat
them as a distinct "AD-like control" group is a study-design decision for the next step.
