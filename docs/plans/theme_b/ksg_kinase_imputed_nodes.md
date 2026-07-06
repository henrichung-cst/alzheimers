# KsG — kinase-substrate gene admission (a third candidate-gene layer)

**Orchestration:** [`incytr_rerun_ksg_ptm_backbone_2026-06-29.md`](../incytr_rerun_ksg_ptm_backbone_2026-06-29.md).

**Status:** plan / pre-implementation. **Theme:** B (Incytr ↔ kinase). **Companion:**
[`backbone_incytr_track.md`](backbone_incytr_track.md) (the live Incytr-viewer design),
[`b4_plan.md`](b4_plan.md) (the kinase→pathway *annotation* bridge this inverts).

## Goal

Add a **third candidate-gene layer** to Incytr pair-mode, alongside DEG (transcript-DE) and prG
(protein-DE). The new layer, **KsG** (kinase-substrate gene), admits a gene to a cluster's `gene.use`
when an enriched kinase attributed to that cluster predicts the gene as a leading-edge substrate —
*even if the gene is not differentially abundant by transcript or protein*. A kinase-supported gene
whose phospho readout is missing **stays missing** — the engine scores whatever phospho signal it
finds, and the `|PDS|≥0.2` floor self-limits over-emission. KsG widens `gene.use`; it does not
impute.

This is the inverse of B4. B4 reads finished pathways and annotates which kinases act on each node;
KsG pushes the *same* substrate bridge upstream into `gene.use`, so a kinase-predicted substrate can
*create* a node. The two compose: a KsG badge's "why" is exactly B4's driving-kinase annotation.

## Locked decisions

1. **Any node.** KsG genes union into both `gene.use_Sender` and `gene.use_Receiver` (no position
   restriction). The engine gates all receiver positions by one flat `gene.use_Receiver`, so
   per-position admission is not expressible upstream anyway; position is a *labeling* concern, not a
   gene.use one.
2. **All three cohorts:** 5xFAD (pilot), Song, t-cell.
3. **Admission-only.** KsG only widens `gene.use`; a kinase-supported gene whose phospho is missing
   stays missing. The `|PDS|≥0.2` floor self-limits over-emission. We never seed or impute the
   phospho substrate.
4. **Activity-only — never manufacture presence.** KsG touches only `gene.use`. We never seed the
   expression or protein substrate.
5. **Song drops sce4 parity-preservation.** KsG injects directly into Song's production build (no
   shadow un-augmented build). `verify-incytr-sce4`'s symmetric-diff exemption widens transgene →
   **transgene ∪ KsG**. Song keeps frozen sce4 per-pair `gene.use` as its *base* (parity constant
   #1b) — KsG augments it; we drop the byte-identity *gate*, not the frozen *input*.

## Mechanism — two changes, all upstream of an unmodified scorer

The scorer is never touched. Everything is input mutation + label provenance, mirroring the existing
`floor_pr` / min-10000 imputation precedents (`incytr_commandline.R:236-242`).

### Toggle = kinase-data presence (not a mode flag)

KsG is gated on whether kinase inputs are supplied — `KSG_MEA_FILE` set ⇒ active; unset ⇒ the driver
is **byte-identical** to the prior no-kinase run. This mirrors `SCE4_GENEUSE_DIR` / `ACK_FILE` (data
presence selects behaviour, anti-shim-clean) and **supersedes decision 5's "widen the parity
exemption"**: `verify-incytr-sce4` simply runs with no kinase inputs and stays byte-identical, while
the production Song build supplies them and deviates. App env vars (set only by KsG-enabled runners):
`KSG_MEA_FILE`(+`_PY_FILE`), `KSG_MOTIF_FILE`, `KSG_ATTRIBUTION_FILE` (long-form kinase,cell_type),
`KSG_CONTRAST` (MEA contrast label — the runner maps `ma_2mo_AppP`→`App_2mo`).

### Layer 1 — admit KsG genes to `gene.use`

`incytr_commandline.R` builds `dg_by_cluster[[cl]] = union(deg_by_cluster[[cl]], prg_by_cluster[[cl]])`
(lines 428-432), assigned to both `gus_by_cluster` / `gur_by_cluster`. KsG adds a third set:

```r
ksg_by_cluster[[cl]] = <KsG genes for cluster cl, this contrast>   # from Incytr::kinase_substrate_gene
dg_by_cluster[[cl]]  = Reduce(union, list(deg, prg, ksg))
```

Injected in `process_pair()` **after** `gene_use_S/R` resolve (lines 604-609), so it applies
uniformly to the frozen (Song) and derived (5xFAD/t-cell) bases.

### Layer 3 — provenance: KsG label

- **Label** ∈ {DEG, prG, **KsG**} — *why the node is a candidate*. `label_node()`
  (`incytr_commandline.R:546-553`) gains a third tier with precedence **DEG > prG > KsG**, so a KsG
  badge marks a node admitted **only** by kinase evidence (the pure value-add). KsG = label code 3,
  the last free slot in the 2-bit packed field (`incytr_index.py:104-116`).

## KsG-set computation — a package method (`Incytr::kinase_substrate_gene`)

"Into the official engine" means the one genuinely new *method* lives in the `Incytr` package,
paralleling `proteomics_gene` — respecting the repo's method/application separation. The app reads
its MEA / motif-map / attribution CSVs and *calls* the method; no Python sidecar, no
`KINASE_GENEUSE_DIR` CSV intermediary.

**`Incytr::kinase_substrate_gene(mea, motif_map, attribution, cell_group, fdr_cutoff = 0.25)`**
→ a data.table of `(gene_symbol, cluster, kinase, NES, channel)`, one row per admitted (gene,
cluster). It parses the `;`-delimited leading-substrate motifs → substrate genes, keeps FDR-passing
kinases, unions the substrate genes of the kinases attributed to each cluster, and picks the driving
kinase per gene by max |NES|.

**Status: IMPLEMENTED + verified.** `~/Projects/work/incytr/R/utils.R` (exported via roxygen),
`tests/testthat/test-kinase_substrate_gene.R` (13 assertions pass). Additive-only — the scoring
path never calls it, so **`test-golden_output.R` (13/13) and `test-sce4_defaults.R` (10/10) still
pass byte-identically**. The scorer needs no change: admission is a larger `gene.use` vector
(`%in%` membership in `enumerate_paths`), and the phospho seam (`compute_omics_cluster_fc`) scores
whatever raw condition columns it is handed.

### KsG-set logic, in order — each filter is an over-emission governor

1. **Active kinase** — MEA `FDR ≤ 0.25`, leading-edge substrates only.
2. **Cell-type match** — kinase attributed to `cluster` (B4's position-aware attribution: Song
   `kinase_hypothesis_table.top_celltype_{1,2,3}`; 5xFAD `celltype_mea` ranked; t-cell — confirm a
   kinase→state/type attribution source at build).

## Pilot findings (Song, 2026-06-29) — what reshaped the design

Measured on `Astrocytes → Microglia` (Microglia as receiver, App_2mo) and the canonical
`Microglia → Cholinergic-Neurons`:

- **Admission is the feature.** Of 500 kinase-supported receiver genes, **499 already have a
  measured phospho value** in Microglia (real values, ps median ≈ 82, not residuals) — they score
  on real data. Only **1** is genuinely unmeasured and contributes nothing to `PhPDS`. KsG is
  overwhelmingly kinase-guided *admission* of measured-but-not-DE genes; the `|PDS|≥0.2` floor
  governs over-emission.
- **Position depends on the pair's geometry, not a rule.** A sender contributes only the Ligand
  node; a receiver contributes Receptor/EM/Target. KsG fires where the *kinase-rich* cell sits:
  Microglia-as-sender ⇒ only Ligands (and the admitted "ligands" — Gad1/Shank1/Rims2 — are
  biologically weak, since a kinase substrate need not be secreted); Microglia-as-receiver ⇒
  EM/Target (496/500 as EM, 500 as Target — the coherent case). "Any node" (decision 1) is kept,
  but the Ligand case is the noisy one.
- **KsG can mint pairs sce4 never had.** The frozen Song base omits `Astrocytes → Microglia`
  entirely (0/0), so against it KsG creates the pair wholesale — a legitimate consequence of
  dropping sce4-preservation (decision 5).

## Cohort handling

| Cohort | gene.use base | Kinase / expression data | Notes |
|---|---|---|---|
| **5xFAD** | derived (`SCE4_GENEUSE_DIR` unset) | ST+pY MEA, per-celltype attribution | **Pilot.** Richest data; no parity lock. |
| **Song** | frozen sce4 per-pair (kept) | full MEA, `kinase_hypothesis_table` | KsG inline; **no shadow build**; parity gate widened (decision 5). |
| **t-cell** | derived | MEA `mea_timecourse.csv` **carries `Leading substrates`** (corrects B4's exclusion) | Confirm kinase→state/type attribution source before Stage 3. |

Runners (`run_pair_mode_5xfad.sh`, `run_pair_mode.sh`, `run_pair_mode_tcells.sh`) each export
`KSG_MEA_FILE` (and companion env vars) when KsG is enabled.

## Over-emission control + mandatory pilot

Under admission-only, KsG genes with a missing phospho contribute nothing to `PhPDS` — the
`|PDS|≥0.2` floor governs. Genes admitted by KsG that already carry measured phospho score their
real `PhPDS`; the floor remains the over-emission control. This is an admission-breadth question
(FDR, leading-edge, cell-type-match tightness); treat it like the sender-breadth issue that
dominated sce4 reproduction.

**Mandatory measured pilot on one 5xFAD contrast before any viewer wiring**, reporting:
- **Admission delta** — KsG genes added per cluster beyond DEG∪prG.
- **Floor-survivor delta** — new paths enumerated vs. surviving `|PDS|≥0.2` (the real deliverable
  delta, not raw enumeration).
- **Position distribution** — where KsG nodes land (L/R/EM/T).
- **Cap eviction** — do KsG paths displace DEG/prG paths in any top-N view.

Ship only if the floor-survivor delta is bounded and defensible; if it floods, tighten the gate
(higher kinase FDR) — never relax the `|PDS|` floor to manage the number.

## Viewer

- **KsG badge** — `_INCYTR_LABEL_VOCAB = ("DEG","prG","KsG")` (`incytr_index.py:10`) cascades to the
  bit-encoding, `label_states`, and shard Categorical automatically; hand-edits: R `label_node`
  (third tier) + JS `_IP_LABEL_COLORS` (`incytr_pathways.js:101-104`, a third color).
- **"Why" tooltip** — the KsG badge names the driving kinase (from `ksg/<contrast>.csv`), closing the
  B4 loop on one surface.

## Verification

1. **Pilot first** (above) — gated human review before viewer work.
2. `verify-incytr-sce4` (Song): gated path-set symmetric-diff vs sce4 Allpathway is **transgene ∪
   KsG** only; gate updated, stays green on that definition.
3. Both viewers build; KsG badge renders; precedence DEG>prG>KsG holds.
4. Memory: the bridge + participation stay DuckDB-streamed.

## Out of scope

- Manufacturing expression or protein **abundance** — explicitly refused (decision 4). KsG touches
  only `gene.use`.
- Deriving the PDS contribution from kinase NES *directly* — the unmodified scorer computes PDS from
  whatever phospho signal exists; we never inject a score or pseudo-FC into the scorer.
- Reverting Song to derived `DEG∪prG` — a separate change; Song keeps the frozen base (decision 5).
- Ack/KGG/Rme1 imputation — kinases don't drive those PTMs (same boundary as B4).

## Open items to confirm at build

- **t-cell kinase→cell-type/state attribution source** for the cell-type-match filter (MEA is
  per-donor timecourse; B4 excluded t-cell and never resolved this).
