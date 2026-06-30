# Theme B4 — Kinase → Incytr pathway integration

**TODO:** §B4. **Audit:** `b4_audit.md` (first pass had multiple FALSE claims — corrected by a second command-backed audit 2026-06-25 + direct verification; see "Audit correction" below). **Wave:** 2 (disjoint backend annotation; no Incytr regen, no heavy compute). **Collision class:** backend join + one small payload column per cohort; `song.py` collides with C1/C3 (after C1), `fivexfad.py` is collision-light. **Couples to:** B3 (5xFAD `wide_ptm/`). (B4 owns only the per-kinase `#Backbones`/`#Paths` counts in `kinase_participation.csv`; the standalone R-EM-T reduction this build once folded in was removed by the backbone grain pivot — see `backbone_incytr_track.md`.)

## Audit correction (read first)
The first audit falsely concluded "Song-only, expression-gate blocked, B3 aspirational." **All three are wrong**, verified directly:
- **5xFAD has kinase MEA** (`kinase_attribution_5xfad/{cortex,hippocampus}_{st,py}_mea_stoichiometry.csv`, carries `Leading substrates`), **pair-mode** (`incytr_pair_mode_5xfad/{cortex,hippocampus}/wide/` **+ `wide_ptm/`**, contrasts `TG_{3,6,9,12}mo_WT`), AND the **expression layer** (`fivexfad_expression_specificity.csv`, `fivexfad_snrna_attribution.csv`). So 5xFAD is the *more* complete target — full expression gate live + PTM pathway set — and the B3→B4 PTM linkage is **real** for 5xFAD.
- **Song** has MEA + `wide/` + kinase cell-type attribution (`kinase_hypothesis_table.csv` `top_celltype_*`) but **no expression-specificity** (confirmed absent; it's the `snrna`/B5 prereq) → Song's expression-gate half is genuinely deferred to B5.
- **T-cell excluded:** its MEA is `mea_timecourse.csv` (no `Leading substrates`) — separate scoping, not B4.

## Decisions (locked, P3 grill 2026-06-25, one question at a time)

**Core = annotation JOIN, no regen.** For each **active kinase** (MEA `Leading substrates`, FDR≤`MEA_FDR_THRESH`=0.25), its leading-edge motifs → `stoichiometry_matrix.csv` → substrate `gene_symbol` (same bridge as D1/C5: `_leading_motifs`/`_substrate_rows`); find pathway rows where that gene sits at `Ligand`/`Receptor`/`EM`/`Target` (Title-case mouse, all four positions). `gene_node_index.json.gz` (exists for Song + both 5xFAD tissues) is the lookup. **No Incytr rerun**, DuckDB-streamed (two 5xFAD parquets >100 MB: cortex `wide/` 167 MB, `wide_ptm/` 194 MB — never whole-file pandas).

**Q1 — Position-aware cell match, annotate-don't-drop.** Pathway cell-ownership: **Ligand is emitted by the *sender*; Receptor/EM/Target live in the *receiver*.** So the kinase's attributed cell-type is checked **position-aware**: gene at **Ligand → match `Sender.group`**; at **Receptor/EM/Target → match `Receiver.group`**. The kinase attribution is a *ranked* list (`top_celltype_{1,2,3}`), so carry a `celltype_match` annotation (matched? at which rank?) — **never drop**. **The stored artifact keeps every hit; the default deliverable/view filters to cell-type matches** — non-matches are retained for traceability only, not the headline. Cluster vocab match verified: the **31 named clusters are identical** across pair-mode `Sender.group`/`Receiver.group` and the attribution/expression tables in both cohorts.

**`channel` field (st/py).** Each hit carries `channel` = `st` (ST-track kinase) | `py` (pY-track kinase), tracking which MEA track surfaced the kinase. 5xFAD pY MEA is sparse (cortex 6 / hippo 1 sig kinases) — most hits are `st`.

**15 `cluster-*` cell types drop by construction.** The expression tables carry 15 unnamed `cluster-*` types with **no** `Sender.group`/`Receiver.group` counterpart → they can never own a node, so the position-aware match never sees them. Reported as excluded (honest), not silently dropped.

**`wide/` vs `wide_ptm/`.** Annotate the pathway set the viewer ships / `gene_node_index` indexes (Song `wide/`; 5xFAD: confirm at build which set the shards derive from). **B4's kinase→node link is phospho-only regardless** — kinases don't drive Ack/KGG, so `wide_ptm`'s extra PTM channels don't change the annotation, only the path universe.

**Q2 — Expression and disease-context are TWO separate annotations, not one gate (5xFAD now; Song after B5).**
- **Expression gate = `fivexfad_expression_specificity.csv`** (per gene×cell_type, **age-pooled**): a **presence/abundance** measure → "is the node's substrate gene expressed in its owning cluster?" Carried **graded** (`fivexfad_fraction_cells_expressing` + `concentration_tier`, not boolean — surface the spectrum). A node whose gene isn't expressed in its owning cluster is **flagged low, not dropped**.
- **Disease context = `fivexfad_snrna_attribution.csv`** (per gene×**age**×cell_type): `fivexfad_lfc`, the substrate gene's own TG-vs-WT change — a **separate, age-matched** annotation, not the gate.
- 5xFAD-only; **Song carries the `celltype_match` flag but no expression/disease-context columns until B5's `snrna` step**, then Song gets the same two layers.

**Q3 — Backbone grain (SETTLED 2026-06-28: two exact participation counts + an R-EM-T reduction).** The fan characterization confirmed the conflation it anticipated — across Song's 769 Receptor-EM spines, Target fans ~547× — so no single grain answers everything. B4 ships **two exact** per-kinase counts (`compute_participation_counts`, DuckDB, not estimated), both any-node (L/R/EM/T) participation over the gated `wide/` paths:
1. **`recep_em_fan.csv`** — per Receptor-EM spine: distinct Ligands (fan-in), Targets (fan-out), sender/receiver pairs. The structural characterization.
2. **`n_backbones`** — distinct (Sender, Receiver, Receptor, EM) **spines** the kinase acts on (the breadth number the kinase tab shows; Target fan-out collapsed).
3. **`n_paths`** — distinct full (Sender, Receiver, Ligand, Receptor, EM, Target) **pathways** the kinase sits along (total routes; ~53× larger).

B4 produces only these per-kinase counts (`kinase_participation.csv`). Backbone grains are emitted separately by the scoring engine (`Cal_pairwise_grid`), not by this build — see `backbone_incytr_track.md`.

**Q4 — Backend bridge ships in B4; viewer wiring is Phase 2 of the fold consolidation.** Two pre-existing orphaned stubs in `01_state.js`:
- `nBackbones` — **kinase-explorer column** (`#Backbones`): now backed by `n_backbones` (+ a `#Paths` column for `n_paths`).
- `drivingKinasesH` (+ `support`/`drivingDirection`/`trend`) — **pathway-detail panel** ("Driving kinases"): pathway → ranked kinases driving it.
The kinase-tab intro *already ships* concrete counts ("CAMK2D 15,028 chains") backed by **no populated column** — these are the `n_backbones` spine count (CAMK2D 14,968); reconcile the intro literal when wiring. B4 ships the backend (`kinase_participation.csv`); the viewer wiring landed in B4.2.

## Stages

**Stage 1 — Substrate bridge (per cohort/tissue/track).** Active kinases (FDR≤0.25) → leading-edge motifs → `stoichiometry_matrix.csv` → `gene_symbol`. DuckDB-streamed, pool-filtered. Song: 1 set (9 contrasts). 5xFAD: per (tissue, track) → 4 MEA files; ST/pY tracked via the `channel` field. Reuse the `_leading_motifs`/`_substrate_rows` logic (extract from `concordance_overlap_AD_excl_01_03.py`, ~30 lines).

**Stage 2 — Node join.** For each substrate gene, look up node occupancies via `gene_node_index.json.gz` (gene → role, sender, receiver). The join produces one hit row per (kinase, contrast, gene, role, sender, receiver). No per-node fold-change is attached: the wide parquets carry `{position}_sclog2FC`/`_pr_`/`_ps_`/`_py_log2FC` (never an `_st_`/`_py_log2FC` keyed by the MEA `channel` field), and no consumer reads a node-level fc/PDS off this artifact — the directional substrate is the path-level `sclog2FC`/PDS already in `wide/`, which B2 and the pathway tab read directly.

**Stage 3 — Annotations.** Per hit attach: `celltype_match` (position-aware vs kinase `top_celltype_{1,2,3}` rank, vectorized join — the hit table is ~9.6M rows so no per-row Python loop; **5xFAD kinase cell-type attribution source = build-time resolution**, see below); 5xFAD `expression_fraction`+`concentration_tier` (specificity, age-pooled) and `disease_lfc` (attribution, age-matched); `channel`; kinase `NES` (signed) + `FDR`.

**Stage 4 — Fan characterization + exact participation counts.** Emit the Receptor–EM fan-structure table (`build_recep_em_fan`). Compute, per kinase, the two exact participation counts `n_backbones` + `n_paths` (`compute_participation_counts`, DuckDB unpivot → 4-key equi-join `(sender, receiver, role, gene)` over the gated `wide/` shards). Both counts are over ALL hits (matched or not) — the count is a substrate-phosphorylator over-representation measure, not cell-type-gated.

**Stage 5 — Outputs (backend; viewer-independent).** `outputs/reports/kinase_incytr_bridge/<cohort>/`:
- `kinase_node_hits.parquet` (flat): one row per (kinase, contrast, substrate_gene, role, Sender.group, Receiver.group, channel, celltype_match, celltype_match_rank[, expression_fraction, concentration_tier, disease_lfc]). The artifact keeps non-matches; a `celltype_match` boolean drives the default filter. No node-level fc/PDS columns (no consumer; the wide path-level `sclog2FC`/PDS is the directional substrate).
- `recep_em_fan.csv` (the fan characterization).
- `kinase_participation.csv` (per kinase → `n_backbones`, `n_paths`).
- `MANIFEST.md`: cohorts, contrasts, FDR cutoff, the canonical floor, the cluster-* exclusion, the Song expression-deferred note.

**Stage 6 — Runner + task.** `alz/cross_reference/kinase_incytr_bridge.py` (run via `pixi run kinase-incytr-bridge` → `python -m alz.cross_reference.kinase_incytr_bridge`). Emits `kinase_participation.csv` per cohort. No viewer edit in B4 (the `#Backbones`/`#Paths` columns + both stubs landed in B4.2).

## Build-time resolutions (verify, don't re-grill)
- **5xFAD kinase cell-type attribution source** for the position-aware match: Song uses `kinase_hypothesis_table.top_celltype_{1,2,3}`; 5xFAD has no obvious kinase-level top_celltype table — resolve whether to derive the kinase's cell type(s) from `fivexfad_snrna_attribution` (cell types where its substrates are expressed/significant) or a 5xFAD attribution artifact. Confirm before Stage 3.
- **Which 5xFAD pathway set the shards/index derive from** (`wide/` vs `wide_ptm/`) — annotate that one.

## Verification
- Active-kinase count at FDR≤0.25 matches the MEA tables (Song 1,196 (kinase,contrast) pairs; 5xFAD cortex_st 591 etc.). A known kinase's leading-edge substrate appears at the expected node position in `kinase_node_hits`.
- `celltype_match` honored: a hit where the kinase's `top_celltype` ≠ the node-owner cluster is PRESENT in the artifact (flagged false) but absent from the default filtered view.
- 5xFAD: a node-gene not expressed in its owning cluster shows low `expression_fraction`, not a dropped row; `disease_lfc` is age-matched to the contrast. Song: expression columns absent (documented), `celltype_match` present.
- Fan characterization: a Receptor–EM spine with many Targets (or many Ligands) is visible in `recep_em_fan.csv` — the data shows the conflation Q3 anticipated. `n_backbones` reproduces the shipped kinase-tab intro literals (CAMK2D 14,968 ≈ the "15,028 chains" number, ranking preserved); `n_paths` is the full-route count alongside it.
- **Memory:** runs under cap; the wide parquets are DuckDB-streamed (participation count + the folded R-EM-T reduction), never whole-file pandas.

## Out of scope
The viewer wiring (`#Backbones`/`#Paths` columns, `Driving kinases` panel, intro-literal reconciliation — landed in B4.2, gated on browser verification), T-cell cohort (different MEA format), any Incytr regen, Song's expression layer (`snrna` prereq), Ack/KGG-driven node annotation (kinases don't drive those PTMs).
