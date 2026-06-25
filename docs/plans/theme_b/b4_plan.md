# Theme B4 — Kinase → Incytr pathway integration

**TODO:** §B4. **Audit:** `b4_audit.md` (first pass had multiple FALSE claims — corrected by a second command-backed audit 2026-06-25 + direct verification; see "Audit correction" below). **Wave:** 2 (disjoint backend annotation; no Incytr regen, no heavy compute). **Collision class:** backend join + one small payload column per cohort; `song.py` collides with C1/C3 (after C1), `fivexfad.py` is collision-light. **Couples to:** B5 (backbone definition) and B3 (5xFAD `wide_ptm/`).

## Audit correction (read first)
The first audit falsely concluded "Song-only, expression-gate blocked, B3 aspirational." **All three are wrong**, verified directly:
- **5xFAD has kinase MEA** (`kinase_attribution_5xfad/{cortex,hippocampus}_{st,py}_mea_stoichiometry.csv`, carries `Leading substrates`), **pair-mode** (`incytr_pair_mode_5xfad/{cortex,hippocampus}/wide/` **+ `wide_ptm/`**, contrasts `TG_{3,6,9,12}mo_WT`), AND the **expression layer** (`fivexfad_expression_specificity.csv`, `fivexfad_snrna_attribution.csv`). So 5xFAD is the *more* complete target — full expression gate live + PTM pathway set — and the B3→B4 PTM linkage is **real** for 5xFAD.
- **Song** has MEA + `wide/` + kinase cell-type attribution (`kinase_hypothesis_table.csv` `top_celltype_*`) but **no expression-specificity** (confirmed absent; it's the `snrna`/B5 prereq) → Song's expression-gate half is genuinely deferred to B5.
- **T-cell excluded:** its MEA is `mea_timecourse.csv` (no `Leading substrates`) — separate scoping, not B4.

## Decisions (locked, P3 grill 2026-06-25, one question at a time)

**Core = annotation JOIN, no regen.** For each **active kinase** (MEA `Leading substrates`, FDR≤`MEA_FDR_THRESH`=0.25), its leading-edge motifs → `stoichiometry_matrix.csv` → substrate `gene_symbol` (same bridge as D1/C5: `_leading_motifs`/`_substrate_rows`); find pathway rows where that gene sits at `Ligand`/`Receptor`/`EM`/`Target` (Title-case mouse, all four positions). `gene_node_index.json.gz` (exists for Song + both 5xFAD tissues) is the lookup. **No Incytr rerun**, DuckDB-streamed (two 5xFAD parquets >100 MB: cortex `wide/` 167 MB, `wide_ptm/` 194 MB — never whole-file pandas).

**Q1 — Position-aware cell match, annotate-don't-drop.** Pathway cell-ownership: **Ligand is emitted by the *sender*; Receptor/EM/Target live in the *receiver*.** So the kinase's attributed cell-type is checked **position-aware**: gene at **Ligand → match `Sender.group`**; at **Receptor/EM/Target → match `Receiver.group`**. The kinase attribution is a *ranked* list (`top_celltype_{1,2,3}`), so carry a `celltype_match` annotation (matched? at which rank?) — **never drop**. **The stored artifact keeps every hit; the default deliverable/view filters to cell-type matches** — non-matches are retained for traceability only, not the headline. Cluster vocab match verified: the **31 named clusters are identical** across pair-mode `Sender.group`/`Receiver.group` and the attribution/expression tables in both cohorts.

**`channel` field (ps/py).** Each hit carries `channel` = `ps` (ST-track kinase) | `py` (pY-track kinase), so the node's directional evidence reads the matching `{position}_ps_log2FC` / `{position}_py_log2FC`. Attaching phospho activity to a node is only meaningful against the same modality. 5xFAD pY MEA is sparse (cortex 6 / hippo 1 sig kinases) — most hits are `ps`.

**15 `cluster-*` cell types drop by construction.** The expression tables carry 15 unnamed `cluster-*` types with **no** `Sender.group`/`Receiver.group` counterpart → they can never own a node, so the position-aware match never sees them. Reported as excluded (honest), not silently dropped.

**`wide/` vs `wide_ptm/`.** Annotate the pathway set the viewer ships / `gene_node_index` indexes (Song `wide/`; 5xFAD: confirm at build which set the shards derive from). **B4's kinase→node link is phospho-only regardless** — kinases don't drive Ack/KGG, so `wide_ptm`'s extra PTM channels don't change the annotation, only the path universe.

**Q2 — Expression and disease-context are TWO separate annotations, not one gate (5xFAD now; Song after B5).**
- **Expression gate = `fivexfad_expression_specificity.csv`** (per gene×cell_type, **age-pooled**): a **presence/abundance** measure → "is the node's substrate gene expressed in its owning cluster?" Carried **graded** (`fivexfad_fraction_cells_expressing` + `concentration_tier`, not boolean — surface the spectrum). A node whose gene isn't expressed in its owning cluster is **flagged low, not dropped**.
- **Disease context = `fivexfad_snrna_attribution.csv`** (per gene×**age**×cell_type): `fivexfad_lfc`, the substrate gene's own TG-vs-WT change — a **separate, age-matched** annotation, not the gate.
- 5xFAD-only; **Song carries the `celltype_match` flag but no expression/disease-context columns until B5's `snrna` step**, then Song gets the same two layers.

**Q3 — Backbone grain is NOT locked; it is data-determined and deferred to B5.** A "backbone" is a **3-element core** (Receptor–EM spine extended up to Ligand or down to Target), NOT the full 4-tuple: the same `L-R-EM` core can hit many Targets, or the same `R-EM-T` core be reached from many Ligands — counting each full path as a distinct pathway conflates *variations of one core* with *distinct backbones*. Which core is right depends on the data. So B4:
1. **Emits a fan-structure characterization** — per **Receptor–EM** spine: distinct **Ligands** (upstream fan-in), distinct **Targets** (downstream fan-out), distinct sender/receiver pairs. This table is what settles the definition.
2. **Parameterizes the backbone key** — a function over `{Ligand,Receptor,EM,Target}` (candidates `L-R-EM`, `R-EM-T`, `R-EM`, full 4-tuple); `n_backbones` is a rollup against the chosen key, changed by one argument.
3. **Defers the firm grain to the data + B5** (B5 owns "rank by backbone, not the widest enumeration"). B4 ships the characterization + a provisional parameterized count; the definition propagates into B5 (see propagation note).

**Q4 — Backend bridge ships in B4; both viewer stubs + stale preamble deferred until the backbone key is settled.** The two pre-existing orphaned stubs in `01_state.js` are two directions of the same bridge:
- `nBackbones` — **kinase-explorer column** (`#Backbones`): kinase → count of its pathways.
- `drivingKinasesH` (+ `support`/`drivingDirection`/`trend`) — **pathway-detail panel** ("Driving kinases"): pathway → ranked kinases driving it.
Both ride on the backbone key, so neither is wired now. **B4 ships the backend bridge table (the substance, viewer-independent); both stubs are filled together once B5 fixes the key (→ B4.2).** Also: the kinase-tab preamble *already ships* concrete counts ("CAMK2D 15,028 chains") backed by **no populated column** — orphaned numbers to reconcile when `#Backbones` is populated (flag, don't ship more).

## Stages

**Stage 1 — Substrate bridge (per cohort/tissue/track).** Active kinases (FDR≤0.25) → leading-edge motifs → `stoichiometry_matrix.csv` → `gene_symbol`. DuckDB-streamed, pool-filtered. Song: 1 set (9 contrasts). 5xFAD: per (tissue, track) → 4 MEA files; ST/pY tracked via the `channel` field. Reuse the `_leading_motifs`/`_substrate_rows` logic (extract from `concordance_overlap_AD_excl_01_03.py`, ~30 lines).

**Stage 2 — Node join.** For each substrate gene, look up node occupancies via `gene_node_index.json.gz` (gene → role, sender, receiver), then pull the matching rows from `wide/*.parquet` (DuckDB) to attach per-node `{position}_{channel}_log2FC` direction + PDS. Contrast/age alignment: Song contrasts match 1:1; 5xFAD MEA `TG_vs_WT_{N}mo` → age N → pair-mode `TG_{N}mo` (extract age, the join key is age).

**Stage 3 — Annotations.** Per hit attach: `celltype_match` (position-aware vs kinase `top_celltype_{1,2,3}` rank; **5xFAD kinase cell-type attribution source = build-time resolution**, see below); 5xFAD `expression_fraction`+`concentration_tier` (specificity, age-pooled) and `disease_lfc` (attribution, age-matched); `channel`; kinase `NES` (signed) + `FDR`.

**Stage 4 — Fan characterization + parameterized backbone count.** Emit the Receptor–EM fan-structure table (Q3). Compute `n_backbones` per kinase against the parameterized key (provisional default pending B5 — pick the data-informed key after inspecting the characterization). Keep every count cell-type-matched + within the gated `wide/` set.

**Stage 5 — Outputs (backend; viewer-independent).** `outputs/reports/kinase_incytr_bridge/<cohort>/`:
- `kinase_node_hits.parquet` (flat, viewer-consumable): one row per (kinase, contrast, substrate_gene, node_role, Sender.group, Receiver.group, channel, node_log2FC, PDS, celltype_match[, expression_fraction, concentration_tier, disease_lfc]). The artifact keeps non-matches; a `celltype_match` boolean drives the default filter.
- `recep_em_fan.csv` (the Q3 characterization).
- `kinase_backbone_counts.csv` (per kinase × backbone-key → `n_backbones`, provisional).
- `MANIFEST.md`: cohorts, contrasts, FDR cutoff, backbone key used (+ "provisional, pending B5"), the cluster-* exclusion, the Song expression-deferred note.

**Stage 6 — Runner + task.** `alz/cross_reference/kinase_incytr_bridge.py` (sibling of D1's module) or `alz/integration/` (uses `config_integration.load_cluster_spine`/paths). Pixi task `kinase-incytr-bridge`. No viewer edit in B4 (the `#Backbones` payload column + both stubs land in B4.2 with the settled key).

## Build-time resolutions (verify, don't re-grill)
- **5xFAD kinase cell-type attribution source** for the position-aware match: Song uses `kinase_hypothesis_table.top_celltype_{1,2,3}`; 5xFAD has no obvious kinase-level top_celltype table — resolve whether to derive the kinase's cell type(s) from `fivexfad_snrna_attribution` (cell types where its substrates are expressed/significant) or a 5xFAD attribution artifact. Confirm before Stage 3.
- **Which 5xFAD pathway set the shards/index derive from** (`wide/` vs `wide_ptm/`) — annotate that one.

## Verification
- Active-kinase count at FDR≤0.25 matches the MEA tables (Song 1,196 (kinase,contrast) pairs; 5xFAD cortex_st 591 etc.). A known kinase's leading-edge substrate appears at the expected node position in `kinase_node_hits`.
- `celltype_match` honored: a hit where the kinase's `top_celltype` ≠ the node-owner cluster is PRESENT in the artifact (flagged false) but absent from the default filtered view.
- 5xFAD: a node-gene not expressed in its owning cluster shows low `expression_fraction`, not a dropped row; `disease_lfc` is age-matched to the contrast. Song: expression columns absent (documented), `celltype_match` present.
- Fan characterization: a Receptor–EM spine with many Targets (or many Ligands) is visible in `recep_em_fan.csv` — i.e. the data actually shows the conflation Q3 anticipated. `n_backbones` recomputes under a different `--backbone-key` without code change.
- **Memory:** runs under cap; peak RSS reported; the two >100 MB 5xFAD parquets are DuckDB-streamed, never whole-file pandas.

## Out of scope
The viewer stubs (`#Backbones` column, `Driving kinases` panel) + the stale preamble reconciliation (→ B4.2, gated on B5's backbone key + browser verification), T-cell cohort (different MEA format), any Incytr regen, the firm backbone definition (B5), Song's expression layer (B5/`snrna` prereq), Ack/KGG-driven node annotation (kinases don't drive those PTMs).
