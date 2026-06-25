# Cross-Theme Contracts (P2)

Shared naming/schema/convention decisions that **every theme plan must cite**. These exist so independently-running parallel agents produce consistent — not contradictory — output. Resolved in the P2 contract grill.

Status: C2 ✅ · C1 ✅ · B3 ✅ (already built — convention only) · B5 ✅ · F1 ✅ · F2 ✅ — **P2 closed**

---

## C2 — Cohort naming convention ✅

**Decision.** Standard convention `MouseC{N}` / `HumanC{N}`, with an underscore split-suffix where a cohort subdivides. Surnames (`Song`, `Mukesh`) and model names (`5xFAD`) are removed from the **display surface**.

| Internal key (UNCHANGED) | Display name | Split suffix |
|---|---|---|
| `song` | **MouseC1** | `MouseC1_App`, `MouseC1_Tau`, `MouseC1_ApTt` (genotypes, per C1) |
| `fivexfad` | **MouseC2** | `MouseC2_Cortex`, `MouseC2_Hippo` (ages stay as contrast selectors, not name suffixes) |
| `mukesh` | **HumanC1** | — |

Index assigned by primacy: C1 = primary IMAC + snRNA cohort, C2 = supporting transgenic model.

**Out of scope.**
- **T-cell cohort** (`tcells`/donor1/donor2) — entirely separate viewer, no surname to fix. Not covered by C2.
- **Reference atlases** (SEA-AD, WMB, NSCLC) — published atlas names; keep as-is (renaming erases external recognizability for collaborators).

**Rename depth — VIEW/LABEL LAYER ONLY.** Rename only what the user reads or exports:
- viewer tab labels, panel headings, axis titles, legend entries
- export filenames (`mousec1_app__kinase_nes.csv`)
- export CSV **column headers** (`song_lfc` → `MouseC1_lfc` at the export boundary)

**Keep internal identifiers unchanged** — `cohort_id="song"`, payload JSON keys (`song_lfc`, `song_specificity`), JS function/DOM names (`_songSpecificityRank`, `ke-filter-song`), module dirs (`alz/cohorts/song/`, `alz/viewer/cohorts/song.py`), and on-disk data paths (`data/datasets/song/`).

**Rationale.** An internal key the user never sees is an identifier, not a dual representation — it is *not* the data-says-X / UI-says-Y trap Theme A forbade (that trap is about user-*facing* inconsistency). Renaming ~40 files of internal code + on-disk data paths buys zero user-facing value, loses git blame, and risks collaborator path breakage. The CSV export is user-facing, so the mapping happens at the export boundary only.

**Implication for agents.** Any theme adding a viewer label, axis title, or CSV export MUST use the display names above and MUST NOT rename internal keys. The display↔internal map is the single source of truth; put it in one place (a `COHORT_DISPLAY` map) and have both viewers + the F2 export layer read from it.

---

## C1 — Song genotype split schema ✅

**Reframe.** C1 is **de-aggregation, not new sub-cohort tabs.** Today several surfaces reduce Song to a single scalar by pooling across the 3 disease genotypes (e.g. the crosstable's `median_nes` via `_kxMedian()` "over ALL measured units"). C1 breaks those out per-genotype.

**The data already supports it.** NES/pathway results are keyed by `contrast`, and the 9 Song contrasts encode genotype as their name prefix (`App_{2,4,6}mo`, `Tau_{2,4,6}mo`, `ApTt_{2,4,6}mo`); `DISEASE_GROUPS = ["App","Tau","ApTt"]` is already in config. The split key is the contrast prefix — no analysis change, no payload duplication.

**Hard invariant — split axis is genotype; never pool across genotypes.** Any Song surface that reduces to a scalar reduces it **per genotype** (App / Tau / ApTt). Applies to: crosstable `median_nes` / `_kxMedian`, `peak_NES` / `peak_contrast`, and the audit verdict's single "Song" NES/LFC column. Each becomes three values, displayed as `MouseC1_App` / `MouseC1_Tau` / `MouseC1_ApTt` (C2 names).

**Timepoint reduction is context-dependent, NOT a fixed rule.** When a surface reduces a genotype's 3 timepoints to one number, median-over-timepoints is an acceptable choice — but it is per-context, not a global mandate (some surfaces keep the full per-timepoint trajectory, some peak, some median). The contract fixes only the *genotype* axis, never the timepoint reduction.

**Specificity stays unified.** `song_specificity` / `song_top_cluster` / τ are genotype-independent — one shared value, never tripled. (This is the TODO C1 rule: cell-type reference stays whole-cohort; NES + pathway split.)

**No pooled "all-genotypes" value anywhere.** Anti-shim: the pooled median is *removed and replaced* by the 3 per-genotype values. No "All genotypes" overview column coexists with the split — that defeats the purpose.

**Consumers.** C3 reads the per-genotype values as side-by-side directional (up/down) columns. B2 builds per-genotype sankeys. Both rely on the genotype tag being explicit on each contrast-derived value.

**Implication for agents.** Any theme touching a Song NES/pathway summary MUST emit per-genotype values (3), never a genotype-pooled scalar. Derive genotype from the contrast prefix (`contrast.split("_")[0]` → App/Tau/ApTt). Do not add genotype to specificity slices.

## B3 — PTM extension schema (acet/ubiq) ✅

**Finding: B3 is already built.** The PTM machinery exists and runs end-to-end for the only cohort that has acetyl/ubiquitin data (5xFAD). This is a documented convention, not pending work. The TODO's "song, 5xfad, tcell" scope is factually wrong (corrected below).

**Canonical channel vocabulary (fixed by the Incytr engine — do not invent new names).**
| Channel | PTM | Notes |
|---|---|---|
| `pr` | total proteome | required (drives prG receiver gene set); `floor_pr` applies |
| `py` | phospho-tyrosine | |
| `ps` | phospho-ser/thr | |
| `Ack` | **acetylation** | optional; engine has `Ack_FC` slot; **do NOT apply `floor_pr`** |
| `KGG` | **ubiquitination** (di-glycine remnant) | optional; engine has `KGG_FC` slot; **do NOT apply `floor_pr`** |

Driver validates `CHANNELS ⊆ {pr,py,ps,Ack,KGG}` (`incytr_commandline.R:148`). Ack/KGG are env-gated (`ACK_FILE`/`KGG_FILE`); unset ⇒ byte-identical to a phospho-only run. No Incytr package changes required.

**Output convention (anti-shim, codified in `run_pair_mode_5xfad.sh`).** Phospho-only → `wide/`; phospho+PTM → `wide_ptm/` (parallel dir, **never overwrites `wide/`**). The phospho-only result is *derived* from the PTM superset via `derive_phospho_from_ptm.py` (subtract Ack/KGG score contribution, drop the 12 PTM-only node columns), so the two are provably consistent rather than independently regenerated. Filtering (SigProb/PDS gate) runs in-place on both, AFTER derive.

**Cohort reality (corrects the TODO).**
- **5xFAD** — has Ack + KGG; `wide_ptm/` built for cortex + hippocampus. **Done.**
- **Song, T-cell** — no acet/ubiq data exists. A PTM run = byte-identical phospho-only. Nothing to generate.
- **Mukesh** — proteomics report *is* `STY-AcK-KGG` (data exists), but Mukesh has **no incytr_pair pathway at all** (bulk-MEA only, no snRNA spine). A Mukesh incytr pathway is not feasible, so PTM-on-Mukesh is out of scope.

**Consumers.** B4 (kinase→pathway) and the viewer consume the 5xFAD `wide_ptm/` schema directly. The 12 PTM-only node columns and `Ack_score`/`KGG_score` are the surfaces a viewer PTM layer would read.

**Implication for agents / meta-plan.** B3 is NOT a Wave-1 contract producer with pending edits — it's a settled convention. Any theme adding PTM must use the channel names + `wide_ptm/` convention above verbatim.

## B5 — Backbone / pathway reduction ✅

**Purpose.** A reduction + ranking layer on top of the existing per-row gate, producing the pathway table B2's sankey consumes. It surfaces the *backbone* — paths that recur across timepoints — instead of the widest per-pair enumeration.

**Input.** Gated pathway rows: the canonical `filter_significant_paths.py` floor (`SigProb > 0.1` in either condition AND `|PDS| >= 0.2`, uncapped). Schema: `Sender.group, Receiver.group, Ligand, Receptor, EM, Target, PDS, SigProb_<cond>`.

**Path identity** = the 6-tuple `(Sender.group, Receiver.group, Ligand, Receptor, EM, Target)`.

**Reduction.**
1. Collapse a path's occurrences across a condition's timepoints → `n_timepoints_present` (count passing the gate). Then across conditions → `n_conditions_present`.
2. **Backbone = high `n_timepoints_present`.** Cholinergic.Neurons is the priority anchor (paths into it surfaced first).

**Ranking — recurrence-first (lexicographic), NOT a composite score.** `backbone_rank` sorts by `n_timepoints_present` desc, then `n_conditions_present` desc, then `|PDS|` desc as tiebreak. A path in 3/3 timepoints always outranks one in 2/3 regardless of PDS. Rationale: matches the TODO's "common across all timepoints **first**, then conditions"; stays explainable for a publication deliverable (a composite `recurrence × |PDS| × specificity` is opaque and hand-weighted).

**Annotations — NEVER hard filters.** Two derived columns inform `backbone_rank` softly and drive interactive viewer filtering, but **drop no rows**:
- `mean_gene_specificity` — mean cell-type specificity of the 4 position genes (Ligand in Sender type; Receptor/EM/Target in Receiver type), derived from `aggexp.csv` (per-gene × 31-cluster share — same method as kinase `song_specificity`; the orphaned `specificity.py` is gone but the `aggexp` substrate is on disk for all cohorts).
- `min_cell_count` — min n_cells across the path's Sender + Receiver clusters.

Rationale for annotation-not-filter: a hard specificity/cell-count floor would **delete recurrent low-specificity or sparse-cluster backbone paths** — exactly the Cholinergic anchor case (1 cell/condition at 2mo; broadly-expressed key ligands). That is the opposite of the goal. Keep all rows; flag honestly; let the viewer filter on demand.

**Output schema (B2 consumes).** Gated path columns **+** `n_timepoints_present, n_conditions_present, backbone_rank, mean_gene_specificity, min_cell_count`.

**Implication for agents.** B5 produces this reduced table; B2's sankey ranks/filters off `backbone_rank` + the annotations and must not re-implement its own reduction. The gate stays in `filter_significant_paths.py` (do not raise its cutoffs — see CLAUDE.md). Per-gene specificity is computed fresh from `aggexp` for *all* pathway genes, not just kinases.

## F1 — Signed-sort convention ✅

**Decision.** Every numeric table column sorts by **signed value**, never `|value|` — both viewers, all tables (crosstable — the current `Math.abs` offender — kinase explorer, incytr pathways, attribution verdict, and any new table).

- **Default direction: descending** (biggest up-regulated on top). The existing header-click **asc/desc toggle is the mechanism for the negative tail** — one click flips to biggest down-regulated on top. No magnitude information is lost relative to abs-sort.
- **Nulls / NaN always sort last**, regardless of direction.
- Scope is the *sort comparator only* — sign display (already present in the crosstable) is unaffected.

**Implication for agents.** Any table-adding theme uses the signed comparator. Do not introduce `Math.abs` in a sort key. The shared comparator + null-last rule should live in one helper both viewers call.

## F2 — CSV export standard ✅

**Decision.** Every data table has a CSV export (audit existing — `exportKinaseCsv` already covers the kinase explorer — and complete the rest), with one standardized format.

- **Export scope:** the **currently filtered + sorted rows** (the user's working set / what they see), not the full underlying table.
- **Numeric precision:** **raw full-precision values**, not the rounded/colored display values — the export feeds downstream analysis.
- **Column headers:** display names per C2 (`MouseC1_lfc`, not `song_lfc`) — this is the user-facing export boundary where the C2 internal→display mapping is applied.
- **Columns included:** data columns only. **Exclude UI-only columns** — color swatches, badge HTML, sparkline/canvas cells, action buttons.
- **Filename:** `<cohort_display>__<table>.csv`, lowercased (e.g. `mousec1_app__crosstable.csv`).

**Implication for agents.** Any table-adding theme ships its export in this format. The C2 `COHORT_DISPLAY` map is the single source for both the header rename and the filename prefix. The F1/F2 sweep applies these uniformly *after* table-adding themes land (per meta-plan Wave 4), not concurrently with them.
