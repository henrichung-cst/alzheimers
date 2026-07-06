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

**Hard invariant — split axis is genotype; never pool across genotypes.** Any Song surface that reduces to a scalar reduces it **per genotype** (App / Tau / ApTt). Applies to: crosstable `median_nes` / `_kxMedian`, the kinase-explorer per-genotype NES trend pill, and the audit verdict's single "Song" NES/LFC column. Each becomes three values, displayed as `MouseC1_App` / `MouseC1_Tau` / `MouseC1_ApTt` (C2 names). (The former `peak_NES` / `trajectory` per-genotype scalars were dropped — the trend pill is classified client-side from the per-genotype NES vector; see `kinase_trend_refactor.md`.)

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

**Output convention (codified in `run_pair_mode_5xfad.sh`).** A single pair-mode product per tissue → `wide/`. For 5xFAD the PTM channels (`Ack`/`KGG`) are always scored into it, so `wide/` carries the `Ack_score`/`KGG_score` columns and the 12 PTM-only node columns alongside the phospho paths. Cohorts without acet/ubiq data (Song, T-cell) leave `ACK_FILE`/`KGG_FILE` unset and their `wide/` is byte-identical to a phospho-only run. Filtering (SigProb/PDS gate) runs in-place on `wide/`.

**Cohort reality (corrects the TODO).**
- **5xFAD** — has Ack + KGG; PTM-inclusive `wide/` built for cortex + hippocampus. **Done.**
- **Song, T-cell** — no acet/ubiq data exists. `ACK_FILE`/`KGG_FILE` unset ⇒ `wide/` is phospho-only. Nothing to generate.
- **Mukesh** — proteomics report *is* `STY-AcK-KGG` (data exists), but Mukesh has **no incytr_pair pathway at all** (bulk-MEA only, no snRNA spine). A Mukesh incytr pathway is not feasible, so PTM-on-Mukesh is out of scope.

**Consumers.** B4 (kinase→pathway) and the viewer consume the 5xFAD `wide/` schema directly. The 12 PTM-only node columns and `Ack_score`/`KGG_score` are the PTM surfaces the viewer reads (auto-surfaced when non-zero via the payload's `score_columns`).

**Implication for agents / meta-plan.** B3 is NOT a Wave-1 contract producer with pending edits — it's a settled convention. Any theme adding PTM must use the channel names + `wide/` convention above verbatim.

## B5 — Backbone grain ✅ (pivoted — engine grain, not a reduction)

**Authoritative spec:** [`theme_b/backbone_incytr_track.md`](theme_b/backbone_incytr_track.md). This section is the cross-theme contract surface only.

**Pivot.** A backbone is **a pathway with fewer nodes**, scored by the same engine on its own nodes — not a precomputed recurrence reduction. The standalone `backbone_reduction.py` / `backbone_rem_t.parquet` / `backbone_rank` approach is **removed**; recurrence is a **live within-disease timepoint filter** in the viewer, never an aggregated rank.

**Grains.** Full (L-R-EM-T) · R-EM-T (drop Ligand) · L-R-EM (drop Target) · R-EM (drop Ligand+Target). The Ligand-exclusion rationale that motivated dropping it from the old key now manifests as the R-EM / L-R-EM grain distinction (Target fans a Receptor-EM spine ~547× — a property the grain selector exposes rather than collapses).

**Emission.** Engine-integrated in `Cal_pairwise_grid` (uses in-memory `style="aFC"` fold-changes — re-scoring from `wide/`, which strips `_aFC`, drifts PDS past the floor). Per grain: `outputs/reports/incytr_pair_mode/backbone/<grain>/<contrast>_backbone_output.parquet`. The path-scoring path stays byte-identical → sce4 parity preserved.

**Floor (unchanged).** `SigProb > 0.1` in either condition AND `|PDS| >= 0.2`, plus ε=0.01. Gate lives in `filter_significant_paths.py` — **do not raise its cutoffs** (CLAUDE.md). Cholinergic-Neurons (sparse, 1-cell/condition) must survive — no hard specificity/cell-count floor.

**Recurrence.** A within-disease timepoint-combination filter on per-(entity × contrast) rows (multiselect {2mo,4mo,6mo} + all/any), evaluated within a single disease. There is **no representative/argmax PDS** — every backbone carries a real per-contrast PDS from `score_spine`.

**Implication for agents.** Consume the per-grain `*_backbone_output.parquet`, never `backbone_rem_t.parquet` or `backbone_rank` (gone). Do not reintroduce a separate reduction or a recurrence "mode" — it is a grain + a filter.

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
