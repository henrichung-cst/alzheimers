# Motif-peer narrowing of kinase cell-type attribution

Replaces `alz/cross_reference/family_transcript_discrimination.py` and its outputs.

Reports, per (kinase, cell type, cohort), **how many of a kinase's motif-confusable peers
are also detected there** — narrowing the set of kinases that could have produced MEA
signal attributed to that cell type. Carries the result into the viewers and onto Incytr
pathway rows.

## Intent

MEA scores a kinase by motif enrichment over a ranked phosphosite list. Kinases with
near-identical motifs receive near-identical NES; the assay cannot say which produced the
signal.

For Song this limitation is total. `outputs/reports/kinase_attribution/mea_stoichiometry.csv`
is one row per (kinase, contrast, track) — **there is no Song per-cell-type MEA** — so the
bridge copies one bulk NES onto all 31 `owning_cluster` values. Every Song cell-type
attribution rests on expression evidence, not a cell-type-resolved score.

Transcript detection is an independent constraint: a kinase not transcribed in a cell type
cannot be responsible for signal attributed there. Counting how many motif peers survive
that constraint in a given cell type narrows the candidate set.

Enabling fact: Song's 31 cell-type strings are byte-identical to Incytr's `Sender.group` /
`Receiver.group`, so the call attaches to pathway rows with no crosswalk.

## The measurement

For each detected (kinase K, cell type C, cohort):

- **candidates N** = 1 + number of K's *informative* motif peers, where informative means
  detected somewhere in that cohort.
- **survivors k** = 1 + number of those peers also detected in C.

Read as: *of N motif-confusable candidates for signal in C, k are transcriptionally
plausible.* **survivors = 1 is the deliverable**: K is the sole plausible source in C. A
sole-source call is *resolved* (N > 1: twins existed and were ruled out — genuine
discrimination) or *unique-motif* (N = 1: K never had a twin, so the row only confirms
detection). The report separates the two; the resolved count is the analysis's real output.

No floor. Every detected kinase is emitted, including unique-motif kinases (N = 1), because
the goal is to show which kinases are attributable to a specific cell type and a
unique-motif kinase is trivially unambiguous. Suppressing short candidate lists would delete
the cleanest attributions.

### Peer set definition

- Centered cosine over Kinase Library PSSMs (`kl.get_matrix(name, mat_type="norm")`, reused
  from `alz/viewer/shared/payload_helpers.py::_build_kinase_motifs`, including its alias
  map). Centering removes shared background composition; without it every pair looks similar.
- **Cut = 0.60**, roughly the 93rd percentile of the centered-cosine distribution.
- **Within kinase type only.** `ser_thr` (299) and `tyrosine` (78) have different position
  ranges and are never motif-confusable. Hard partition, not a parameter.
- **Per-kinase neighbour sets, not disjoint clusters.** Settled empirically: single-linkage
  transitive closure collapses — at cut 0.70 the largest ser/thr component is 188 of 299
  kinases, and even at 0.85 it is 30. "The only one of 188" is not a claim. Local
  neighbourhood density is 0.78–0.82, so neighbour sets are near-cliques and behave well.

### Reporting the two kinds of sole-source separately

Sole-source (survivors = 1) must not be reported as one number: it lumps genuine
discrimination with rows the analysis contributed nothing to. In Song, 471 sole-source rows
split into 211 *resolved* (twins existed, all absent in C) and 260 *unique-motif* (no twins,
detection-only). The headline reports resolved separately; the motif cut is not tuned to
move the count, since tighter cuts shrink peer sets and inflate sole-source by ruling out
less. Candidate-pool thinness is exposed by median N per cohort — a cohort detecting few
kinases (T-cell, median N = 10) inflates sole-source by absence, not discrimination.

## Naming constraint (hard)

`exclusivity` is **taken**. `alz/bulk_mea/exclusivity_tier.py` defines a shipped
kinase-level tier — "is this kinase confined to few cell types?" via inverse-Simpson
`effective_n` — driving the confidence pill in all three cohorts. This work must not reuse
the word. Columns are the two plain counts: `motif_peers_detected` (k),
`motif_peers_informative` (N). No coined score.

## Scope

### In

- Cross-reference module computing (k, N) per kinase × cell type × cohort.
- **All three cohorts** — Song, 5xFAD, T-cell. Needs only a detection fraction per
  (kinase, cell type), which all three carry.
- **N is cohort-specific** — peers undetectable in a cohort are excluded from its
  denominator, because their silence carries no evidence there. The same kinase therefore
  shows different N across cohorts; each surface must label which cohort's N it shows.
  Cross-cohort comparison of N is meaningless and is not offered.
- Viewer: ratio chip (`3/16`) on surfaces already showing per-cell-type evidence
  (Attribution sub-tab, Crosstable), expanding to the peer roster with each peer's detection
  fraction. Reuses the collapse/expand pattern in `kinase_explorer.js` and existing pill CSS.
- Incytr: same chip on pathway rows, joined on (kinase, `owning_cluster`).
- Deletion of `family_transcript_discrimination.py` and
  `outputs/reports/family_transcript_discrimination/`.

### Out

- No NES reweighting, redistribution, or reranking.
- No new viewer tab.
- No Incytr filter or filter default. The Incytr CSV export honors the displayed Top-N cap,
  so a filter silently changes exported content — separate, reviewed decision.
- No change to `exclusivity_tier` or the confidence pill.
- No family grouping, primary or secondary.
- No protein or activity claim. Transcript only.

## Constraints

- Detection is `specificity.DETECTION_FRAC_MIN` (0.10), imported not redefined.
  `concentration_tier` is a compositional-share threshold and is **not** a presence call —
  the two disagree on 32% of Song kinase × cell-type rows.
- **Grain is kinase × cell type, no contrast axis.** Verified: detection is identical across
  all 9 contrasts for every (kinase, cell type), zero exceptions. Matches
  `kinase_celltype_evidence` (n = 11,687).
- Payload: two integers on `kinase_celltype_evidence` plus a peer roster keyed by kinase,
  alongside `kinase_motifs`.
- **The 15.6M-row `kinase_node_hits.parquet` is never read whole or rewritten.** The
  (k, N) lookup is ~11.7k rows and joins client-side. Any parquet inspection is DuckDB-streamed.
- Cell-type vocabularies differ per cohort. Song and 5xFAD carry 31 labels each
  (5xFAD after `cluster-*` is dropped), but the label sets are distinct — the
  matching count is coincidence, not a shared vocabulary. Per-cohort throughout.
- 5xFAD has real per-cell-type MEA (`fivexfad_celltype_mea.parquet`); Song and T-cell do
  not. The measurement means less where a cell-type-resolved score already exists — state
  this, do not flatten it.
- T-cell peer sets are thin and its 10% floor is a correctness gate; its median candidate
  count is reported so a sole-source rate inflated by shallow detection is not mistaken for
  discrimination.

## Success

1. A per (kinase, cell type, cohort) table carrying survivors k, candidates N, and the peer
   roster with each peer's detection fraction — every value auditable without a rerun.
2. Song values attach to Incytr rows with zero unmatched cell-type strings. **Met (2026-07-20):**
   all four cohorts emit the 16-col terminal-edge schema; Song sidechain slice = 1,160,725 edges,
   100% populated `kinase`+`owning_cluster`, join unmatched=0 over 31 labels.
3. An analyst reading a kinase can see the cell types where it is the sole plausible source,
   and against exactly which twins elsewhere.
4. Counts reported per cohort: rows emitted, resolved vs unique-motif sole-source, kinases
   with a unique motif, median N, and the survivor distribution. A weak resolved count is a
   finding to report, not a threshold to retune.

## Verification

- Peer sets: 377/377 Song kinases resolve a PSSM today; any regression in that count fails.
- Song: 59 of 377 kinases are detected in no cell type — assert they are excluded from every
  candidate set, since including them would manufacture free absences.
- Incytr join: assert zero unmatched cell-type strings for Song.
- Payload contract check via the existing `alz/viewer/verify_payload_contract.py`.

## Known limitations to state in the report

- Transcript ≠ protein ≠ activity. Absence is the informative direction; presence is not
  evidence of activity.
- snRNA dropout scales with cell-type depth; per-cell-type nuclei counts are not in the
  gate. A peer absent from a shallow cell type may be a depth artifact.
- The motif cut at 0.60 is a threshold on a continuum. Peers just below it remain somewhat
  confusable and are not counted.
