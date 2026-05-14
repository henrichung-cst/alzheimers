# Mukesh / NBB Human AD Ingest Plan

One-off converter that reshapes the Mukesh lab's DIA phospho + total-proteomics
data (Human AD vs Control, Netherlands Brain Bank, 17 donors) into the
Song-shaped artifacts consumed by the existing live pipeline (Stage 2 onward).

This script is **single-use / rarely-used**. It is not a generalized importer.
The expectation is hand-curation of additional methodologies into our existing
schema, not extension of the pipeline to absorb arbitrary formats.

## Scope

- Reshape only. Stage 2 (MEA) onward is unchanged for now.
- Stage 1 IRS plex normalization is bypassed — DIA is label-free, single-batch.
- One contrast: `AD vs CTRL` (binary).
- No donor metadata available → no covariates in OLS.
- Cell-type attribution is out of scope for this task. When re-enabled later,
  this dataset uses SEA-AD only (no WMB, no Song within-cohort).
- Viewer integration is out of scope for this task. When re-enabled, this
  becomes a "Human" data slice alongside Song.

## Locked decisions (from design discussion)

| Topic | Decision |
|---|---|
| Stoichiometry | Keep (`log2 phospho − log2 protein`) |
| Outlier rule | Reuse `config.OUTLIER_ZSCORE_THRESH` on 17 samples post-log2 |
| Canonical accession | **UniProt Swiss-Prot canonical isoform**, fallback to longest isoform containing the peptide, `isoform_specific=True` tag for fallbacks |
| Site ID format | `{canonical_accession}_{S\|T\|Y}{position}` (matches Song convention) |
| Audit columns | Preserve original `PG.ProteinGroups` and `EG.ProteinPTMLocations` per row — zero information loss |
| UniProt cache | `data/datasets/mukesh/analysis_cache/uniprot/canonical_map.json` |
| UniProt source | Swiss-Prot reviewed only, organism 9606 |
| Cache key | Gene symbol → `{canonical_acc, sequence, isoforms: [{acc, length}, ...]}` |
| Cache refresh | Never auto-refresh; manual delete to rebuild |
| pY non-phospho rows | Drop from reshape (raw on disk preserved) |
| pY non-Y phosphos | Drop. Considered exclusive from IMAC S/T file. No cross-routing. |
| Edge case policy | **Measure first, decide after.** Phase B emits diagnostics, no drops beyond the two above. |
| Multi-phospho peptides | One row per `[Phospho (STY)]` marker, sharing peptide quantity (Song convention) |

## Empirical inputs (measured 2026-05-13)

- IMAC peptide report: 79,105 rows; 52,771 phospho-bearing (66.7%);
  residue mix S 72.5% / T 23.2% / Y 4.3%; 312 multi-phospho peptides (0.6%).
- pY peptide report: 61,155 rows; **2,042 phospho-bearing (3.3%)**; among
  phospho instances Y 86.8% / S 9.9% / T 3.3% (S+T → drop); 9 multi-phospho
  peptides (0.4%).
- Total proteomics protein report: 9,259 protein groups × 17 samples
  (10 AD, 7 CTRL).
- Sample IDs encoded in column names. No external sample-list workbook.
- Unique genes in phospho data: 4,841 (IMAC). pY gene set is a near-subset.
- Multi-isoform peptide rows: 39.4% of IMAC phospho rows. Only 0.7% of genes
  have peptides that genuinely distinguish isoforms.

## File tree (target)

```
alz/ingest_mukesh.py                                  # single script, multi-phase CLI

data/datasets/mukesh/analysis_cache/uniprot/
  canonical_map.json                                  # built by Phase A
  unresolved_genes.csv                                # genes UniProt did not resolve
  fetch.log                                           # per-gene fetch trace

outputs/reports/data_ingest_human/
  sample_mapping.csv                                  # Phase C: 17 rows (sample_id, group)
  synthesis_audit.csv                                 # Phase B: per-row diagnostics
  synthesis_audit_summary.txt                         # Phase B: prevalence table

outputs/reports/kinase_attribution_human/
  stoichiometry_matrix.csv                            # Phase C
  raw_phospho_normalized.csv                          # Phase C
  synthesis_dropped.csv                               # Phase C: rows excluded with reason
  ingest_manifest.json                                # Phase C: provenance + parameters
```

The `_human` suffix on output dirs keeps these isolated from the Song
artifacts. Stage 2+ will read from these paths when invoked under a
human-NBB Kedro env (deferred).

## CLI surface (single script)

```bash
python alz/ingest_mukesh.py --uniprot-cache          # Phase A
python alz/ingest_mukesh.py --diagnose               # Phase B (requires Phase A)
python alz/ingest_mukesh.py --reshape                # Phase C (requires Phase B review)
python alz/ingest_mukesh.py --summary                # Print summary of cached outputs
```

Phase C is gated: it refuses to run until a human (the user) has reviewed
the Phase B audit and added `policies_reviewed: true` to a small policy file
documenting the chosen edge-case handling. This is the "no unilateral drops"
guardrail.

---

## Phase A — UniProt canonical-isoform cache

### Purpose

Build a permanent, idempotent map from gene symbol to UniProt's canonical
Swiss-Prot accession + canonical sequence + list of isoforms. Used by both
Phase B (position recovery) and Phase C (motif extraction).

### Inputs

- `data/datasets/mukesh/proteomics/*.csv` — for `PG.Genes` enumeration
- `data/datasets/mukesh/phospho/IMAC/*.csv`
- `data/datasets/mukesh/phospho/pY/*.csv`

### Algorithm

1. **Collect gene symbols.** Stream each input CSV, take `PG.Genes` column,
   union into a set. Strip whitespace; ignore empty. Expected size ~5k.
2. **Load existing cache** if `canonical_map.json` exists. Compute the diff
   `{genes_in_data} - {genes_in_cache}` so we only fetch the new ones.
3. **Fetch from UniProt REST.** For each missing gene, single request:
   ```
   GET https://rest.uniprot.org/uniprotkb/search
     ?query=gene_exact:{SYMBOL}+AND+organism_id:9606+AND+reviewed:true
     &fields=accession,sequence,cc_alternative_products,gene_primary,protein_name
     &format=json
     &size=10
   ```
4. **Resolve canonical accession.**
   - If exactly one Swiss-Prot hit → use that accession + sequence.
   - If multiple hits → prefer the one whose `accession` appears in the
     observed `PG.ProteinGroups` for this gene. If still ambiguous, prefer
     the entry with `gene_primary == SYMBOL`. If still ambiguous, log to
     `fetch.log` and pick the first; flag `ambiguous_canonical: true` in the
     cache entry for that gene.
   - If zero hits → write to `unresolved_genes.csv` with reason
     `no_reviewed_human_hit`. Skip.
5. **Resolve isoform set.** Parse `cc_alternative_products` from the chosen
   entry. Record `[{acc: "P10636-1", length: 758, is_canonical: true}, ...]`.
   For each non-canonical isoform UniProt lists, fetch its sequence in a
   second call (FASTA endpoint, batched). Some isoforms differ from the
   canonical by only inclusion/exclusion of a single exon — we need the
   actual sequence to handle isoform-specific peptides in Phase B.
6. **Write atomically.** Tempfile + os.replace into `canonical_map.json` so
   crashes don't corrupt the cache.
7. **Concurrency.** `ThreadPoolExecutor(max_workers=8)` for the per-gene
   calls. UniProt rate limit is 200 req/s; 8 workers is safe.
8. **Retry.** Per request: 3 attempts with exponential backoff (1s, 4s, 16s)
   on 5xx / network errors. 4xx are terminal — log + continue.

### Output schema

`canonical_map.json` keyed by gene symbol:

```json
{
  "MAPT": {
    "canonical_accession": "P10636",
    "canonical_sequence": "MAEPRQ...",
    "canonical_length": 758,
    "protein_name": "Microtubule-associated protein tau",
    "isoforms": [
      {"accession": "P10636-1", "length": 758, "is_canonical": true,  "sequence": "MAEPRQ..."},
      {"accession": "P10636-2", "length": 352, "is_canonical": false, "sequence": "MAEPRQ..."},
      ...
    ],
    "fetched_at": "2026-05-13T15:04:00Z",
    "ambiguous_canonical": false
  },
  ...
}
```

### Failure handling

- Network / 5xx: 3 retries then log to `unresolved_genes.csv` with
  reason `network_error`. Re-runnable.
- 404 / no reviewed hit: log reason `no_reviewed_human_hit`. These
  rows will fall through to Phase B's Option-A fallback (first-listed
  accession from `PG.ProteinGroups`) and be tagged `canonical_unresolved=true`.
- Ambiguous canonical choice: pick + tag; surfaces in Phase B audit.

### Expected cost

- ~5k unique genes × ~1 call each (+ isoform FASTA fetches) ≈ ~6k requests.
- At 8 concurrent workers, ~5 min wall time on first run.
- Cache size: ~80 MB (sequences dominate).
- Idempotent. Subsequent runs hit only unresolved genes.

### Acceptance criteria

- `canonical_map.json` exists and parses as JSON.
- Coverage report: ≥98% of unique genes resolved. Sub-98% triggers warning.
- Sanity check on 5 hard-coded controls: MAPT → P10636, APP → P05067,
  SNCA → P37840, GAPDH → P04406, ACTB → P60709.
- `unresolved_genes.csv` exists (even if empty) for downstream phases.

---

## Phase B — Measure-first diagnostic pass (no policy decisions yet)

### Purpose

Run peptide-position recovery + canonical-accession resolution on every
phospho-bearing row in both IMAC and pY files, emit a per-row audit, and
print a prevalence table. No drops beyond the two locked above
(pY non-phospho rows; pY non-Y phosphos).

The output of this phase is **input to a policy review**, not pipeline
artifacts. Phase C is gated on review.

### Inputs

- Phase A cache: `canonical_map.json`
- IMAC + pY CSVs

### Algorithm (per phospho-bearing input row)

1. **Locked drops.**
   - In pY file: skip rows where `EG.ModifiedSequence` has no `[Phospho (STY)]`.
   - In pY file: per phospho marker, drop the marker if the residue is S or T.
     If a row has zero markers left after this filter, skip the row.
2. **Canonical accession lookup.** From `PG.Genes` look up the cache entry.
   - If gene unresolved → fall back to first accession in `PG.ProteinGroups`;
     record `canonical_unresolved=true`. Skip steps 3-5; emit a stub audit row.
3. **Peptide containment in canonical sequence.**
   - Find all occurrences of `PEP.StrippedSequence` in the canonical sequence.
   - Record `canonical_match_count` (0, 1, ≥2).
   - If `canonical_match_count == 1`: position recovery proceeds against
     canonical.
   - If `canonical_match_count == 0`: try each non-canonical isoform in
     descending length order; take the first that contains the peptide
     exactly once. Record `isoform_match_acc`, set `isoform_specific=true`.
   - If `canonical_match_count >= 2`: record but do not pick. Emit all match
     starts to the audit. (Policy decision later: drop, take first, take all,
     or use modified-sequence flanking to disambiguate.)
   - If no isoform contains the peptide either: record `peptide_not_in_any`.
4. **Position recovery per phospho marker.** For each `[Phospho (STY)]`:
   - Parse peptide-relative position from `EG.ModifiedSequence` (strip
     non-phospho mods, count residues left of the marker).
   - Compute absolute position = `match_start + peptide_relative_position`.
   - Read `computed_residue = canonical_sequence[absolute_position - 1]`
     (1-indexed convention).
   - Record `residue_matches = (computed_residue == marker_residue)`.
5. **Cross-check vs `EG.ProteinPTMLocations`** (IMAC only — pY lacks this
   column). For each marker, compare computed `(residue, absolute_position)`
   to the parsed `EG.ProteinPTMLocations` entry. Record
   `spectronaut_agrees=true|false|n/a`. Disagreements are loud — they would
   indicate either a peptide-mismatch bug or that Mukesh's group used a
   different reference (e.g. non-canonical isoform numbering).

### Audit row schema (`synthesis_audit.csv`)

One row per (input row × phospho marker). Columns:

| Column | Type | Description |
|---|---|---|
| `track` | str | `IMAC` or `pY` |
| `source_row_idx` | int | 0-based row index in the source CSV |
| `gene` | str | `PG.Genes` |
| `pg_protein_groups` | str | original `PG.ProteinGroups` (audit) |
| `pep_stripped_sequence` | str | peptide |
| `eg_modified_sequence` | str | original modified-sequence string (audit) |
| `eg_protein_ptm_locations` | str | original PTM-locations string (IMAC only; empty for pY) |
| `canonical_accession` | str | from cache, or first-listed if unresolved |
| `canonical_unresolved` | bool | true if gene not in cache |
| `canonical_match_count` | int | matches of peptide in canonical sequence |
| `canonical_match_starts` | str | semicolon-joined 1-indexed start positions |
| `isoform_match_acc` | str | populated only when canonical_match_count == 0 |
| `isoform_specific` | bool | true if peptide is in non-canonical only |
| `peptide_not_in_any` | bool | true if no isoform contains the peptide |
| `peptide_phospho_position` | int | 1-indexed position within peptide |
| `absolute_position` | int | 1-indexed position within parent protein |
| `computed_residue` | str | letter at absolute_position in parent sequence |
| `marker_residue` | str | residue immediately preceding `[Phospho (STY)]` in modseq |
| `residue_matches` | bool | computed_residue == marker_residue |
| `spectronaut_position` | int | parsed from EG.ProteinPTMLocations (IMAC only) |
| `spectronaut_agrees` | str | true / false / n_a |
| `phospho_count_in_peptide` | int | number of `[Phospho (STY)]` markers in modseq |
| `multi_phospho` | bool | phospho_count_in_peptide > 1 |
| `site_id_proposed` | str | `{accession}_{residue}{position}` — what Phase C *would* emit |

### Summary table (printed + written to `synthesis_audit_summary.txt`)

Counts and percentages broken down by track for:

- Total phospho-bearing source rows kept after locked drops
- Genes unresolved by UniProt cache
- `canonical_match_count`: 0 / 1 / ≥2
- `isoform_specific` rows
- `peptide_not_in_any` rows (potential bugs or sequence-data issues)
- `residue_matches == False` (potential bugs)
- `spectronaut_agrees == False` (IMAC only — potential numbering disagreement)
- `multi_phospho` rows
- Per-residue breakdown of `computed_residue`

### What we expect to learn

- **Prevalence of ambiguous peptide matches** (`canonical_match_count >= 2`).
  Hypothesis: <5%. Decision: drop vs first-match vs flank-disambiguation.
- **Prevalence of `peptide_not_in_any`.** Hypothesis: <2%. Likely either
  signal-peptide cleavage, gene-symbol mismatch with UniProt, or
  contamination. Decision: drop with reason logged.
- **Prevalence of `residue_matches == False`.** Hypothesis: ~0%. Non-zero
  means there is a bug in our parser or a real mismatch between Spectronaut's
  ID and the canonical sequence. Decision: investigate before Phase C.
- **Prevalence of `spectronaut_agrees == False`** (IMAC). Hypothesis: low,
  but if non-trivial it tells us Mukesh's reference proteome is not
  Swiss-Prot canonical and we may need to revisit Q1.

### Acceptance criteria

- `synthesis_audit.csv` is written and row count equals number of phospho
  markers after locked drops.
- Summary table printed.
- Script exits cleanly. **No `stoichiometry_matrix.csv` or
  `raw_phospho_normalized.csv` produced.**

---

## Phase C — Reshape into Song-shaped artifacts (gated)

### Purpose

Apply the policies chosen after Phase B review, then emit the artifacts
Stage 2 expects.

### Gating

`alz/ingest_mukesh.py --reshape` reads
`docs/audits/mukesh_ingest_policies.yml` (path TBD) and refuses to proceed
unless `policies_reviewed: true` is set and every edge-case category has an
explicit policy. The file documents the chosen handling per category, with
a one-line rationale each.

Template:

```yaml
policies_reviewed: true
reviewed_by: henri
reviewed_at: 2026-05-XX
edge_cases:
  canonical_match_count_zero:
    policy: fallback_to_isoform   # drop | fallback_to_isoform
    rationale: "..."
  canonical_match_count_multi:
    policy: drop                  # drop | first_match | flank_disambig
    rationale: "..."
  peptide_not_in_any:
    policy: drop
    rationale: "..."
  residue_mismatch:
    policy: drop                  # must be drop if observed; investigation pre-req
    rationale: "..."
  spectronaut_disagreement_imac:
    policy: ...                   # TBD based on prevalence
    rationale: "..."
  canonical_unresolved:
    policy: ...                   # likely drop with sidecar
    rationale: "..."
```

### Algorithm

1. **Sample manifest.** Parse the 17 sample columns from each input file's
   header. Extract `(sample_id, group)` from the `*-AD-NN` / `*-CTRL-NN`
   suffix. Write `outputs/reports/data_ingest_human/sample_mapping.csv`.
   Cross-check sample IDs are consistent across the three input files.
2. **Apply policies.** Read `synthesis_audit.csv`. Drop rows according to
   `mukesh_ingest_policies.yml`. Log drops to `synthesis_dropped.csv` with
   reason column.
3. **Protein quant.** Read the protein report; log2 transform `PG.Quantity`
   columns; key by `PG.ProteinAccessions` (canonical-aligned to the choice
   from Phase A). No IRS. Write a tidy DataFrame indexed by accession with
   17 sample columns.
4. **Phospho quant per track (IMAC, pY).** Read peptide reports;
   log2 transform the appropriate quantity column (TBD: `PEP.Quantity`,
   `PEP.MS2Quantity`, or `EG.TotalQuantity` — needs a separate small
   decision after looking at completeness/sparsity of each). Aggregate
   to the chosen site IDs (one row per phospho marker after Phase B/C
   resolution). Sites that share a peptide inherit the same quant.
5. **Outlier exclusion.** Apply `config.OUTLIER_ZSCORE_THRESH` within-group
   robust z-score rule on the 17 log2-protein samples. Same rule from
   `data_ingest.py`. Write `sample_exclusions.csv`. Excluded samples are
   dropped from both protein and phospho quant going forward.
6. **Motif extraction (±7).** For each surviving site, slice the canonical
   sequence around `absolute_position`. Pad with `_` for sites near
   termini. Write `motif` column.
7. **Stoichiometry.** Join phospho rows to protein rows on canonical
   accession. Compute `log2_phospho - log2_protein` per sample. Sites where
   the parent protein is not quantified → emit a `stoichiometry_dropped.csv`
   sidecar with reason `parent_protein_not_quantified`.
8. **Emit Stage-2-compatible artifacts:**
   - `stoichiometry_matrix.csv` — columns: `site_id`, `protein_id`,
     `gene_symbol`, `site_position`, `motif`, then 17 (or fewer post-outlier)
     sample columns.
   - `raw_phospho_normalized.csv` — same shape, but pre-stoichiometry log2
     phospho values.
   - These match Song's Stage-1 output schema, so Stage 2 reads them
     unmodified (modulo a small Kedro env overlay for the contrast set).
9. **Manifest.** Write `ingest_manifest.json` with input file paths + sizes
   + checksums, UniProt cache fingerprint, policy file checksum, output
   row counts, timestamp.

### Stage-2 compatibility shim

The Mukesh contrast set is `{AD_vs_CTRL}` (one element). Stage 2's factorial
OLS expects 9 contrasts. Two options:

- **Option (i):** Add a Kedro env (`conf/human_nbb/parameters.yml`) overriding
  the contrast list to just `AD_vs_CTRL` and the design matrix to
  `~ group`. Per-track config (`PHOSPHO_TRACKS`) gets a parallel entry
  pointing to the `_human` output dirs.
- **Option (ii):** Pad to 9 contrasts with NaN. Ugly. Rejected.

Option (i). Implementation deferred — Phase C as written just emits the
stoichiometry artifacts; the Kedro env overlay is a separate small commit
once Phase C is verified.

### Acceptance criteria

- All Phase C policies are documented in `mukesh_ingest_policies.yml`.
- `stoichiometry_matrix.csv` and `raw_phospho_normalized.csv` exist with
  expected column shape (5 metadata + N sample columns).
- `sample_mapping.csv` has 17 rows with `group ∈ {AD, CTRL}` and no
  duplicate sample_ids.
- `ingest_manifest.json` records every input file hash.
- Row counts in `synthesis_dropped.csv` + emitted rows == Phase B audit
  row count.

---

## Design rationale (recorded for posterity)

### Why we collapse to canonical isoform rather than label by observed isoform

The mass spectrometer measures **peptides**, not isoforms. A tryptic peptide
observed at a given m/z and retention time could have originated from any
isoform of the parent gene whose protein sequence contains that peptide.
There is no information in the MS signal that distinguishes "this came from
2N4R tau" vs "this came from 0N3R tau" when both isoforms contain the peptide.

This has three consequences that drove the design:

1. **Multi-isoform peptides are not multi-observations.** Writing a shared
   peptide out as N rows (one per isoform containing it) triplicates a single
   physical measurement. It would inflate downstream sample sizes and break
   the independence assumption OLS / GSEA rely on. Only peptides that are
   genuinely **isoform-specific** — i.e. their sequence is contained in
   exactly one isoform of the gene — carry isoform identity in the data.
   Phase B confirmed these are ~2.4% of IMAC markers and ~0.7% of pY markers.
   Those rows are labeled with the specific isoform's accession; the rest
   are labeled canonical.

2. **Motif extraction is isoform-invariant for shared peptides.** Kinase
   enrichment uses a ±7 amino-acid window around the phospho residue. That
   window is determined by the exon containing the residue plus a few
   flanking residues. Any isoform that contains the peptide also contains
   its flanking sequence (otherwise the peptide wouldn't match). So picking
   canonical vs any other shared-containing isoform yields the **same
   motif**. Canonical is therefore a labeling convention; it is not a
   biological claim about which isoform produced the signal.

3. **Site identity must be stable across datasets.** `P10636_S113` means the
   same residue whether read here, in Song's mouse data (after homology
   mapping), in PhosphoSitePlus, or in a future Mukesh cohort. If we
   labeled by whichever isoform Mukesh's reference FASTA happened to use,
   site IDs would become study-specific artifacts and could not be joined
   across datasets. Aggregation across peptides covering the same residue
   (different missed cleavages, terminal trimming, charge states) also
   relies on a canonical-keyed site ID — otherwise the same physical S
   ends up in two site rows because two peptides got assigned to two
   different isoforms.

The trade-off accepted: ~5.8% of IMAC sites will carry a position number
different from the one in Mukesh's source spreadsheet, because their
search FASTA used non-canonical isoforms (notably 0N3R for tau, per the
filename `…allTauIsoform…`). The peptide identity, residue identity, and
motif are unchanged. This shows up as `spectronaut_agrees=false` in the
Phase B audit. It is not a residue mismatch (`residue_matches=true` for
all of these); it is purely a numerical-label offset. Cross-referencing
back to the Mukesh table by `(gene, peptide, peptide-relative position)`
remains unambiguous; cross-referencing by `(gene, position)` would
mislead.

### Why we cannot answer "are isoform signals correlated?" from this data

Each protein-report `PG.Quantity` is the summed signal across all
isoforms whose peptides got rolled into the protein group at
quantification time. Isoform structure is already collapsed before we
see the data. Recovering per-isoform abundance from a shotgun DIA
experiment without targeted isoform-specific enrichment is essentially
impossible — it would require paired long-read RNA-seq plus a strong
deconvolution model. Out of scope here. The literature (e.g. tau 3R/4R
in tauopathies, MAP2 splicing in neuronal maturation) confirms isoform
ratios are not constant across tissue, cell type, or disease, so the
canonical collapse hides isoform-shift effects we cannot recover anyway.
This is a known limitation, not a fixable one with bulk MS alone.

## Out of scope (explicit non-goals)

- Generalizing the pipeline to ingest arbitrary DIA / DDA formats.
- Wiring the Mukesh data into the unified viewer (separate task, future).
- Cell-type attribution on Human data (separate task, future; SEA-AD only
  when re-enabled).
- Donor-level covariate modeling (no metadata available).
- Stage 2 (MEA enrichment) execution on Human data (separate task; this
  plan only produces the inputs).

## Open questions deferred to after Phase B review

- Which quantity column in the peptide reports to use as the phospho
  intensity (`PEP.Quantity` vs `PEP.MS2Quantity` vs `EG.TotalQuantity`).
  Needs a sparsity / completeness comparison after Phase A is in place,
  since we'll be reading the file anyway.
- Whether to filter sites by minimum sample coverage (e.g. quantified in
  ≥N of 17 samples) before emitting Phase C artifacts.
- Whether `tau allTauIsoform` upstream normalization conflicts with our
  per-sample log2 step (revisit if MAPT-row diagnostics look unusual in
  Phase B / C).

## Status

- [x] Design discussion complete (Q1–Q5 resolved)
- [x] Plan written (this doc)
- [ ] Phase A implemented
- [ ] Phase A run + cache populated
- [ ] Phase B implemented
- [ ] Phase B run + audit reviewed
- [ ] Policies documented in `mukesh_ingest_policies.yml`
- [ ] Phase C implemented
- [ ] Phase C run + artifacts validated
- [ ] Stage-2 Kedro env overlay added (deferred mini-task)
