# Alzheimer Datasets for InCytr: Input Mapping and Validation

Date: 2026-03-24

## Purpose

This note identifies two Alzheimer-related dataset bundles that are plausible targets for running through InCytr in this workspace:

1. `5xFAD`
2. `Song` (the Yuyu Song collaboration, most likely `yuyu01`)

The goal is to map each dataset to a concrete set of InCytr inputs and record what has been validated directly from the local files.

Session focus for 2026-03-24:

- Prioritize a `5xFAD` audit centered on `data/lucie_proteomics/`.
- Treat the packaged `data/incytr_collections/5xFAD/` tables as a comparison target, not the default truth source.
- Deprioritize `Song` for this session because its InCytr-oriented bundle is already largely assembled.

This note distinguishes four different validation levels:

- `Structural compatibility`: files have the columns, labels, and gene identifiers needed to run InCytr.
- `Post-collapse compatibility`: transcriptomics and omics tables can be harmonized after aggregation to common cell groups and conditions.
- `Cohort-level provenance`: local evidence supports that the transcriptomics and proteomics belong to the same biological study or compiled cohort.
- `Sample-level provenance`: local evidence supports exact matching between individual animals or biological samples across modalities.

These levels are not interchangeable. A bundle can be runnable in InCytr without proving exact sample matching.

## What InCytr Expects

From the local InCytr repository and legacy Yuyu runner scripts, a complete run needs:

- A transcriptomics object or normalized expression matrix plus cell metadata.
- Metadata columns that can define:
  - sender cell group
  - receiver cell group
  - cluster labels (`group.by`, often `Type`)
  - condition labels for comparison
- An input gene list with cluster-specific genes.
- Optional multi-omics tables:
  - total proteomics (`pr`)
  - phospho-Ser/Thr (`ps`)
  - phospho-Tyr (`py`)
- Optional kinase mapping data with at least:
  - `gene`
  - `site_pos`
  - `motif.geneName`

The Yuyu command-line workflow specifically expects:

- `incytr_obj.rds`
- `input_gene_list.csv` with columns `gene` and `cluster`
- deconvoluted proteomics tables whose columns encode `condition_cluster`
- `kldata.csv`, transformed into the required `gene/site_pos/motif.geneName` form

## What InCytr Produces

The final exported result is a pathway-level table from `Export_results()`.

Each row is a candidate signaling path of the form:

`Ligand -> Receptor -> Effector Molecule -> Target`

Important output content includes:

- sender and receiver cell groups
- pathway members: ligand, receptor, EM, target
- `SigProb` and related transcript-derived statistics
- fold-change summaries across conditions
- modality-specific scores:
  - transcriptomic score (`TPDS`)
  - proteomic/phospho-derived scores
  - kinase contribution when kinase data is integrated
- final multimodal pathway score (`PDS`)
- permutation-derived significance, including `p_value`
- path identifiers such as `ID_1` for merging across comparisons

In practice, this means each dataset bundle below must support:

- matched cell-group labels between transcriptomics, marker genes, and proteomics
- condition labels that can be split into `condition1` vs `condition2`
- gene identifiers that can intersect across modalities

## Dataset 1: 5xFAD

### Recommended input bundle

Transcriptomics:

- `data/gdrive_shared/lore00/transcriptomics/reclustering/named_lore00.RDS`
- supporting flat metadata:
  - `data/gdrive_shared/lore00/transcriptomics/obs_df.csv`

Marker/input genes:

- `../incytr/examples/5xad_data/Allmarkers_18groups.csv`

Proteomics:

- `../incytr/examples/5xad_data/processed_pr_WT_v2.csv`
- `../incytr/examples/5xad_data/processed_pr_5X_v2.csv`
- `../incytr/examples/5xad_data/processed_ps_WT_v2.csv`
- `../incytr/examples/5xad_data/processed_ps_5X_v2.csv`
- `../incytr/examples/5xad_data/processed_py_WT_v2.csv`
- `../incytr/examples/5xad_data/processed_py_5X_v2.csv`

Kinase data:

- `../incytr/examples/5xad_data/kldata_pspy.csv`

Do not use as direct InCytr inputs:

- raw Lucie proteomics `.sne` files in `data/lucie_proteomics/`

Those files are useful upstream source data, but they are not in the CSV format that InCytr consumes directly.

### Validation summary

Validated directly:

- `obs_df.csv` contains sample-level and cluster metadata with columns including:
  - `sample`
  - `Cluster_fine`
  - `Cluster_coarse`
- `obs_df.csv` includes 32 samples spanning both `5XFAD_*` and `WildT_*` sample names.
- The 32 transcriptomics samples decompose into:
  - genotype: `5XFAD` vs `WildT`
  - age: `03mo`, `06mo`, `09mo`, `12mo`
  - a third axis encoded as `C` vs `H`
- `Allmarkers_18groups.csv` contains:
  - `cluster`
  - `gene`
- The marker file uses genotype-specific cluster labels such as:
  - `Astrocytes_WT`
  - `Astrocytes_5X`
  - `Microglia_WT`
  - `Microglia_5X`
- The processed PR/PS/PY files all use the same eight broad cluster columns:
  - `Astrocytes`
  - `Endothelial.cells`
  - `Excitatory.neurons`
  - `Interneurons`
  - `Medium.spiny.neurons`
  - `Microglia`
  - `Oligodendrocytes`
  - `OPCs`
- `kldata_pspy.csv` already contains the required kinase fields:
  - `gene`
  - `site_pos`
  - `motif.geneName`
- `HE_gene_075_121024.csv` is a transcript-derived summary table with:
  - `gene_symbol`
  - `cluster`
  - `condition`
- `HE_gene_075_121024.csv` is already collapsed to:
  - eight broad clusters
  - two conditions: `WT` and `5X`
- The `lore00` coarse transcriptomic labels are strongly compatible with the broad labels used in the packaged InCytr inputs after simple normalization:
  - `Excitatory neurons` -> `Excitatory.neurons`
  - `Endothelial cells` -> `Endothelial.cells`
  - `Medium spiny neurons` -> `Medium.spiny.neurons`
- Gene overlap between `named_lore00.RDS` and the packaged 5xFAD transcript-derived tables is high:
  - `HE_gene_075_121024.csv`: 397 / 412 unique genes present
  - `Allmarkers_18groups.csv`: 3602 / 3726 unique genes present

Validated mismatch or caveat:

- `Allmarkers_18groups.csv` contains an extra broad group: `Other`.
- The processed PR/PS/PY files do not have an `Other` column.
- Therefore, `Other` is not fully supported as a multi-omics sender/receiver group unless additional processing is done.
- The transcriptomics source retains sample structure across age and `C/H`, but the packaged InCytr transcript-derived and omics tables have already collapsed that structure to genotype-level `WT` vs `5X`.
- The final packaged omics tables therefore cannot support sample-level matching checks on their own.
- The raw Lucie proteomics files available locally are upstream `.sne` files with coarse descriptors such as tissue, sex, and age-window, but they do not expose an accessible local sample manifest linking them to `lore00` sample IDs.

Interpretation:

- `5xFAD` is structurally compatible with InCytr if the workflow uses:
  - transcriptomics from `lore00`
  - processed omics from `../incytr/examples/5xad_data`
- The strongest current interpretation is that the packaged 5xFAD InCytr inputs are a heavily aggregated study compilation that is highly compatible with `lore00` transcriptomics and likely derived from the same or a very similar cohort after collapsing to broad cell groups and genotype-level conditions.
- The supported multi-omic broad groups are the eight groups listed above.
- `Other` should be excluded or treated as transcriptomics-only unless a matching proteomics aggregation is created.

### Provenance status

- `Structural compatibility`: strong
- `Post-collapse compatibility`: strong
- `Cohort-level provenance`: plausible but not proven directly from a local manifest
- `Sample-level provenance`: not established

### Remaining uncertainty

- `named_lore00.RDS` metadata could be accessed directly, but full Seurat class introspection remains limited by the absence of `SeuratObject`.
- The `C/H` transcriptomics axis is biologically important and appears to represent a tissue split, but the final packaged InCytr tables have already discarded that axis.
- No local file currently proves exact animal-level or sample-level matching between:
  - `lore00` transcriptomics
  - Lucie raw proteomics `.sne` files
  - `../incytr/examples/5xad_data/processed_*.csv`
- Therefore `5xFAD` should be treated as a cohort-inferred, post-collapse integration target rather than a sample-paired multiomics bundle.

### 5xFAD provenance manifest

The current local 5xFAD evidence chain can be indexed as follows.

Transcriptomics source and metadata:

- `data/gdrive_shared/lore00/transcriptomics/reclustering/named_lore00.RDS`
- `data/gdrive_shared/lore00/transcriptomics/obs_df.csv`
- `data/gdrive_shared/lore00/transcriptomics/log00.md`

Transcript-derived packaged InCytr support tables:

- `../incytr/examples/5xad_data/Allmarkers_18groups.csv`
- `../incytr/examples/5xad_data/HE_gene_075_121024.csv`

Packaged omics tables currently closest to direct InCytr input:

- `../incytr/examples/5xad_data/processed_pr_WT_v2.csv`
- `../incytr/examples/5xad_data/processed_pr_5X_v2.csv`
- `../incytr/examples/5xad_data/processed_ps_WT_v2.csv`
- `../incytr/examples/5xad_data/processed_ps_5X_v2.csv`
- `../incytr/examples/5xad_data/processed_py_WT_v2.csv`
- `../incytr/examples/5xad_data/processed_py_5X_v2.csv`
- `../incytr/examples/5xad_data/kldata_pspy.csv`

Lucie upstream proteomics files relevant to 5xFAD:

- `data/lucie_proteomics/5xFAD_cortex_PY/20260203_111707_011626_LD_cortex_pY.sne`
- `data/lucie_proteomics/5xFAD_cortex_PY/20260225_163300_011626_LD_cortex_pY_XICs.sne`
- `data/lucie_proteomics/5xFAD_cortex_total/20260302_124444_260203_LD_5xFAD_cortex_male_total.sne`
- `data/lucie_proteomics/5xFAD_hippocampus_mo6vsmo12_ACK/20260202_140346_011926_Lucie_Hippocampus_male_Mo6-12_5xFAD.sne`
- `data/lucie_proteomics/5xFAD_hippocampus_mo6vsmo12_ACK/20260225_152939_011926_Lucie_Hippocampus_male_Mo6-12_5xFAD_XICs.sne`

Lucie parsing and recovery references:

- `data/lucie_proteomics/Tcell_parsed_tables/Ensembl_sne_parser.R`
- `../incytr/docs/LUCIE_DATA_GUIDE.md`
- `../incytr/docs/notes/5xfad-data-recovery.md`
- `docs/integrations/5xfad-lucie-manifest.json`
- `code/lucie_5xfad_manifest.py`

What these files support directly:

- `lore00` provides a strong candidate transcriptomic source for the packaged 5xFAD transcript-derived tables.
- The packaged `5xad_data` CSVs are the most direct current InCytr inputs.
- The Lucie `.sne` files are plausible upstream proteomics sources, but they are not yet validated as the exact source of the packaged `processed_*.csv` tables.

What can already be inferred from the Lucie file contents:

- The local Lucie 5xFAD inventory consists of five raw `.sne` files:
  - cortex pY:
    - `data/lucie_proteomics/5xFAD_cortex_PY/20260203_111707_011626_LD_cortex_pY.sne`
    - `data/lucie_proteomics/5xFAD_cortex_PY/20260225_163300_011626_LD_cortex_pY_XICs.sne`
  - cortex total proteome:
    - `data/lucie_proteomics/5xFAD_cortex_total/20260302_124444_260203_LD_5xFAD_cortex_male_total.sne`
  - hippocampus AcK:
    - `data/lucie_proteomics/5xFAD_hippocampus_mo6vsmo12_ACK/20260202_140346_011926_Lucie_Hippocampus_male_Mo6-12_5xFAD.sne`
    - `data/lucie_proteomics/5xFAD_hippocampus_mo6vsmo12_ACK/20260225_152939_011926_Lucie_Hippocampus_male_Mo6-12_5xFAD_XICs.sne`
- The local Lucie files are not plain tabular exports that can be passed directly to the current parser script:
  - the files contain UTF-16LE-like embedded headers with run names followed by binary payload
  - `file` reports only generic `data`
  - the current parser script assumes `read_delim("YOUR_INPUT_FILE.sne")`, which is not valid for these local 5xFAD files as currently stored
- cortex pY file names embed male-only runs across multiple ages:
  - `M3`
  - `M6`
  - `M9`
  - `M12`
- the two cortex pY files expose 27 embedded raw runs each, including one pooled `M6` sample
- the cortex total proteome file exposes 28 embedded raw runs, including pooled `M6` and `M12` samples
- hippocampus AcK file names encode a narrower male-only `M6` vs `M12` design
- the two hippocampus AcK files expose 19 embedded raw runs each
- none of the embedded Lucie run names inspected locally encode explicit `WT` vs `5XFAD` labels
- therefore genotype assignment cannot currently be recovered from file names alone
- therefore the Lucie 5xFAD data is biologically relevant and likely more upstream than the packaged InCytr tables, but modality coverage and condition labeling are uneven and not yet sufficient for direct reconstruction of the packaged `pr` / `ps` / `py` condition tables

### Lucie-first audit status

If the working assumption is that `lucie_proteomics` is more up to date than the packaged `5xFAD` tables, the correct consequence is not to use Lucie directly in InCytr yet. The correct consequence is to treat Lucie as the primary upstream source that still needs explicit extraction, conditioning, and deconvolution.

Validated implications:

- `lucie_proteomics` currently has stronger upstream provenance value than the packaged `processed_*.csv` files.
- `lucie_proteomics` is not yet structurally compatible with InCytr.
- the packaged `processed_pr_*`, `processed_ps_*`, and `processed_py_*` files remain the only local tables already in the exact InCytr-ready shape:
  - one `gene_symbol` column
  - eight deconvoluted broad cell-group columns per modality
  - condition already collapsed to `WT` vs `5X`
- a Lucie-first workflow therefore requires an intermediate reconstruction layer before InCytr:
  - extract readable quantitative tables from `.sne`
  - recover or attach sample metadata including genotype
  - collapse PTM/protein rows to gene-level values in a controlled way
  - deconvolute bulk/PTM signal into transcriptomics-derived broad cell groups
  - only then split to InCytr condition tables

### Phase 1 status: manifest built

Phase 1 now has a reproducible local artifact:

- manifest generator:
  - `code/lucie_5xfad_manifest.py`
- generated manifest:
  - `docs/integrations/5xfad-lucie-manifest.json`

What Phase 1 establishes directly:

- the local `5xFAD` Lucie inventory for this audit is exactly five `.sne` files
- the manifest normalizes embedded raw run names from the UTF-16LE-readable header region of each file
- the cortex `py` pair covers:
  - `M3`
  - `M6`
  - `M9`
  - `M12`
  with 27 embedded runs per file and a pooled `M6` run
- the cortex total-proteome file covers:
  - `M3`
  - `M6`
  - `M9`
  - `M12`
  with 28 embedded runs and pooled `M6` and `M12` runs
- the hippocampus `AcK` pair covers:
  - `M6`
  - `M12`
  with 19 embedded runs per file and no pooled run detected
- all five files remain classified as:
  - `blocked_on_export_or_binary_extraction`
- genotype labels are still not explicit in the recovered embedded run names
- simple local extraction probes are negative for both a cortex file and a hippocampus file:
  - not a ZIP container
  - not a TAR container
  - no obvious Spectronaut report field names recovered from the scanned UTF-16LE-readable header window
- the current best extraction recommendation is therefore:
  - export a readable report table from Spectronaut, or use another vendor-supported export path, before attempting table-level parsing in this workspace

Current limitation of the manifest:

- it fingerprints file contents and encoded run structure
- it does not yet recover quantitative tables or full internal schema from the `.sne` payload

### 5xFAD downstream reconstruction plan

The reconstruction plan should proceed in stages and keep provenance evidence at each stage.

Stage 1: parse and fingerprint the Lucie source files

- First determine the correct extraction route for the local `.sne` files.
- Do not assume the current `Ensembl_sne_parser.R` can read them directly.
- The immediate task is to convert each `.sne` into a readable tabular export or otherwise recover its internal report tables.
- Record:
  - file path
  - modality
  - tissue
  - sex
  - age labels encoded in the run names
  - whether pooled runs are present
  - whether genotype labels are present explicitly or only inferable indirectly
  - quantitative columns recovered
  - gene identifier columns recovered
- Save a compact manifest of the parsed outputs rather than relying on file names alone.

Stage 2: characterize biological scope before any reshaping

- Determine whether each parsed Lucie file contains:
  - only male samples or mixed sex
  - one tissue or multiple tissues
  - one age pair or a broader age series
  - one genotype or multiple genotypes
- Determine whether genotype can be mapped at all from the recovered metadata.
- Treat genotype recovery as a hard requirement before trying to recreate `WT` vs `5X` InCytr inputs.
- Do not assume these files correspond to the full `lore00` cohort until that scope is explicitly checked.

Stage 3: compare parsed Lucie outputs against the packaged 5xFAD InCytr tables

- Compare gene overlap against:
  - `processed_pr_*.csv`
  - `processed_ps_*.csv`
  - `processed_py_*.csv`
- Compare condition structure:
  - whether Lucie retains age-specific and tissue-specific runs while packaged tables collapse to `WT` vs `5X`
  - whether Lucie can support genotype separation at all from recovered metadata
- Compare modality structure:
  - whether a parsed Lucie file plausibly maps to `pr`, `py`, or another PTM layer such as `AcK`
  - note that there is currently no local Lucie `ps` candidate in the verified 5xFAD inventory
  - note that `AcK` is a potentially valuable extension layer but is not part of the standard 5xFAD vignette inputs

Stage 4: decide between two reconstruction targets

- `Target A`: reproduce the existing packaged 5xFAD InCytr inputs as closely as possible
  - requires collapsing transcriptomics and proteomics to common broad groups and genotype-level conditions
- `Target B`: build a new provenance-first 5xFAD InCytr bundle from upstream data
  - may result in a narrower but more defensible analysis, for example:
    - cortex-only
    - male-only
    - selected age contrasts

Stage 5: generate InCytr-ready tables from the chosen target

- Transcriptomics:
  - derive broad cell groups from `lore00`
  - generate marker/input genes
  - define explicit condition labels compatible with the chosen biological comparison
- Proteomics:
  - summarize parsed Lucie quantitative columns to gene-level condition tables
  - explicitly define site-to-gene collapse for PTM layers
  - if multi-sample runs are retained, define how they are collapsed
  - deconvolute or otherwise harmonize bulk/PTM measurements to the transcriptomic broad cell groups expected by InCytr
  - emit final tables in the exact InCytr shape, for example:
    - `gene_symbol`
    - `Astrocytes_pr`
    - `Microglia_pr`
    - `Excitatory.neurons_pr`
    - analogous columns for each supported modality
- Kinase data:
  - reshape the relevant source to `gene`, `site_pos`, `motif.geneName`

Stage 6: preserve provenance outputs

- For every reconstructed table, write a sidecar manifest that records:
  - source files used
  - transformation script
  - biological filters applied
  - whether age, tissue, or sex axes were discarded
  - whether the final product is:
    - sample-paired
    - cohort-level
    - post-collapse only

### Lucie to InCytr execution plan

The end goal is an InCytr-ready `5xFAD` bundle that is derived as directly as possible from the Lucie upstream proteomics files while remaining explicit about every collapse and inference step.

Working principle:

- Do not optimize first for immediate InCytr execution.
- Optimize first for a reproducible reconstruction path from Lucie source data to InCytr-ready tables.
- Treat InCytr execution as the final validation stage of that reconstruction.

Phase 1: inventory and extraction readiness

Goal:

- establish exactly what each Lucie file contains and whether it can be exported locally into a readable table

Tasks:

- create a machine-readable Lucie manifest for the five local `5xFAD` `.sne` files
- record:
  - file path
  - file size
  - modality guess
  - tissue
  - age labels
  - pooled-run indicators
  - embedded raw run names
- determine the actual extraction path for these local `.sne` files:
  - direct parse in R if possible
  - export from Spectronaut if required
  - alternative recovery if the `.sne` payload is internally structured but not plain text

Deliverables:

- `5xFAD` Lucie manifest
- extraction decision note per file
- list of files that can already be exported versus files still blocked

Gate to move on:

- at least one `pr` or `py` Lucie file is recoverable into a readable quantitative table with gene or protein identifiers and sample columns

Phase 2: recovered-table audit

Goal:

- understand the biological and analytic scope of the recovered Lucie tables before any attempt to reshape them

Tasks:

- inspect recovered columns for:
  - sample identifiers
  - quantitative value columns
  - accession columns
  - gene symbol fields
  - PTM site fields
- determine whether genotype, age, tissue, and replicate structure are recoverable from:
  - column names
  - table metadata
  - associated manifests
- separate base analytical units:
  - total protein rows
  - phosphosite rows
  - acetylation rows
  - XIC-derived tables versus non-XIC tables

Deliverables:

- per-file schema summary
- explicit statement of what biological axes are directly observed versus inferred
- list of unresolved metadata dependencies, especially genotype mapping

Gate to move on:

- a recovered table has enough metadata to define at least one defensible comparison design

Phase 3: target-comparison design

Goal:

- choose the first biologically coherent Lucie-derived comparison to operationalize

Candidate targets:

- `Target A`: packaged-table replication
  - attempt to recreate genotype-collapsed `WT` vs `5X` tables comparable to the packaged InCytr inputs
- `Target B`: provenance-first restricted analysis
  - choose a narrower analysis that matches what Lucie actually supports, for example:
    - cortex-only
    - male-only
    - one modality at a time
    - selected age contrasts

Decision criteria:

- genotype labels are recoverable or defensibly mappable
- enough replicates exist for the chosen contrast
- the transcriptomics side can be aligned to the same biological scope
- the reshape burden is proportional to the value of the comparison

Deliverables:

- written comparison design
- explicit condition definitions
- explicit inclusion and exclusion rules

Gate to move on:

- one first-pass comparison is selected and documented

Phase 4: transcriptomics alignment

Goal:

- build the transcriptomics reference needed to deconvolute Lucie measurements and to run InCytr on the same biological scope

Tasks:

- start from `lore00` as the transcriptomic anchor
- define the broad cell-group vocabulary to use downstream:
  - `Astrocytes`
  - `Microglia`
  - `Excitatory.neurons`
  - `Interneurons`
  - `Oligodendrocytes`
  - `OPCs`
  - `Endothelial.cells`
  - `Medium.spiny.neurons`
- decide whether `Other`, `Unknown`, `High MT`, and related categories are:
  - excluded
  - merged
  - kept only for transcript-level analyses
- subset or collapse transcriptomics to the same biological scope as the chosen Lucie comparison
- generate the marker or signature inputs needed for deconvolution and InCytr

Deliverables:

- transcriptomics alignment spec
- broad-group mapping table
- input gene list candidate for InCytr

Gate to move on:

- transcriptomic broad groups and Lucie comparison scope are harmonized

Phase 5: Lucie reshape and deconvolution

Goal:

- convert recovered Lucie tables into gene-by-cell-group tables that match InCytr input expectations

Tasks:

- define the collapse rule from protein-group or site-level rows to gene-level rows
- document how duplicate mappings are handled
- decide whether values are summarized by:
  - mean
  - median
  - max
  - another explicit rule
- deconvolute recovered Lucie measurements using transcriptomics-derived cell-group references
- emit condition-specific modality tables in InCytr shape, for example:
  - `gene_symbol`
  - `Astrocytes_pr`
  - `Microglia_pr`
  - `Excitatory.neurons_pr`
- repeat for each supported modality:
  - `pr`
  - `py`
  - optional `AcK` extension if retained

Deliverables:

- intermediate cleaned Lucie tables
- deconvoluted modality tables
- transformation manifests describing every collapse and normalization step

Gate to move on:

- at least one modality yields two condition tables in valid InCytr-ready shape

Phase 6: compatibility validation against packaged references

Goal:

- determine whether the Lucie-derived reconstruction is compatible with existing packaged `5xFAD` inputs and where it diverges

Tasks:

- compare gene overlap against packaged `processed_*.csv`
- compare broad cell-group coverage
- compare condition definitions
- compare dynamic range and sparsity patterns
- identify whether discrepancies reflect:
  - different modality processing
  - different biological scope
  - different deconvolution assumptions
  - true dataset divergence

Deliverables:

- Lucie-versus-packaged comparison report
- decision on whether the Lucie-derived bundle is ready for InCytr or still needs refinement

Gate to move on:

- reconstructed tables are either accepted for first-pass InCytr analysis or sent back for another reshape iteration

Phase 7: first InCytr run

Goal:

- execute a limited but defensible InCytr analysis on Lucie-derived `5xFAD` inputs

Tasks:

- select one sender-receiver pair, for example:
  - `Astrocytes -> Microglia`
- use transcriptomics and Lucie-derived modality tables restricted to the chosen comparison
- run a minimal end-to-end InCytr workflow
- verify:
  - file compatibility
  - metadata compatibility
  - expected output structure
  - whether scores are numerically stable enough to interpret

Deliverables:

- first Lucie-derived InCytr output table
- run manifest
- issue log for any structural or biological failures

Phase 8: scale-out

Goal:

- extend the workflow beyond the first successful comparison

Tasks:

- add more sender-receiver pairs
- add more comparisons
- decide whether to support:
  - only the standard `pr` and `py` layers
  - or an extended Lucie-specific workflow including `AcK`
- harden the workflow into reusable scripts and validators

Deliverables:

- reusable Lucie-to-InCytr pipeline
- documented supported comparison set
- clearly versioned outputs

Immediate next actions

1. Build the Lucie manifest as a reproducible artifact rather than relying on ad hoc inspection.
2. Determine the extraction path for at least one cortex file and one hippocampus file.
3. Audit the first recovered table for quantitative fields, identifiers, and recoverable condition metadata.
4. Choose the first reconstruction target:
   - packaged-table replication
   - or a narrower provenance-first comparison
5. Only after that, start implementing the reshape and deconvolution code needed for InCytr.

## Dataset 2: Song (`yuyu01`)

### Recommended input bundle

Transcriptomics:

- `data/incytr_collections/song/transcriptomics/incytr_obj.rds`

Alternative transcriptomics references:

- `data/incytr_collections/song/transcriptomics/170_gex_celltypes_00.h5ad`
- `data/incytr_collections/song/transcriptomics/165_gex_clusters_00_named.h5ad`

Marker/input genes:

- `data/incytr_collections/song/markers/input_gene_list.csv`
- supporting references:
  - `data/incytr_collections/song/markers/allmarkers.csv`
  - `data/incytr_collections/song/markers/HEG_df.csv`

Proteomics:

- active local bundle:
  - `data/incytr_collections/song/proteomics/pr_yuyu_deconvoluted.csv`
  - `data/incytr_collections/song/proteomics/ps_yuyu_deconvoluted.csv`
  - `data/incytr_collections/song/proteomics/py_yuyu_deconvoluted.csv`
- preserved historical collaborator bundle for reference:
  - `data/incytr_collections/song/proteomics/legacy/pr_yuyu_deconvoluted.csv`
  - `data/incytr_collections/song/proteomics/legacy/ps_yuyu_deconvoluted.csv`
  - `data/incytr_collections/song/proteomics/legacy/py_yuyu_deconvoluted.csv`

Kinase data:

- `data/incytr_collections/song/kinase/kldata.csv`

This is the authoritative localized InCytr-ready Alzheimer bundle in the workspace because it now includes:

- a prebuilt `incytr_obj.rds`
- active deconvoluted proteomics in the same naming convention used by the local runner
- preserved legacy deconvoluted proteomics for downstream comparison
- a matching input gene list
- copied collaborator runner records under `data/incytr_collections/song/method_records/collaborator_incytr_snapshot/`

Operational interpretation:

- `data/incytr_collections/song/` is now the default Song workspace
- `data/gdrive_shared/yuyu01/` should be treated as upstream archive and provenance source, not the default runtime dependency
- the active local `pr` / `ps` / `py` files are regenerated with the standardized `A_obs + DESP` workflow, while the exact old collaborator outputs are preserved under `proteomics/legacy/`

### Validation summary

Validated directly:

- `input_gene_list.csv` has the required columns:
  - `gene`
  - `cluster`
- `input_gene_list.csv` uses eight broad clusters:
  - `Astrocytes`
  - `Endothelial cells`
  - `Gaba`
  - `Glut`
  - `Medium spiny neurons`
  - `Microglia`
  - `Oligodendrocytes`
  - `OPCs`
- `allmarkers.csv` encodes condition-specific cluster labels whose broad base names collapse to the same eight clusters.
- `pr_yuyu_deconvoluted.csv` uses deconvoluted columns of the form `condition_cluster`, for example:
  - `ma_2mo_WTyp_Astrocytes`
  - `fe_6mo_AppP_Glut`
- The broad clusters extracted from the PR matrix exactly match the eight clusters in `input_gene_list.csv`.
- The PR matrix exposes 24 explicit conditions:
  - `fe_2mo_AppP`
  - `fe_2mo_ApTt`
  - `fe_2mo_Ttau`
  - `fe_2mo_WTyp`
  - `fe_4mo_AppP`
  - `fe_4mo_ApTt`
  - `fe_4mo_Ttau`
  - `fe_4mo_WTyp`
  - `fe_6mo_AppP`
  - `fe_6mo_ApTt`
  - `fe_6mo_Ttau`
  - `fe_6mo_WTyp`
  - `ma_2mo_AppP`
  - `ma_2mo_ApTt`
  - `ma_2mo_Ttau`
  - `ma_2mo_WTyp`
  - `ma_4mo_AppP`
  - `ma_4mo_ApTt`
  - `ma_4mo_Ttau`
  - `ma_4mo_WTyp`
  - `ma_6mo_AppP`
  - `ma_6mo_ApTt`
  - `ma_6mo_Ttau`
  - `ma_6mo_WTyp`
- `ps_yuyu_deconvoluted.csv` and `py_yuyu_deconvoluted.csv` follow the same condition-cluster naming pattern.
- `kldata.csv` contains the source fields used by the legacy runner to derive:
  - `gene`
  - `site_pos`
  - `motif.geneName`
  via `GENE_NAME`, `site_position`, and kinase-related columns.
- `incytr_obj.rds` metadata contains:
  - `sample`
  - `Group`
  - `Sex`
  - `Genotype`
  - `Time`
- `incytr_obj.rds` contains 28 transcriptomics sample IDs such as:
  - `C199_ma_2mo_AppP`
  - `C200_ma_2mo_Ttau`
  - `C201_ma_2mo_ApTt`
  - `D093_ma_6mo_AppP`
  - `D095_ma_6mo_WTyp`
  - `E145_fe_2mo_AppP`
- `Sample_list_72mice (1).xlsx` contains animal-level proteomics metadata with:
  - age
  - sex
  - genotype
  - mouse identifier
- The proteomics workbook and transcriptomics sample names use the same mouse identifiers and biological condition encoding, for example:
  - `C199` + male + `2mo` + `APP` <-> `C199_ma_2mo_AppP`
  - `C200` + male + `2mo` + `T22` <-> `C200_ma_2mo_Ttau`
  - `C201` + male + `2mo` + `T22/APP` <-> `C201_ma_2mo_ApTt`
  - `D093` + male + `6mo` + `APP` <-> `D093_ma_6mo_AppP`
  - `D095` + male + `6mo` + `WT` <-> `D095_ma_6mo_WTyp`
- `yuyu_samplekey.csv` explicitly maps mass-spec condition IDs to the condition codes used in the deconvolution workflow.

Interpretation:

- `yuyu01` is a strong InCytr candidate and is the most deployment-ready AD dataset in this workspace.
- The broad cluster harmonization is already done.
- The local evidence supports real cohort linkage between transcriptomics and proteomics, not just label compatibility.
- The final deconvoluted PR/PS/PY matrices are grouped summaries rather than one-column-per-animal matrices, so the operational InCytr inputs are group-level rather than sample-paired.
- The proteomics tables already support many within-sex, within-age, and within-model comparisons by passing any two of the 24 condition labels into the Yuyu runner.

### Provenance status

- `Structural compatibility`: strong
- `Post-collapse compatibility`: strong
- `Cohort-level provenance`: strong
- `Sample-level provenance`: partially supported by shared animal identifiers across source materials, but the final deconvoluted InCytr input matrices are group-level summaries rather than exact sample-paired matrices

### Remaining uncertainty

- Direct `.h5ad` introspection was not performed in this environment.
- The final deconvoluted matrices collapse proteomics to condition-group summaries, so exact per-animal pairing is not preserved in the final InCytr-ready tables.

## Recommended Decision

Use these two target bundles, but treat their provenance status differently:

1. `5xFAD`
  - primary audit target for this session
  - transcriptomics anchor from `lore00`
  - proteomics/PTM source of truth should be treated as `data/lucie_proteomics/` pending extraction
  - packaged `data/incytr_collections/5xFAD/proteomics/processed_*.csv` should be treated as the current comparison target and fallback operational inputs
  - exclude or specially handle `Other`
  - if the packaged tables are used before Lucie reconstruction is complete, use them only with the explicit understanding that this is a post-collapse, cohort-inferred integration target rather than a proven sample-paired bundle

2. `Song`
  - use `data/incytr_collections/song/`
  - this is now the most complete and directly InCytr-oriented local bundle
  - treat `data/gdrive_shared/yuyu01/` as the upstream archive rather than the active workspace
  - active proteomics are the localized `A_obs + DESP` outputs, with historical collaborator outputs preserved under `proteomics/legacy/`
  - deprioritized for this session because its audit is already substantially complete

## Practical Comparison

`5xFAD` is the correct focus when the biological priority is the 5xFAD model and there is value in reconstructing a more provenance-first omics bundle from Lucie upstream files, even if that requires additional extraction and deconvolution work before InCytr can run.

`Song` is suitable when the goal is a broader AD-model comparison set with many condition combinations already encoded in deconvoluted multi-omics tables and stronger cohort provenance.

If the goal is fastest operationalization, `Song` is still simpler. If the goal is highest confidence about the upstream 5xFAD proteomics source, `5xFAD` should take priority and the Lucie reconstruction path should be treated as the main task.

## Follow-up Work

Recommended next steps:

1. Build a Lucie manifest extractor for `5xFAD` that records, per `.sne` file:
   - modality
   - tissue
   - embedded raw run names
   - ages present
   - pooled runs present or absent
   - whether genotype labels are explicitly recoverable
2. Determine the correct export path from local Lucie `.sne` into tabular form, since the current parser assumes plain-text input but the verified local files are not plain text.
3. After export, compare recovered gene identifiers and quantitative fields against `data/incytr_collections/5xFAD/proteomics/processed_*.csv`.
4. For `5xFAD`, document the biological meaning of the `C/H` axis in `lore00` and explicitly state what information is discarded when collapsing transcriptomics to the genotype-level `WT` vs `5X` comparison used by the packaged omics tables.
5. For `5xFAD`, decide whether the first Lucie-derived reconstruction target should be:
   - a packaged-table replication target
   - or a narrower provenance-first target such as cortex-only male comparisons
6. For `Song`, enumerate the initial comparison set to run first, for example:
   - `ma_6mo_WTyp` vs `ma_6mo_AppP`
   - `ma_6mo_WTyp` vs `ma_6mo_Ttau`
   - `ma_6mo_WTyp` vs `ma_6mo_ApTt`
7. Re-run object-level validation in an environment with:
   - `SeuratObject`
   - `anndata`
   - `h5py`
