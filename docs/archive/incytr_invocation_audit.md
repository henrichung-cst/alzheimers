> **Archived 2026-05-09.** This document covers the legacy shadow-fork integration code at `alz/integration/{wrappers,adapters,sidecar,tests}/` + orchestrator shells, all relocated to `~/Projects/work/incytr_integration_archive/` on 2026-05-08. Forward-looking guidance lives in `docs/incytr_remediation_plan.md`.

# Incytr Invocation Audit

This audit compares the native Incytr-style result schema visible in
`sample.xlsx` with the current Alzheimer's factorial Incytr invocation under
`alz/integration/`.

## Executive summary

`sample.xlsx` exposes the full native Incytr evidence surface for a
two-condition pathway call: transcriptomic node fold changes, condition-level
signaling probabilities, protein node fold changes, phospho node fold changes,
structural kinase evidence, modality direction counts, and final multimodal
PDS.

The simplified factorial goal is to make factorial Incytr behave like the
pairwise Incytr evidence flow while preserving the factorial model: pathway
enumeration and transcriptomic signaling are fit across the factorial design,
then direct proteome and PTM support are attached at pathway-node level and
used to compute native-style `PPDS`, `PhPDS_ps`, `PhPDS_py`,
`multimodel_score`, and `PDS` columns per contrast.

Kinase enrichment and kinase-imputed pathway expansion are useful downstream
augmentation layers, but they are no longer baseline dependencies of factorial
Incytr. Likewise, SEA-AD-to-WMB mapping is treated as an optional compatibility
mode, not as a required Incytr concept.

## Sequence information audit

The available AD evidence streams are transcriptomics, total proteomics, and
PTM/phosphoproteomics. Native Incytr expects these to meet at the pathway-node
level: every `Ligand`, `Receptor`, `EM`, and `Target` can carry a transcript
fold change, a protein fold change, and S/T or Y phosphorylation support. The
current factorial invocation only partially does this.

| Evidence stream | Available upstream data | What currently goes into factorial Incytr | What needs to be added |
|---|---|---|---|
| Transcriptomics | Song snRNA-seq h5ad at `data/datasets/song/transcriptomics/170_gex_celltypes_00.h5ad`; exported by `export_expression_factorial.py` to `factorial/expression_matrix.mtx`, `expression_metadata.csv`, and `animal_metadata.csv` | Yes. The R wrapper computes per-animal weighted-quantile expression, enumerates pathways, computes per-animal `SigProb`, fits the 9 factorial contrasts as `TPDS_*`, `SE_*`, and `pvalue_*`, and emits per-node `*_sclog2FC_*` / `*_sc_aFC_*` support plus `sc_up/down_*` counts. | Add categorical node labels if users need sample.xlsx-style label fields. |
| Total proteomics | Total proteome raw/source data from `data_ingest.py`; WMB-class proportional decomposition at `outputs/reports/deconvolution/wmb_decomposition/pr_wmb_decomposition.csv`; parent-protein matching at `outputs/reports/data_ingest/phospho_protein_matching.csv` | Yes. `export_multiomics_evidence_factorial.py` emits per-contrast `pr_log2FC` / `pr_aFC` support keyed to the factorial Incytr cell-type vocabulary. The R wrapper attaches `*_pr_*` node columns and computes `PPDS_*`. | Replace the current global fallback with truly aligned cell-type-resolved proteome values when those inputs are available. |
| S/T PTM | Stoichiometry and site-level OLS at `outputs/reports/kinase_attribution/stoichiometry_matrix.csv`, `site_level_ols.csv`, `mea_stoichiometry.csv`; WMB-class decomposition at `outputs/reports/deconvolution/wmb_decomposition/ps_wmb_decomposition.csv` | Yes as direct PTM node evidence. `export_multiomics_evidence_factorial.py` emits `ps_log2FC` / `ps_aFC` plus site metadata and site FDR. The R wrapper attaches `*_ps_*` node columns and computes `PhPDS_ps_*`. | Keep kinase-derived MEA support separate from native PTM node evidence. |
| pY PTM | pY stoichiometry and site-level OLS at `outputs/reports/kinase_attribution/stoichiometry_matrix_pY.csv`, `site_level_ols_pY.csv`, `mea_stoichiometry_pY.csv`; WMB-class decomposition at `outputs/reports/deconvolution/wmb_decomposition/py_wmb_decomposition.csv` | Yes as direct PTM node evidence. pY paths are configured, the adapter emits `py_log2FC` / `py_aFC` plus site metadata and site FDR, and the R wrapper computes `PhPDS_py_*`. | Keep pY separate from S/T rather than folding both into a single phospho score. |
| Kinase library / structural kinase evidence | Incytr `kldata_pspy.csv`, exported by `export_kldata.py`; factorial `kl_output_all_contrasts.csv` from MEA | Optional only. Set `ENABLE_KINASE_AUGMENTATION=1` to run `export_kldata.py`, `export_kl_output_factorial.py`, kinase-imputed pathway expansion, and SiK columns. | Do not require kinase enrichment for baseline factorial Incytr. Archive or keep kinase support as a downstream augmentation layer. |

### What goes in now

The simplified baseline factorial runner,
`alz/integration/run_factorial_all_pairs.sh`, executes these adapters:

1. `export_expression_factorial.py` for transcriptomics.
2. `export_multiomics_evidence_factorial.py` for total proteome, S/T PTM, and
   pY PTM node support.
3. `run_incytr_factorial_all_pairs.R` for pathway enumeration, factorial
   transcriptomic scoring, multiomics evidence attachment, and per-contrast
   native-style scores.

Optional kinase augmentation is enabled with:

```bash
ENABLE_KINASE_AUGMENTATION=1 bash alz/integration/run_factorial_all_pairs.sh
```

That optional path runs `export_kldata.py`, `export_kl_output_factorial.py`,
and `export_kinase_imputed_genes_factorial.py`. It changes the pathway
universe and adds kinase support, so it should not be treated as the baseline
factorial Incytr invocation.

### What to add

Add the missing streams in two layers.

First, add **evidence surfacing** for every final pathway row. This should be
a contrast-long table keyed by:

```text
sender, receiver, contrast, Ligand, Receptor, EM, Target, Path
```

and carrying native-like support columns:

```text
Ligand_sclog2FC, Receptor_sclog2FC, EM_sclog2FC, Target_sclog2FC,
Ligand_pr_log2FC, Receptor_pr_log2FC, EM_pr_log2FC, Target_pr_log2FC,
Ligand_ps_log2FC, Receptor_ps_log2FC, EM_ps_log2FC, Target_ps_log2FC,
Ligand_py_log2FC, Receptor_py_log2FC, EM_py_log2FC, Target_py_log2FC,
sc_up, sc_down, pr_up, pr_down, ps_up, ps_down, py_up, py_down
```

This makes the evidence visible and lets us audit whether each pathway is
supported by RNA, protein, S/T phosphorylation, or pY phosphorylation.

Second, decide whether to compute **native-style multimodal pathway scores**
per contrast:

```text
PPDS, PhPDS_ps, PhPDS_py, multimodel_score, PDS
```

The factorial wrapper now computes these as a documented contrast-aware analog
of Incytr's two-condition multiomics scoring. Optional kinase augmentation can
be added afterward, but baseline `PDS_*` is not contingent on kinase
enrichment.

### How new data should be added

New data should enter through Python adapters that write stable intermediate
files under `alz/integration/intermediates/factorial/`, not through direct
joins inside the R scorer. The R layer should consume prepared, schema-checked
tables.

Recommended adapter contracts:

1. `export_proteome_factorial.py`

   Read total-proteome evidence, preferably the WMB-class decomposition
   `pr_wmb_decomposition.csv` when we want cell-type-resolved support, or the
   bulk total proteome matrix when we only want cohort-level support. Emit
   contrast-level gene effects with explicit cell-type vocabulary:

   ```text
   contrast, cell_type, gene_symbol, pr_log2FC, pr_aFC,
   source, n_obs, provenance
   ```

2. `export_ptm_factorial.py --track st|py`

   Read `site_level_ols.csv` / `site_level_ols_pY.csv` and, where appropriate,
   `ps_wmb_decomposition.csv` / `py_wmb_decomposition.csv`. Collapse multiple
   sites per gene by a documented rule such as max absolute contrast effect,
   strongest FDR-passing site, or signed weighted mean. Emit:

   ```text
   contrast, cell_type, gene_symbol, ptm_track, ptm_log2FC, ptm_aFC,
   site_id, site_position, site_fdr, aggregation_rule, provenance
   ```

3. `export_transcript_evidence_factorial.py`

   Either extend the R wrapper to emit node-level modeled RNA effects, or add a
   Python/R sidecar that computes the same factorial contrasts over
   per-animal weighted-quantile expression. Emit:

   ```text
   contrast, cell_type, gene_symbol, sc_log2FC, sc_aFC,
   n_animals, df_resid, provenance
   ```

4. `join_factorial_evidence.py`

   Join pathway rows to the transcript/protein/PTM evidence tables for all four
   pathway node positions. This should produce the user-facing evidence table
   and the native-like direction counts.

The main design constraint is cell-type vocabulary. Baseline factorial Incytr
now takes the vocabulary from `factorial/expression_metadata.csv`. The
multiomics adapter first looks for exact omics columns in that vocabulary. If
they are absent, it uses a condition-level global mean across available omics
columns and records that assignment. This keeps factorial Incytr independent of
SEA-AD/WMB mapping. Set `ENABLE_CELLTYPE_MAPPING=1` only for the legacy
compatibility mode that maps SEA-AD subclasses onto WMB classes.

## Column-family audit

| Native Incytr evidence family | `sample.xlsx` columns | Current factorial status | Notes |
|---|---|---|---|
| Pathway identity | `Ligand`, `Receptor`, `EM`, `Target`, `Path` | Used and exported | Recomputed from IncytrDB and AD snRNA-seq; not read from `sample.xlsx`. |
| Sender / receiver | `Sender.group`, `Receiver.group`, `ID_1`, `ID_2` | Sender/receiver exported; IDs still not emitted | Wide output now includes `sender`, `receiver`, `Sender.group`, and `Receiver.group`. `ID_1` / `ID_2` remain intentionally absent until we verify native Incytr ID semantics. |
| Node labels | `Ligand.label`, `Receptor.label`, `EM.label`, `Target.label` | Still not emitted | Baseline output now carries node-level modality values and direction counts, but not categorical labels like DEG/protein/PTM evidence class. |
| Transcript node FC | `*_sclog2FC` | Emitted per contrast | Factorial computes modeled node effects for ligand, receptor, EM, and target using the same animal-level expression structures used by TPDS. |
| Condition SigProb | `SigProb_ApTt`, `SigProb_WTyp` | Internal only | Factorial computes per-animal SigProb, then OLS contrasts. Raw condition-level SigProb columns are not exported. |
| Transcript pathway score | `log2FC`, `aFC`, `TPDS` | `TPDS` emitted; `log2FC` / `aFC` not aliased | Factorial emits `TPDS_{contrast}`, `SE_{contrast}`, and `pvalue_{contrast}` in wide output, and unsuffixed `TPDS`, `SE`, `pvalue` in the contrast-long export. Pairwise `log2FC` / `aFC` are not copied because factorial uses modeled contrasts rather than one direct two-condition fold change. |
| Protein node FC | `*_pr_log2FC`, `*_pr_aFC` | Emitted per contrast | `export_multiomics_evidence_factorial.py` prepares total-protein support and the R wrapper attaches it to all four pathway nodes. |
| Protein pathway score | `PPDS` | Emitted per contrast | Computed as the mean transformed protein node support for each pathway and contrast. |
| Phospho-S/T node FC | `*_ps_log2FC`, `*_ps_aFC` | Emitted per contrast | S/T PTM support is direct node evidence, not the external kinase-support score. |
| Phospho-S/T pathway score | `PhPDS_ps` | Emitted per contrast | Computed separately from pY support. |
| Phospho-Y node FC | `*_py_log2FC`, `*_py_aFC` | Emitted per contrast | pY paths are configured and handled separately from S/T PTM. |
| Phospho-Y pathway score | `PhPDS_py` | Emitted per contrast | Computed separately from S/T support. |
| Direction counts | `sc_up/down`, `pr_up/down`, `ps_up/down`, `py_up/down` | Emitted per contrast | Counts how many of the four pathway nodes move up or down in each modality. |
| Structural kinase cases | `SiK_*` | Optional | Emitted only when `ENABLE_KINASE_AUGMENTATION=1`. Not a baseline dependency. |
| Structural kinase EI / score | `SiK_*_EI_*`, `SiK_score_*` | Optional | Emitted only when `ENABLE_KINASE_AUGMENTATION=1`. |
| Final multimodal score | `multimodel_score`, `PDS` | Emitted per contrast | Baseline `PDS_*` equals direct multimodal support without kinase augmentation. Optional kinase augmentation can add SiK support. |
| Kinase support | Not native in `sample.xlsx` as external score | Optional custom layer | Downstream `kinase_support_score_{contrast}` remains an augmentation/analysis layer, not baseline Incytr evidence. |

## Current factorial invocation

The production runner is:

```bash
bash alz/integration/run_factorial_all_pairs.sh
```

That runner executes by default:

1. `export_expression_factorial.py`
2. `export_multiomics_evidence_factorial.py`
3. `run_incytr_factorial_all_pairs.R`

Kinase augmentation is opt-in:

```bash
ENABLE_KINASE_AUGMENTATION=1 bash alz/integration/run_factorial_all_pairs.sh
```

The R wrapper writes wide receiver Parquet at:

```text
factorial/all_pairs/recv_{receiver}.parquet
```

with:

```text
sender, receiver, Sender.group, Receiver.group,
Ligand, Receptor, EM, Target, Path,
pathway_evidence, imputed_nodes, n_animals, df_resid,
TPDS_{contrast}, SE_{contrast}, pvalue_{contrast},
*_sclog2FC_{contrast}, *_sc_aFC_{contrast},
*_pr_log2FC_{contrast}, *_pr_aFC_{contrast},
*_ps_log2FC_{contrast}, *_ps_aFC_{contrast},
*_py_log2FC_{contrast}, *_py_aFC_{contrast},
sc_up/down_{contrast}, pr_up/down_{contrast},
ps_up/down_{contrast}, py_up/down_{contrast},
PPDS_{contrast}, PhPDS_ps_{contrast}, PhPDS_py_{contrast},
multimodel_score_{contrast}, PDS_{contrast}
```

It also writes a contrast-long, sample-style export at:

```text
factorial/all_pairs_long/recv_{receiver}_long.parquet
```

where each row is one pathway x one contrast. In that table the supported
sample-style columns are unsuffixed:

```text
contrast, sender, receiver, Sender.group, Receiver.group,
Ligand, Receptor, EM, Target, Path,
Ligand_sclog2FC, Receptor_sclog2FC, EM_sclog2FC, Target_sclog2FC,
Ligand_pr_log2FC, Receptor_pr_log2FC, EM_pr_log2FC, Target_pr_log2FC,
Ligand_ps_log2FC, Receptor_ps_log2FC, EM_ps_log2FC, Target_ps_log2FC,
Ligand_py_log2FC, Receptor_py_log2FC, EM_py_log2FC, Target_py_log2FC,
sc_up, sc_down, pr_up, pr_down, ps_up, ps_down, py_up, py_down,
TPDS, PPDS, PhPDS_ps, PhPDS_py, multimodel_score, PDS
```

The downstream custom kinase-support step is optional. When run, it writes
per-pair CSVs with:

```text
Path, EM, Target, Receptor, Ligand,
kinase_support_score_{contrast},
kinase_support_score_sum_{contrast},
n_distinct_kinases_{contrast},
concordance_flag_{contrast},
TPDS_{contrast}
```

## Remaining gaps

1. **Cell-type-resolved proteome/PTM support is still limited by input data.**
   Baseline Incytr no longer depends on SEA-AD/WMB mapping. If exact omics
   columns do not match the factorial cell-type vocabulary, the adapter uses a
   condition-level global mean and records that assignment. The cleanest future
   input is proteome/PTM evidence already expressed in the same cell-type
   vocabulary as transcriptomics.

2. **Node evidence labels are not yet emitted.**
   The numeric evidence surface is present, but categorical node labels such as
   DEG/protein/PTM-supported remain to be added if users need sample.xlsx-style
   labels.

3. **Optional kinase augmentation remains custom.**
   It should stay clearly separated from baseline Incytr outputs so users do
   not confuse external kinase enrichment support with direct PTM node support.

## Remediation options

### Minimum viable surfacing

Keep the factorial TPDS model and custom kinase-support model, but add an
evidence-audit export keyed by `(sender, receiver, Path, contrast)` with:

- per-node transcriptomic modeled effect or condition summaries;
- per-node total-protein effect where available;
- per-node S/T phospho effect where available;
- per-node pY effect where available;
- native-like direction counts (`sc_up/down`, `pr_up/down`, `ps_up/down`,
  `py_up/down`);
- explicit provenance flags distinguishing native Incytr-equivalent evidence
  from custom kinase-support evidence.

This would address the user-facing transparency issue without changing the
statistical path.

### Full native-evidence parity

Implement factorial-aware analogs of the native Incytr multiomics columns:

- factorial proteome adapter for `*_pr_*` and `PPDS`;
- factorial phospho adapters for S/T and pY `*_ps_*`, `*_py_*`,
  `PhPDS_ps`, and `PhPDS_py`;
- factorial SiK/EI scoring for `SiK_*`, `SiK_*_EI_*`, and `SiK_score_*`;
- a documented combined score that either reproduces native `PDS` semantics
  per contrast or deliberately keeps `TPDS` and all omics support separate.

This is the right direction if the expectation is “Incytr output should expose
all supporting modalities per pathway.”

### Recommended path

Do both in stages:

1. Add the minimum evidence-audit export first, because it is mostly a
   reporting/surfacing task and immediately improves interpretability.
2. Add pY and proteome into the integration adapters, because those are real
   evidence omissions relative to the available data.
3. Decide explicitly whether to compute native-style `PDS` in factorial mode or
   keep separate columns. If kept separate, rename and document the final
   ranking fields so users do not mistake custom kinase support for native
   `PhPDS` or `PDS`.
