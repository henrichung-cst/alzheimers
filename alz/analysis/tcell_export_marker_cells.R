#!/usr/bin/env Rscript
# Extract the compact per-cell evidence needed for T-cell labeling from the
# multi-GB Seurat object in one load:
#   * log-normalized RNA expression for the declared marker genes;
#   * continuous Seurat S and G2/M scores used only for nuisance regression;
#   * raw CITE-seq antibody UMI counts for lineage/state context and isotypes;
#   * RNA/Protein QC metadata used to audit time-dependent detection depth.
# Per-cell labels are assigned downstream by tcell_state_labels.py.
#
# Run under a memory cap (the object is ~5 GB compressed, ~15-30 GB in RAM):
#   systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G \
#     pixi run Rscript alz/analysis/tcell_export_marker_cells.R <donor>
#
# Args: <donor> (donor1|donor2). Reads the marker list written by
#   `pixi run python alz/analysis/tcell_percell_auroc.py --write-markers <path>`.

suppressPackageStartupMessages({library(Seurat); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
donor <- if (length(args) >= 1) args[[1]] else stop("usage: Rscript ... <donor>")

rds <- list(
  donor1 = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
  donor2 = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds"
)[[donor]]
marker_file <- "outputs/reports/tcell_labeling/auroc/marker_genes.txt"
outdir <- "outputs/reports/tcell_labeling/auroc"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

markers <- readLines(marker_file)
markers <- markers[nzchar(markers)]

message("[", donor, "] loading ", rds)
obj <- readRDS(rds)
if (!inherits(obj, "Seurat")) stop("not a Seurat object: ", class(obj))

assay <- "RNA"
if (!assay %in% Assays(obj)) stop("RNA assay is required for exhaustion-marker evidence")
DefaultAssay(obj) <- assay
message("[", donor, "] assay=", assay, " cells=", ncol(obj), " genes=", nrow(obj))

# Export full-precision continuous cycle covariates with the compact matrix.  The
# historical re-clustering artifact rounded these values to three decimals; a
# clean report rebuild should not depend on that lossy intermediate.  Phase is
# retained for QC only and is never used by the regression or AUROC targets.
cycle_columns <- c("S.Score", "G2M.Score", "Phase")
if (!all(cycle_columns %in% colnames(obj@meta.data))) {
  s_genes <- intersect(cc.genes.updated.2019$s.genes, rownames(obj))
  g2m_genes <- intersect(cc.genes.updated.2019$g2m.genes, rownames(obj))
  if (!length(s_genes) || !length(g2m_genes)) {
    stop("RNA assay lacks genes required for Seurat cell-cycle scoring")
  }
  obj <- CellCycleScoring(
    obj,
    s.features = s_genes,
    g2m.features = g2m_genes,
    set.ident = FALSE
  )
}

# log-normalized 'data' slot — matches the log-space choice in the pseudobulk path.
mat <- GetAssayData(obj, assay = assay, layer = "data")
present <- intersect(markers, rownames(mat))
missing <- setdiff(markers, rownames(mat))
if (length(missing)) message("[", donor, "] markers absent: ", paste(missing, collapse = ", "))

sub <- as.matrix(t(mat[present, , drop = FALSE]))  # cells x markers, dense & tiny
df <- data.frame(barcode = colnames(obj), sub, check.names = FALSE)
df[["Phase"]] <- as.character(obj@meta.data[colnames(obj), "Phase"])
df[["S.Score"]] <- as.numeric(obj@meta.data[colnames(obj), "S.Score"])
df[["G2M.Score"]] <- as.numeric(obj@meta.data[colnames(obj), "G2M.Score"])

out <- file.path(outdir, paste0(donor, "_marker_cell_expr.csv"))
write.csv(df, out, row.names = FALSE)
message("[", donor, "] wrote ", out, " (", nrow(df), " cells x ", length(present),
        " markers plus continuous cell-cycle covariates)")

# Raw ADT evidence. Feature aliases accommodate the donor-specific Ki-67 name;
# checkpoint proteins are not present in either panel. Donor2-only TOX/BATF/
# PRDM1/GZMB are exported when measured and remain optional downstream evidence.
if (!"Protein" %in% Assays(obj)) stop("Protein assay is required")
protein_counts <- GetAssayData(obj, assay = "Protein", layer = "counts")
feature_aliases <- list(
  CD3_protein_umi = c("CD3(UCHT1)-Ab"),
  CD4_protein_umi = c("CD4(RPA-T4)-Ab"),
  CD8_protein_umi = c("CD8(SK1)-Ab"),
  TCF1_protein_umi = c("TCF1-TCF7-Ab"),
  Ki67_protein_umi = c("Ki67-Ab", "Ki-67-Ab"),
  NCAM1_protein_umi = c("NCAM1-Ab"),
  mouse_isotype_umi = c("Mouse-Isotype-Control-Ab"),
  rabbit_isotype_umi = c("Rabbit-Isotype-Control-Ab")
)
optional_features <- c(
  TOX_protein_umi = "Tox-Tox2-Ab",
  BATF_protein_umi = "BATF-Ab",
  PRDM1_protein_umi = "Blimp-1-PRDI-BF1-Ab",
  GZMB_protein_umi = "Granzyme-B-Ab"
)
resolve_feature <- function(aliases) {
  present_aliases <- aliases[aliases %in% rownames(protein_counts)]
  if (!length(present_aliases)) return(NA_character_)
  present_aliases[[1L]]
}
features <- vapply(feature_aliases, resolve_feature, character(1L))
missing_features <- names(features)[is.na(features)]
if (length(missing_features)) {
  stop("required Protein evidence absent: ", paste(missing_features, collapse = ", "))
}

adt <- data.frame(barcode = colnames(obj), check.names = FALSE)
for (output_name in names(features)) {
  adt[[output_name]] <- as.numeric(protein_counts[features[[output_name]], colnames(obj)])
}
for (output_name in names(optional_features)) {
  feature <- optional_features[[output_name]]
  if (feature %in% rownames(protein_counts)) {
    adt[[output_name]] <- as.numeric(protein_counts[feature, colnames(obj)])
  }
}

metadata <- obj@meta.data
for (column in c("nCount_RNA", "nFeature_RNA", "nCount_Protein", "nFeature_Protein")) {
  if (column %in% colnames(metadata)) adt[[column]] <- metadata[[column]]
}
mitochondrial_column <- grep(
  "^(percent\\.mt|percent_mito|mito_percent)$",
  colnames(metadata),
  ignore.case = TRUE,
  value = TRUE
)
if (length(mitochondrial_column)) {
  adt[["percent_mitochondrial"]] <- metadata[[mitochondrial_column[[1L]]]]
}

adt_outdir <- "outputs/reports/tcell_labeling/adt"
dir.create(adt_outdir, showWarnings = FALSE, recursive = TRUE)
adt_out <- file.path(adt_outdir, paste0(donor, "_adt_evidence.csv"))
write.csv(adt, adt_out, row.names = FALSE)
message("[", donor, "] wrote ", adt_out, " (", nrow(adt), " cells x ",
        ncol(adt) - 1L, " evidence columns)")
