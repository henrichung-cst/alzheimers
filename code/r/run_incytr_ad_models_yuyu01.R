
# AD Models Analysis Script (yuyu01)
# Adapts the incytr_commandline.R logic to run a specific comparison
# Comparison: ma_6mo_WTyp vs ma_6mo_AppP
# Sender -> Receiver: Astrocytes -> Microglia

library(Seurat)
library(dplyr)

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  stop("Unable to determine script path from commandArgs().")
}

script_path <- get_script_path()
repo_root <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
package_dir <- normalizePath(file.path(repo_root, "..", "incytr"), mustWork = TRUE)

# Load Incytr from local source
if (requireNamespace("devtools", quietly = TRUE)) {
  tryCatch({
    devtools::load_all(package_dir)
  }, error = function(e) {
    message("Failed to load_all: ", e$message)
    message("Attempting manual source...")
    files <- list.files(file.path(package_dir, "R"), pattern = "\\.R$", full.names = TRUE)
    sapply(files, source)
  })
} else {
  stop("devtools is required to load the package from source.")
}
# homologene is used for human2mouse mapping. 
# If not installed, kinase integration will be skipped.
if (requireNamespace("homologene", quietly = TRUE)) {
  library(homologene)
}

# 1. Setup Parameters
condition1 <- "ma_6mo_WTyp"
condition2 <- "ma_6mo_AppP"
Sender.group <- "Astrocytes"
Receiver.group <- "Microglia"

message(paste("Analyzing:", Sender.group, "->", Receiver.group))
message(paste("Conditions:", condition1, "vs", condition2))

# 2. Load Databases
# Ideally these are in the package, but the script loaded them manually.
# We'll try to rely on package data first.
if (!exists("DB_Layer1_mouse_filtered")) {
  data("DB_Layer1_mouse_filtered")
  data("DB_Layer2_mouse_filtered")
  data("DB_Layer3_mouse_filtered")
}

DB.M <- list(
  DB_Layer1_mouse_filtered,
  DB_Layer2_mouse_filtered,
  DB_Layer3_mouse_filtered
)

# 3. Load Data Inputs
base_dir <- "data/incytr_collections/song"

# Seurat Object
seurat_path <- file.path(base_dir, "transcriptomics", "incytr_obj.rds")
if (!file.exists(seurat_path)) stop("Seurat object not found: ", seurat_path)
message("Loading Seurat object...")
Data.input <- readRDS(seurat_path)

# Prepare Metadata
Data.input@meta.data$Type <- Data.input@active.ident
Data.input@meta.data$condition <- as.factor(Data.input@meta.data$Group)

# Proteomics & Phospho Inputs
message("Loading omics data...")
pr <- read.csv(file.path(base_dir, "proteomics", "pr_yuyu_deconvoluted.csv"), check.names = FALSE)
ps <- read.csv(file.path(base_dir, "proteomics", "ps_yuyu_deconvoluted.csv"), check.names = FALSE)
py <- read.csv(file.path(base_dir, "proteomics", "py_yuyu_deconvoluted.csv"), check.names = FALSE)
kldata <- read.csv(file.path(base_dir, "kinase", "kldata.csv"))
input_gene_list <- read.csv(file.path(base_dir, "markers", "input_gene_list.csv"))

# 4. Process Omics Data (Logic from incytr_commandline.R)
process_omics <- function(df, cond, suffix) {
  # Select columns for condition
  df_sub <- select(df, contains(cond))
  # Rename columns to remove prefix/suffix garbage, keep cell type + suffix
  # Expected col name format in CSV: "{cond}_{celltype}" e.g. "ma_6mo_WTyp_Astrocytes"
  # We want "Astrocytes_pr"
  
  # The original script logic:
  # colnames(pr_1) <- paste0(sub(paste0(condition1,".*_"), "", colnames(pr_1)), "_pr")
  # This regex seems to assume something like "condition1_stuff_CellType".
  # Let's verify our column names: "ma_6mo_WTyp_Astrocytes"
  # condition1 = "ma_6mo_WTyp"
  # sub(paste0("ma_6mo_WTyp", ".*_"), "", "ma_6mo_WTyp_Astrocytes") -> "Astrocytes" (if .* matches nothing, but _ matches _)
  # Actually, ".*_" is greedy.
  # If col is "ma_6mo_WTyp_Astrocytes", paste0(cond, ".*_") matches "ma_6mo_WTyp_Astrocytes" ?? No.
  # Wait, if pattern is "ma_6mo_WTyp.*_", it matches "ma_6mo_WTyp_"
  # sub("ma_6mo_WTyp.*_", "", "ma_6mo_WTyp_Astrocytes") -> "Astrocytes"
  # Yes, looks correct.
  
  new_names <- paste0(sub(paste0(cond, ".*_"), "", colnames(df_sub)), suffix)
  colnames(df_sub) <- new_names
  
  # Add gene symbol
  # The CSV has "Gene Symbol" or "gene_symbol"
  if ("Gene Symbol" %in% colnames(df)) {
    df_sub$gene_symbol <- df$`Gene Symbol`
  } else if ("gene_symbol" %in% colnames(df)) {
    df_sub$gene_symbol <- df$gene_symbol
  } else {
    stop("Gene symbol column not found")
  }
  
  # Aggregate duplicate genes
  df_sub <- df_sub %>%
    group_by(gene_symbol) %>%
    summarise_all(mean, na.rm = TRUE)
    
  return(df_sub)
}

pr_1 <- process_omics(pr, condition1, "_pr")
pr_2 <- process_omics(pr, condition2, "_pr")

ps_1 <- process_omics(ps, condition1, "_ps")
ps_2 <- process_omics(ps, condition2, "_ps")

py_1 <- process_omics(py, condition1, "_py")
py_2 <- process_omics(py, condition2, "_py")

# Process Kinase Data
# Map human to mouse using homologene
message("Processing kinase data...")
# Verify homologene is working or mock it if needed
# If homologene fails (network issue?), we might need a fallback or skip.
# Assuming it works or user has data.
tryCatch({
  kldata_mouse <- human2mouse(kldata$GENE_NAME)
  kldata <- merge(kldata[, c(3, 5, 13)], kldata_mouse, by.x = "GENE_NAME", by.y = "humanGene")
  kldata <- kldata[, c(2, 3, 4)]
  names(kldata) <- c("gene", "site_pos", "motif.geneName")
}, error = function(e) {
  warning("Homologene mapping failed: ", e$message)
  # If failed, we can't use kinase data effectively or just use raw names if they match
})

# 5. Define Genes of Interest
DG.Sender <- unique(input_gene_list$gene[input_gene_list$cluster == Sender.group])
DG.Receiver <- unique(input_gene_list$gene[input_gene_list$cluster == Receiver.group])

# Add genes from proteomics fold change (Simplified logic from commandline.R)
fc_df <- data.frame(
  gene_symbol = pr_1$gene_symbol,
  Sender.1 = pr_1[[paste0(Sender.group, "_pr")]],
  Sender.2 = pr_2[[paste0(Sender.group, "_pr")]],
  Receiver.1 = pr_1[[paste0(Receiver.group, "_pr")]],
  Receiver.2 = pr_2[[paste0(Receiver.group, "_pr")]]
)

fc_df$sender_fc <- fc_df$Sender.2 / fc_df$Sender.1
fc_df$receiver_fc <- fc_df$Receiver.2 / fc_df$Receiver.1
fc_df$sender_fc2 <- ifelse(fc_df$sender_fc > 1, fc_df$sender_fc, 1/fc_df$sender_fc)
fc_df$receiver_fc2 <- ifelse(fc_df$receiver_fc > 1, fc_df$receiver_fc, 1/fc_df$receiver_fc)

fc_df[is.na(fc_df)] <- 0
fc_df[sapply(fc_df, is.infinite)] <- 0

DG.Sender_total <- top_n(fc_df, 500, sender_fc2)$gene_symbol
DG.Receiver_total <- top_n(fc_df, 500, receiver_fc2)$gene_symbol

DG.Sender <- unique(c(DG.Sender, intersect(DG.Sender_total, rownames(Data.input))))
DG.Receiver <- unique(c(DG.Receiver, intersect(DG.Receiver_total, rownames(Data.input))))

DG.Sender <- intersect(DG.Sender, rownames(Data.input))
DG.Receiver <- intersect(DG.Receiver, rownames(Data.input))

# 6. Run Incytr Pipeline

# Create Object
Xobject <- create_Incytr(
  object = Data.input@assays$originalexp@data, # Use originalexp as per commandline script
  meta = Data.input@meta.data,
  sender = Sender.group,
  receiver = Receiver.group,
  group.by = "Type",
  condition = c(condition1, condition2),
  assay = NULL,
  do.sparse = TRUE
)

# Pathway Inference
Xobject <- pathway_inference(
  Xobject,
  DB = DB.M,
  gene.use_Sender = DG.Sender,
  gene.use_Receiver = DG.Receiver
)

# Expression
Xobject <- Expr_bygroup(Xobject, mean_method = "mean")

# SigProb
Xobject <- Cal_SigProb(Xobject, K = 0.5, N = 2, cutoff_SigProb = 0.0, correction = 0.001)

# Multi-omics
Xobject <- Integr_multiomics(
  Xobject, 
  pr.data_condition1 = pr_1,
  pr.data_condition2 = pr_2,
  pr.correction = 0.001,
  ps.data_condition1 = ps_1,
  ps.data_condition2 = ps_2,
  ps.correction = 0.001,
  py.data_condition1 = py_1,
  py.data_condition2 = py_2,
  py.correction = 0.001
)

# Evaluation
Xobject <- Pathway_evaluation(Xobject, k_logi = 2)

# Kinase Integration
if (exists("kldata") && ncol(kldata) >= 3) {
  Xobject <- Integr_kinasedata(
    Xobject,
    kldata = kldata,
    mean_method = "mean", 
    cell_group = levels(Xobject@meta$Type),
    fold_threshold = 10
  )
} else {
  warning("Skipping kinase integration due to missing kldata")
}

# PDS
Xobject <- Cal_PDS(Xobject, KPDS.weight = 0.5, cutoff_PDS = 0.0)

# Permutation Test (Fast run)
message("Running permutation test (nboot=5)...")
Xobject <- Permutation_test(
  Xobject,
  nboot = 5,
  seed.use = 1L,
  mean_method = "mean"
)

# Export
output <- Export_results(Xobject, indicator = TRUE)
out_file <- paste0("examples/AD_yuyu01_", condition1, "_vs_", condition2, "_output.csv")
write.csv(output, out_file, row.names = FALSE)
message("Analysis complete. Results saved to ", out_file)
