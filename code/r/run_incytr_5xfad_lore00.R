
# 5xFAD Analysis Script (Lore00)
# Adapts the vignette workflow to use the staged data in data/gdrive_shared

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
    # Fallback to manual sourcing if load_all fails (e.g. strict dependency checks)
    message("Attempting manual source...")
    files <- list.files(file.path(package_dir, "R"), pattern = "\\.R$", full.names = TRUE)
    # Order matters: Class/Utils first usually better, but sourcing all might work for simple cases
    # We specifically ensure Incytr_class.R is sourced early if possible, but list.files is alphabetical
    # Incytr_class.R comes after evaluation.R alphabetically? No.
    # I, M, U.
    # Let's just source them.
    sapply(files, source)
  })
} else {
  stop("devtools is required to load the package from source.")
}
library(Seurat)
library(dplyr)

# 1. Load Data
message("Loading data...")

# Seurat Object
seurat_path <- "data/gdrive_shared/lore00/transcriptomics/reclustering/named_lore00.RDS"
if (!file.exists(seurat_path)) stop("Seurat object not found: ", seurat_path)
Data.input <- readRDS(seurat_path)

# Marker Genes
markers_path <- "examples/5xad_data/Allmarkers_18groups.csv"
Allmarkers <- read.csv(markers_path)

# Proteomics & Phosphorylation
# Note: These are already processed and split by condition in the example data
pr_5X <- read.csv("examples/5xad_data/processed_pr_5X_v2.csv")
pr_WT <- read.csv("examples/5xad_data/processed_pr_WT_v2.csv")
ps_5X <- read.csv("examples/5xad_data/processed_ps_5X_v2.csv")
ps_WT <- read.csv("examples/5xad_data/processed_ps_WT_v2.csv")
py_5X <- read.csv("examples/5xad_data/processed_py_5X_v2.csv")
py_WT <- read.csv("examples/5xad_data/processed_py_WT_v2.csv")

# Kinase Library
kldata <- read.csv("examples/5xad_data/kldata_pspy.csv")

# Load Databases (Bundled with package, but loading explicit files if needed)
# The package should load these automatically or we load from data/
# The vignette loads them manually. Let's try to load them from the package data directory if possible,
# or assume they are available in the working session after library(Incytr).
# However, the vignette loads .rda files. Let's check where they are.
# They are in data/*.rda. library(Incytr) should make them available.

# 2. Setup Analysis Parameters
Sender.group <- "Astrocytes"
Receiver.group <- "Microglia"
Conditions <- c("5X", "WT")

message(paste("Analyzing:", Sender.group, "->", Receiver.group))

# Define input genes (sender and receiver) from markers
# The marker table has a 'cluster' column like "Astrocytes_5X"
DG.Sender <- Allmarkers$gene[Allmarkers$cluster %in% c(paste0(Sender.group, "_5X"), paste0(Sender.group, "_WT"))]
DG.Receiver <- Allmarkers$gene[Allmarkers$cluster %in% c(paste0(Receiver.group, "_5X"), paste0(Receiver.group, "_WT"))]

# 3. Create Incytr Object
# Need to check if 'Type' is the correct group.by column in the lore00 object
# The vignette uses 'Type'. Let's verify if the object has it.
if (!"Type" %in% colnames(Data.input@meta.data)) {
  # If Type is missing, try to use active.ident or another column
  message("'Type' column not found in metadata. Using active.ident.")
  Data.input@meta.data$Type <- Idents(Data.input)
}

IncytrObj <- create_Incytr(
  object = Data.input@assays$RNA@data,
  meta = Data.input@meta.data,
  sender = Sender.group,
  receiver = Receiver.group,
  group.by = "Type",
  condition = Conditions,
  assay = NULL,
  do.sparse = TRUE
)

# 4. Pathway Inference
# We need to construct the DB list as expected by pathway_inference
# Assuming the package data is loaded:
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

IncytrObj <- pathway_inference(
  IncytrObj,
  DB = DB.M,
  gene.use_Sender = DG.Sender,
  gene.use_Receiver = DG.Receiver
)

# 5. Expression by Group
IncytrObj <- Expr_bygroup(IncytrObj, mean_method = "mean") # Using mean as per recommendation in README/Commandline

# 6. Signaling Probability
IncytrObj <- Cal_SigProb(IncytrObj, K = 0.5, N = 2, cutoff_SigProb = 0.0, correction = 0.001)

# 7. Single Cell Fold Change (Optional)
IncytrObj <- Cal_scFC(IncytrObj)

# 8. Multi-omics Integration
IncytrObj <- Integr_multiomics(
  IncytrObj,
  pr.data_condition1 = pr_5X,
  pr.data_condition2 = pr_WT,
  pr.correction = 0.001,
  ps.data_condition1 = ps_5X,
  ps.data_condition2 = ps_WT,
  ps.correction = 0.001,
  py.data_condition1 = py_5X,
  py.data_condition2 = py_WT,
  py.correction = 0.001
)

# 9. Pathway Evaluation
IncytrObj <- Pathway_evaluation(IncytrObj, k_logi = 2)

# 10. Kinase Integration
# kldata in example is already suitable? Vignette uses it directly.
IncytrObj <- Integr_kinasedata(
  IncytrObj,
  kldata = kldata,
  mean_method = "mean",
  cell_group = levels(IncytrObj@meta$Type),
  fold_threshold = 10
)

# 11. Final Score (PDS)
IncytrObj <- Cal_PDS(IncytrObj, KPDS.weight = 0.5, cutoff_PDS = 0.0)

# 12. Permutation Test
# Using low nboot for testing purposes
message("Running permutation test (nboot=5)...")
IncytrObj <- Permutation_test(
  IncytrObj,
  nboot = 5,
  seed.use = 1L,
  mean_method = "mean",
  cutoff_p_value = 1.0 # Keep all for now
)

# 13. Export
output <- Export_results(IncytrObj, indicator = TRUE)

# Write result
out_file <- "examples/5xFAD_lore00_output.csv"
write.csv(output, out_file, row.names = FALSE)
message("Analysis complete. Results saved to ", out_file)
