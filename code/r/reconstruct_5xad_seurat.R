library(Seurat)
library(Matrix)
library(dplyr)
library(stringr)

# Define paths
data_dir <- "data/gdrive_shared/lore00/transcriptomics"
matrix_file <- file.path(data_dir, "analysis/240712_ex1_comb1/all-sample/DGE_filtered/count_matrix.mtx")
genes_file <- file.path(data_dir, "analysis/240712_ex1_comb1/all-sample/DGE_filtered/all_genes.csv")
cells_file <- file.path(data_dir, "analysis/240712_ex1_comb1/all-sample/DGE_filtered/cell_metadata.csv")
obs_file <- file.path(data_dir, "obs_df.csv")

message("Reading count matrix...")
# The MTX file header says Rows=cells, Cols=genes. 
# Seurat expects Genes x Cells, so we need to transpose.
counts <- readMM(matrix_file)
counts <- t(counts)

message("Reading gene and cell metadata...")
genes <- read.csv(genes_file)
cells_raw <- read.csv(cells_file)

# Set row and column names
# Ensure unique gene names (some genes might have same name but different ID)
rownames(counts) <- make.unique(genes$gene_name)
colnames(counts) <- cells_raw$bc_wells

message("Reading annotated observations...")
obs <- read.csv(obs_file, row.names = 1)

# The barcodes in obs seem to have a different suffix (-0 instead of __s1)
# Let's try to match them based on the core barcode (wind indices)
# Example: 27_01_68__s1 in cells_raw vs 27_01_68-0 in obs

# Extract core barcode for matching if direct match fails
cells_raw$core_bc <- str_extract(cells_raw$bc_wells, "^[0-9]+_[0-9]+_[0-9]+")
obs$core_bc <- str_extract(rownames(obs), "^[0-9]+_[0-9]+_[0-9]+")

# Check for duplicates in core barcodes within the same sample
# Since these are different samples pooled, the core_bc + sample should be unique
cells_raw$match_id <- paste0(cells_raw$core_bc, "_", cells_raw$sample)
obs$match_id <- paste0(obs$core_bc, "_", obs$sample)

message("Aligning metadata...")
# Join obs metadata to cells_raw
# We only want cells that are in the 'obs' (the filtered/annotated set)
cells_metadata <- cells_raw %>%
  inner_join(obs %>% select(match_id, Cluster_fine, Cluster_coarse, leiden), by = "match_id")

# Subset count matrix to annotated cells
counts_subset <- counts[, cells_metadata$bc_wells]

# Create Seurat object
message("Creating Seurat object...")
Data.input <- CreateSeuratObject(counts = counts_subset)

# Add metadata
Data.input$Type <- cells_metadata$Cluster_coarse
Data.input$Sample <- cells_metadata$sample
Data.input$Condition <- ifelse(grepl("5XFAD", cells_metadata$sample), "5X", "WT")

# Normalize data
message("Normalizing data...")
Data.input <- NormalizeData(Data.input)

# Save the object
message("Saving Seurat object to 5xad_data_Seurat.RDS...")
saveRDS(Data.input, "5xad_data_Seurat.RDS")

message("Reconstruction complete.")
