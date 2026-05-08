suppressPackageStartupMessages({
  library(Matrix)
})

sanitize_celltype <- function(x) {
  gsub("/", "-", gsub(" ", "_", x))
}

load_ad_factorial_inputs <- function(input_dir = "code/integration/intermediates/factorial") {
  mat <- Matrix::readMM(file.path(input_dir, "expression_matrix.mtx"))
  mat <- as(mat, "dgCMatrix")

  genes <- read.csv(file.path(input_dir, "expression_genes.csv"), stringsAsFactors = FALSE)$gene
  barcodes <- read.csv(file.path(input_dir, "expression_barcodes.csv"), stringsAsFactors = FALSE)$barcode
  rownames(mat) <- genes
  colnames(mat) <- barcodes

  meta <- read.csv(
    file.path(input_dir, "expression_metadata.csv"),
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  animal_meta <- read.csv(
    file.path(input_dir, "animal_metadata.csv"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (!"condition" %in% colnames(meta)) {
    required_condition_cols <- c("genotype", "timepoint")
    missing_condition_cols <- setdiff(required_condition_cols, colnames(meta))
    if (length(missing_condition_cols) > 0) {
      stop("expression_metadata.csv is missing required columns: ",
           paste(missing_condition_cols, collapse = ", "))
    }
    meta$condition <- paste(meta$genotype, meta$timepoint, sep = "_")
  }

  design_cols <- c(
    "const", "App", "Tau", "Int",
    "time_4mo", "time_6mo",
    "App_x_time4", "App_x_time6",
    "Tau_x_time4", "Tau_x_time6"
  )
  missing_cols <- setdiff(c("animal_id", design_cols), colnames(animal_meta))
  if (length(missing_cols) > 0) {
    stop("animal_metadata.csv is missing required columns: ",
         paste(missing_cols, collapse = ", "))
  }

  design <- as.matrix(animal_meta[, design_cols])
  rownames(design) <- animal_meta$animal_id

  cell_types <- sort(unique(as.character(meta$labels)))
  list(
    expr = mat,
    meta = meta,
    animal_meta = animal_meta,
    design = design,
    senders = cell_types,
    receivers = cell_types,
    ptm = NULL
  )
}

build_factorial_contrasts <- function(animal_meta = NULL) {
  contrasts <- list(
    App_2mo  = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    App_4mo  = c(0, 1, 0, 0, 0, 0, 1, 0, 0, 0),
    App_6mo  = c(0, 1, 0, 0, 0, 0, 0, 1, 0, 0),
    Tau_2mo  = c(0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    Tau_4mo  = c(0, 0, 1, 0, 0, 0, 0, 0, 1, 0),
    Tau_6mo  = c(0, 0, 1, 0, 0, 0, 0, 0, 0, 1),
    ApTt_2mo = c(0, 1, 1, 1, 0, 0, 0, 0, 0, 0),
    ApTt_4mo = c(0, 1, 1, 1, 0, 0, 1, 0, 1, 0),
    ApTt_6mo = c(0, 1, 1, 1, 0, 0, 0, 1, 0, 1)
  )
  contrasts
}

apply_pair_filter <- function(senders, receivers, pair_filter = Sys.getenv("PAIR_FILTER", "")) {
  if (identical(pair_filter, "")) {
    return(list(senders = senders, receivers = receivers))
  }

  parts <- strsplit(pair_filter, ":", fixed = TRUE)[[1]]
  if (length(parts) != 2) {
    stop("PAIR_FILTER must have form '<sender>:<receiver>', got: ", pair_filter)
  }

  sender_filter <- trimws(parts[[1]])
  receiver_filter <- trimws(parts[[2]])
  if (!identical(sender_filter, "*")) {
    senders <- intersect(senders, sender_filter)
  }
  if (!identical(receiver_filter, "*")) {
    receivers <- intersect(receivers, receiver_filter)
  }
  if (length(senders) == 0 || length(receivers) == 0) {
    stop("PAIR_FILTER selected no sender/receiver pairs: ", pair_filter)
  }

  list(senders = senders, receivers = receivers)
}
