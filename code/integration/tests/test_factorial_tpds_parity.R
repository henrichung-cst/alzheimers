#!/usr/bin/env Rscript
# Phase A parity check — verify the wrapper's fit_contrast_ols produces TPDS
# values numerically equal to incytr/R/factorial.R::Contrast_SigProb on a
# small synthetic SigProb matrix.
#
# This pins down the log + OLS + logi(k = 2/log(2)) chain. After Phase A the
# two paths must agree to within 1e-10.
#
# Run:
#   micromamba run -n incytr Rscript code/integration/tests/test_factorial_tpds_parity.R

incytr_dir <- Sys.getenv("INCYTR_DIR", unset = "../incytr")
if (!dir.exists(incytr_dir)) {
  stop("Set INCYTR_DIR or run from alzheimers root with ../incytr alongside.")
}
suppressPackageStartupMessages({
  pkgload::load_all(incytr_dir, quiet = TRUE)
})

set.seed(2026)

n_animals <- 8L
n_pathways <- 50L
animals <- paste0("a", seq_len(n_animals))

# Two-condition treatment design, matching how create_Incytr() builds the
# default factorial design.
cond <- factor(rep(c("WT", "Dis"), each = n_animals / 2L), levels = c("WT", "Dis"))
X <- model.matrix(~ cond)
rownames(X) <- animals

# Random SigProb in (0, 1) with a real treatment effect on a subset of pathways.
sp <- matrix(runif(n_animals * n_pathways, 0.01, 0.6),
             nrow = n_pathways, ncol = n_animals,
             dimnames = list(paste0("p", seq_len(n_pathways)), animals))
sp[1:20, cond == "Dis"] <- pmin(sp[1:20, cond == "Dis"] * 1.8, 0.99)

# --- Wrapper path (pasted from run_incytr_factorial_all_pairs.R::fit_contrast_ols) ---
hat_mat <- solve(crossprod(X)) %*% t(X)               # p x n
XtX_inv <- solve(crossprod(X))
contrast_vec <- c(0, 1)                                 # treatment contrast
contrast_names <- "treat"
contrast_mat <- matrix(contrast_vec, nrow = 1)

pseudocount <- 1e-10
Y_log <- log(sp + pseudocount)
beta_mat <- hat_mat %*% t(Y_log)                       # p x n_pw
beta_c_wrapper <- as.numeric(contrast_vec %*% beta_mat)
k_logi <- 2 / log(2)
tpds_wrapper <- logi(beta_c_wrapper, k = k_logi)

# --- Package path (Contrast_SigProb) ---
sp_df <- data.frame(Path = rownames(sp))
for (a in animals) sp_df[[paste0("SigProb_", a)]] <- sp[, a]

obj <- new("Incytr",
           pathways = data.frame(Path = rownames(sp)),
           sigprob.byanimal = sp_df,
           Evaluation = data.frame(),
           design = X,
           contrasts = list(treat = contrast_vec),
           options = list(mode = "factorial"))

obj <- Contrast_SigProb(obj, transform = "log",
                        k_logi = k_logi, pseudocount = pseudocount)
tpds_pkg <- obj@Evaluation[["TPDS_treat"]]

# --- Compare ---
tol <- 1e-10
delta <- max(abs(tpds_wrapper - tpds_pkg))
cat(sprintf("max |TPDS_wrapper - TPDS_package| = %.3e (tol %.0e)\n", delta, tol))
stopifnot(all(tpds_wrapper >= -1 - 1e-12 & tpds_wrapper <= 1 + 1e-12))
if (delta > tol) {
  stop(sprintf("Phase A parity FAILED: max delta %.3e exceeds tolerance %.0e", delta, tol))
}

cat("Phase A parity OK: wrapper TPDS matches package Contrast_SigProb within ",
    sprintf("%.0e", tol), "\n", sep = "")
