# Single registry for INCYTR_* runtime knobs.
#
# All optional behavior in the integration pipeline is gated by env vars under
# the `INCYTR_*` namespace. Defaults declared here are the contract; the
# shell runner (`code/integration/incytr_runtime.sh`) mirrors them so that
# adapter-skip decisions made before R starts agree with the R-side reads.
#
# See docs/integrations/incytr_layer_inventory.md for the rationale and
# revival pointers for each knob.

incytr_runtime <- function() {
  list(
    # Kinase pack (off): bundles ALZ-19 kinase-imputed gene expansion + ALZ-20
    # kinase support score sidecar. Off by default — enabling reverts the
    # production pipeline to a Section-C-augmented configuration.
    layer_kinase_pack    = identical(Sys.getenv("INCYTR_LAYER_KINASE_PACK", "0"), "1"),

    # Backbone permutations (off): within-receiver shuffle null distributions.
    # Off pending the backbone-vs-pathway design revisit.
    layer_backbone_perms = identical(Sys.getenv("INCYTR_LAYER_BACKBONE_PERMS", "0"), "1"),

    # DuckDB pre-prune SigProb cutoff. Default 0.0 = native-equivalent
    # (no pre-prune). Production runs may raise this if benchmarks justify;
    # 0.01 is mathematically lossless for any reasonable top-K cut.
    cutoff_sigprob       = as.numeric(Sys.getenv("INCYTR_CUTOFF_SIGPROB", "0.0"))
  )
}
