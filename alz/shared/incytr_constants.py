"""Incytr-parity constants shared across the AD application.

Single source of truth for the `Cal_foldchange` / `Cal_scFC` correction terms,
so the Python consumers (the round-trip verification harness and the normalized
substrate index metadata) never drift from the R driver. The authoritative
values live at the driver call sites in
``alz/incytr_pair/incytr_commandline.R``; mirror any change there here.

  - EPSILON_OMICS : protein/phospho `Cal_foldchange(correction = 0.001)`
                    (incytr_commandline.R:376-389).
  - EPSILON_SC    : transcript `Cal_scFC(correction = 0.01)`
                    (incytr_commandline.R:435) — sce4-parity override of the
                    analysis.R:248 default (1e-5).
"""

EPSILON_OMICS = 1e-3
EPSILON_SC = 0.01
