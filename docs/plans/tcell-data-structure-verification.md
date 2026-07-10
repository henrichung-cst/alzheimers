# T-cell data structure verification

Confirm the pipeline's understanding of the T-cell data matches:
- donor1 = IMAC data; donor2 = no IMAC data
- Both have total proteome + pY data
- Skip kinase MEA for donor2
- Different timepoints (not directly comparable, same experiment parameters)
- Both have scRNA; run Incytr on both (on deconvoluted); run kinase MEA on bulk
