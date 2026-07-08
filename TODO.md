need to refactor the the tcell analysis. the cell state labeling might be incorrect or low confidence

First we need to find the umap cluster data for the tcell data (donors 1 and 2)
then we need to plot the umap colored by the PROJECTILs assignment
visually inspect this graph
the design of this graphs should address how differentiatable PROJECTILs cell state assignments are from each other
cell states that are positioned closer to each other in the embedding space may lack differentiation and should be combined into a later cluster.

Second, the previous analysis brings into question the suitability for the PROJECTILs analysis for our dataset.
We plan for an alterative (and eventually overwrite the PROJECTILs) cell type assignment
The method is as follows.

Separate the data by umap clusters
For each cluster, find the top 30 or 50 mode expressed genes
We then have a matrix of cluster by gene and expression
Ask an AI model to interpret the T cell state assignment (CD8, CD4, TREG, substate) based on this list of marker genes, ask for confidence
^ the above assignment is by far the weakest point of analysis, introducing a brittle and potentially hallucinating AI stype into the analysis
if possible, we would want to find a static reference for calculation T cell states from marker genes

We then examine the cell type assignment from the above marker gene data and then we regenerate the incytr pathway analysis using this new cell type mapping


Separately we also want to calculate the correlation between proteomics and transcriptomics data by gene. Something like for each gene,
calculate the correlation between that genes proteomics and that genes transcriptomics data in tcells
