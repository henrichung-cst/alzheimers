# Kinase upstream/downstream regulation network

Using the PhosphoSite Kinase Library (check if access is already available in-repo; fetch if not),
build a kinase regulation network. Two layers:
1. Reference hierarchy: "if kinase A is upregulated, it should cause downregulation of downstream kinase B."
2. Observed overlay: what actually happens in the disease phenotype (both A and B upregulated? concordant? discordant?).

Viewer goal: click a kinase → show its reference regulation neighbors → overlay observed
disease-direction arrows. Filter to kinases co-expressed in the same cell type.

Precedes [`kinase-family-discrimination.md`](kinase-family-discrimination.md).
