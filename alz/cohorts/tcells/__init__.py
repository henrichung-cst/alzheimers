"""Shared T-cell cohort vocabulary."""


def state_lineage(state: str) -> str:
    """Return the CD4/CD8 lineage encoded by an evidence-backed state name."""
    value = str(state)
    if value.startswith("CD4"):
        return "CD4"
    if value.startswith("CD8"):
        return "CD8"
    raise ValueError(f"T-cell state has no CD4/CD8 lineage: {value!r}")
