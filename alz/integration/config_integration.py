"""Integration-specific configuration.

Holds the constants the active integration helpers read:
  - input/output paths
  - `load_cluster_spine()` (consumed by snrna_proportions, verify_decomposition,
    and the viewer)

Genotype/contrast coding lives in `alz.config` (`SAP_FACTORIAL`,
`CONTRAST_COEFS`). The factorial Incytr path was archived 2026-05-18; see
`archive/incytr_factorial_2026-05-18/` if you need the retired wrapper.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys

sys.path.insert(0, os.path.join(REPO_ROOT, "alz"))
from alz.shared import config as main_config  # noqa: E402

# Factorial design constants (FACTORIAL_CONTRASTS, DESIGN_COLUMNS,
# MUTANT_TO_DESIGN, TIMEPOINT_TO_DESIGN, FACTORIAL_GENOTYPES, …) were used
# only by the archived factorial Incytr driver (deleted 2026-05-18). The
# pair-mode integration consumes the spine + paths below; for canonical
# genotype/contrast coding see `alz.config.SAP_FACTORIAL` and
# `alz.config.CONTRAST_COEFS`.

# ---------------------------------------------------------------------------
# Cluster spine — Levy-t5 (31 clusters, min_cells=5, no rank gate)
# ---------------------------------------------------------------------------
# Transcript-side label vocabulary for Incytr senders/receivers. Built once by
# alz/integration/build_cluster_spine.py. Single source of truth:
# data/incytr_frozen/v2_46clusters/spines/levy_t5/cluster_spine.csv
# (in_spine == True). Spine *name* lives in main_config.CLUSTER_SPINE_NAME;
# the list of 31 cluster names lives in main_config.CLUSTER_SPINE.
CLUSTER_SPINE_FILE = os.path.join(
    REPO_ROOT, "data", "incytr_frozen", "v2_46clusters", "spines",
    main_config.CLUSTER_SPINE_NAME, "cluster_spine.csv",
)
BARCODE_TO_CLUSTER_FILE = os.path.join(
    REPO_ROOT, "data", "incytr_frozen", "v2_46clusters", "barcode_to_cluster.csv"
)

# Per-cluster decomposition outputs (Stage 6).
DECOMPOSITION_DIR = os.path.join(
    REPO_ROOT, "outputs", "reports", "decomposition", main_config.CLUSTER_SPINE_NAME
)
PHOSPHO_PER_CLUSTER_FILE = os.path.join(DECOMPOSITION_DIR, "phospho_per_cluster.parquet")
PROTEIN_PER_CLUSTER_FILE = os.path.join(DECOMPOSITION_DIR, "protein_per_cluster.parquet")


def resolve_cluster_spine_file(name: str = main_config.CLUSTER_SPINE_NAME) -> str:
    """Resolve cluster_spine.csv for spine `name` under spines/<name>/."""
    return os.path.join(
        REPO_ROOT, "data", "incytr_frozen", "v2_46clusters", "spines", name,
        "cluster_spine.csv",
    )


def load_cluster_spine(name: str = main_config.CLUSTER_SPINE_NAME) -> list[str]:
    """Return the ordered list of in-spine cluster names for spine `name`."""
    import pandas as pd

    df = pd.read_csv(resolve_cluster_spine_file(name))
    return df.loc[df["in_spine"] == True, "cluster_name"].tolist()  # noqa: E712


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
H5AD_PATH = main_config.SONG_H5AD_FILE
