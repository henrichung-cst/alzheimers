"""Integration-specific configuration for the thin Incytr factorial wrapper.

Holds only the constants that load.R and export_factorial_inputs.py read:
  - factorial filter values (sex, genotypes, timepoints)
  - design matrix columns + per-condition encodings
  - factorial contrast vectors
  - input/output paths

The legacy materialized-derivations layer (KINASE_IMPUTATION_*, LAMBDA_*,
N_PERMUTATIONS, resolve_universe_id, etc.) was retired with the wrapper
rewrite. Recover from git history if anything reaches for it.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys

sys.path.insert(0, os.path.join(REPO_ROOT, "alz"))
import config as main_config  # noqa: E402

# ---------------------------------------------------------------------------
# Factorial filter (males-only, all 4 genotypes x 3 timepoints)
# ---------------------------------------------------------------------------
FACTORIAL_GENOTYPES = ["WTyp", "AppP", "Ttau", "ApTt"]
FACTORIAL_TIMEPOINTS = ["2mo", "4mo", "6mo"]
FACTORIAL_SEX = "ma"

# ---------------------------------------------------------------------------
# Design matrix (matches kinase_enrich.py OLS)
# ---------------------------------------------------------------------------
DESIGN_COLUMNS = [
    "const", "App", "Tau", "Int",
    "time_4mo", "time_6mo",
    "App_x_time4", "App_x_time6",
    "Tau_x_time4", "Tau_x_time6",
]

MUTANT_TO_DESIGN = {
    "WTyp": {"App": 0, "Tau": 0, "Int": 0},
    "AppP": {"App": 1, "Tau": 0, "Int": 0},
    "Ttau": {"App": 0, "Tau": 1, "Int": 0},
    "ApTt": {"App": 1, "Tau": 1, "Int": 1},
}

TIMEPOINT_TO_DESIGN = {
    "2mo": {"time_4mo": 0, "time_6mo": 0},
    "4mo": {"time_4mo": 1, "time_6mo": 0},
    "6mo": {"time_4mo": 0, "time_6mo": 1},
}

# Contrast coefficient vectors, ordered to match DESIGN_COLUMNS.
FACTORIAL_CONTRASTS = {
    "App_2mo":  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "App_4mo":  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    "App_6mo":  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "Tau_2mo":  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "Tau_4mo":  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    "Tau_6mo":  [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    "ApTt_2mo": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "ApTt_4mo": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    "ApTt_6mo": [0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
}

# Explicit (cond_ref, cond_alt) for each contrast — replaces the package's
# heuristic that misroutes multi-coefficient interaction contrasts.
FACTORIAL_CONTRAST_CONDITIONS = {
    "App_2mo":  ("WTyp_2mo", "AppP_2mo"),
    "App_4mo":  ("WTyp_4mo", "AppP_4mo"),
    "App_6mo":  ("WTyp_6mo", "AppP_6mo"),
    "Tau_2mo":  ("WTyp_2mo", "Ttau_2mo"),
    "Tau_4mo":  ("WTyp_4mo", "Ttau_4mo"),
    "Tau_6mo":  ("WTyp_6mo", "Ttau_6mo"),
    "ApTt_2mo": ("WTyp_2mo", "ApTt_2mo"),
    "ApTt_4mo": ("WTyp_4mo", "ApTt_4mo"),
    "ApTt_6mo": ("WTyp_6mo", "ApTt_6mo"),
}
assert set(FACTORIAL_CONTRAST_CONDITIONS) == set(FACTORIAL_CONTRASTS)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
H5AD_PATH = main_config.SONG_H5AD_FILE

# Inputs to the R wrapper, written by export_factorial_inputs.py.
FACTORIAL_INPUT_DIR = os.path.join(REPO_ROOT, "data", "incytr_factorial_inputs")

# R wrapper output root (parquet + views.sql).
FACTORIAL_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "reports", "incytr_factorial")
