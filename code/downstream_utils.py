"""Shared utilities for downstream AD kinase enrichment analyses.

Provides data loaders, biological constants, and helpers used by
analyze_temporal_trajectories.py (the single remaining downstream script).
"""

import glob
import os
import warnings

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import config
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Output mode → directory mapping
# ---------------------------------------------------------------------------
MODE_DIRS = {
    "deconv": "outputs/deconv",
    "bulk": "outputs/bulk",
    "deconv_tyrosine": "outputs/deconv_tyrosine",
    "bulk_tyrosine": "outputs/bulk_tyrosine",
}

ALL_MODES = list(MODE_DIRS.keys())

CONDITIONS = ["Ttau", "AppP", "ApTt"]
TIMEPOINTS = ["2mo", "4mo", "6mo"]

# ---------------------------------------------------------------------------
# Cell-type groupings
# ---------------------------------------------------------------------------
GLIAL_CLUSTERS = ["Astrocytes", "Microglia", "Oligodendrocytes"]
NEURONAL_CLUSTERS = [
    "Erbb4-VIP-inhibitory-neurons",
    "Excitatory-Pyramidal",
    "Excitatory-Pyramidal-Satb2-Cux2",
    "Excitatory-Rorb",
    "Foxp2-Excitatory-Neurons-layers-6-and-2-3",
    "Striatal-medium-spiny-neuron",
    "glutamatergic-excitatory-neurons",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def get_output_dir(mode):
    """Return the absolute output directory path for a given mode."""
    return os.path.join(os.getcwd(), MODE_DIRS[mode])


def get_clusters(mode):
    """Return the list of cluster names for a mode."""
    if "bulk" in mode and "deconv" not in mode:
        return ["bulk"]
    return GLIAL_CLUSTERS + NEURONAL_CLUSTERS


def load_kinase_results(mode):
    """Load kinase_results.csv for a given mode (all kinases with significance_tier)."""
    path = os.path.join(get_output_dir(mode), "kinase_results.csv")
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_significant_kinases(mode):
    """Load only the 'significant' tier from kinase_results.csv."""
    df = load_kinase_results(mode)
    if df.empty or "significance_tier" not in df.columns:
        return df
    return df[df["significance_tier"] == "significant"].copy()


def load_kinase_summary(mode):
    """Load kinase_summary.csv for a given mode."""
    path = os.path.join(get_output_dir(mode), "kinase_summary.csv")
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_enrichment_summary(mode):
    """Load enrichment_summary.csv for a given mode."""
    path = os.path.join(get_output_dir(mode), "enrichment_summary.csv")
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_global_ranking(mode):
    """Load rankings/global_ranking.csv for a given mode."""
    path = os.path.join(get_output_dir(mode), "rankings", "global_ranking.csv")
    if not os.path.exists(path):
        warnings.warn(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_all_enrichment_results(mode):
    """Load all per-comparison enrichment CSVs into a dict.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are comparison names (filename stem without _enrichment_results.csv),
        values are DataFrames with kinase as index.
    """
    result_dir = os.path.join(get_output_dir(mode), "enrichment_results")
    if not os.path.isdir(result_dir):
        warnings.warn(f"Directory not found: {result_dir}")
        return {}
    results = {}
    for path in sorted(glob.glob(os.path.join(result_dir, "*_enrichment_results.csv"))):
        name = os.path.basename(path).replace("_enrichment_results.csv", "")
        df = pd.read_csv(path, index_col=0)
        results[name] = df
    return results


def parse_comparison_name(name, mode):
    """Extract condition, timepoint, cluster from a comparison filename stem.

    Returns dict with keys: condition, timepoint, cluster (or None if unparseable).
    """
    # Deconv pattern: ma_2mo_WTyp_Astrocytes_vs_ma_2mo_Ttau_Astrocytes_pct5_lff0.01
    # Bulk pattern: M_2mo_WT_vs_M_2mo_T22_pct5_lff0.01
    parts = name.split("_vs_")
    if len(parts) != 2:
        return None

    fg_part = parts[1]  # e.g., ma_2mo_Ttau_Astrocytes_pct5_lff0.01 or M_2mo_T22_pct5_lff0.01

    if "bulk" in mode and "deconv" not in mode:
        # Bulk: M_2mo_T22_pct5_lff0.01 or M_2mo_T22_APP_pct5_lff0.01
        tokens = fg_part.split("_")
        timepoint = tokens[1]
        # Condition suffix — reverse map from bulk names to canonical
        # In filenames, "/" in condition names becomes "_" (e.g. T22/APP → T22_APP)
        reverse_cond = {}
        for canon, suffix in config.BULK_CONDITION_MAP.items():
            reverse_cond[suffix.replace("/", "_")] = canon
        # Try longest token spans first to avoid partial matches (T22 before T22_APP)
        cond_raw = None
        for end in range(len(tokens), 2, -1):
            candidate = "_".join(tokens[2:end])
            if candidate in reverse_cond:
                cond_raw = candidate
                break
        condition = reverse_cond.get(cond_raw, cond_raw) if cond_raw else None
        return {"condition": condition, "timepoint": timepoint, "cluster": "bulk"}
    else:
        # Deconv: ma_2mo_Ttau_Astrocytes_pct5_lff0.01
        bg_part = parts[0]  # ma_2mo_WTyp_Astrocytes
        bg_tokens = bg_part.split("_")
        # bg: [ma, 2mo, WTyp, Astrocytes] (cluster may have hyphens but not underscores)
        timepoint = bg_tokens[1]
        # Cluster from bg: everything after WTyp
        cluster = "_".join(bg_tokens[3:])

        fg_tokens = fg_part.split("_")
        # fg: [ma, 2mo, Ttau, Astrocytes, pct5, lff0.01] or [ma, 2mo, ApTt, ...]
        condition = fg_tokens[2]
        return {"condition": condition, "timepoint": timepoint, "cluster": cluster}


# ---------------------------------------------------------------------------
# Re-filtering
# ---------------------------------------------------------------------------

def refilter_enrichment(mode, pval_thresh, lff_thresh, pct_filter=None):
    """Re-apply significance filters to enrichment CSVs at alternative thresholds.

    Parameters
    ----------
    mode : str
        Output mode (e.g., "deconv", "bulk").
    pval_thresh : float
        Adjusted p-value threshold.
    lff_thresh : float
        Minimum |LFF| threshold.
    pct_filter : float or None
        If set, only read CSVs whose filename contains ``pct{pct_filter}``.

    Returns
    -------
    pd.DataFrame
        Columns: kinase, condition, timepoint, cluster, direction, lff, adj_pval, pct.
    """
    all_results = load_all_enrichment_results(mode)
    pct_tag = f"pct{pct_filter}" if pct_filter is not None else None

    rows = []
    for name, df in all_results.items():
        if pct_tag and pct_tag not in name:
            continue
        meta = parse_comparison_name(name, mode)
        if meta is None:
            continue
        sig = df[
            (df["most_sig_fisher_adj_pval"] <= pval_thresh) &
            (df["most_sig_log2_freq_factor"].abs() >= lff_thresh)
        ]
        for kinase in sig.index:
            lff = sig.loc[kinase, "most_sig_log2_freq_factor"]
            rows.append({
                "kinase": kinase,
                "condition": meta["condition"],
                "timepoint": meta["timepoint"],
                "cluster": meta["cluster"],
                "direction": "+" if lff > 0 else "-",
                "lff": lff,
                "adj_pval": sig.loc[kinase, "most_sig_fisher_adj_pval"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_analysis_dir(analysis_name, mode):
    """Create and return the output directory for an analysis within a mode."""
    out = os.path.join(get_output_dir(mode), analysis_name)
    os.makedirs(out, exist_ok=True)
    return out


def save_fig(fig, path, dpi=200):
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def graceful_empty(df, label):
    """Check if a DataFrame is empty. If so, warn and return True."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        print(f"  [SKIP] {label}: no data")
        return True
    if isinstance(df, dict) and len(df) == 0:
        print(f"  [SKIP] {label}: no data")
        return True
    return False



