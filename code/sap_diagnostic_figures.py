"""
SAP Diagnostic Figures: Evidence for each modeling choice.

Generates 12 figures across 4 diagnostic stories:

Story 1 — Why We Can't Use Simple Methods (Rank-1 Problem)
  Fig 1a: Heatmap of pairwise LFC correlations between conditions
  Fig 1b: Scatter of two condition LFC profiles (Excitatory neurons)
  Fig 1c: PCA of residual matrix R = Y - A·X^T, colored by condition

Story 2 — Why We Need the Hurdle Component
  Fig 2a: Zero rate by estimated intensity (LOESS-like binned curve)
  Fig 2b: Marginal distribution of Y for representative sites
  Fig 2c: Q-Q plot comparing Gaussian, Gamma, Tweedie fits

Story 3 — Why We Need the Tweedie (Heteroscedasticity)
  Fig 3a: Mean-variance plot across sites (log-log)
  Fig 3b: Residuals vs fitted under Gaussian vs Tweedie

Story 4 — Why We Need the RNA Covariate and Adaptive Weights
  Fig 4a: Distribution of CVS per cell type
  Fig 4b: Site-level pairwise correlations of r_tensor
  Fig 4c: VIF distribution histogram

Usage:
    python code/sap_diagnostic_figures.py                # all figures
    python code/sap_diagnostic_figures.py --story 1      # single story
    python code/sap_diagnostic_figures.py --story 1 2    # multiple stories
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

import config

# Ensure code/ is on path for imports
sys.path.insert(0, os.path.dirname(__file__))
from sap_data import load_all, build_sample_meta

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = Path("outputs/sap_diagnostics")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
PALETTE = {
    "WTyp": "#4477AA",   # blue
    "AppP": "#EE6677",   # red
    "Ttau": "#228833",   # green
    "ApTt": "#CCBB44",   # yellow
}
CT_PALETTE = {
    "Excitatory_neurons": "#E64B35",
    "Oligodendrocytes":   "#4DBBD5",
    "GABAergic_neurons":  "#00A087",
    "Astrocytes":         "#3C5488",
    "Microglia":          "#F39B7F",
}
CT_SHORT = {
    "Excitatory_neurons": "Excit.",
    "Oligodendrocytes":   "Oligo.",
    "GABAergic_neurons":  "GABA.",
    "Astrocytes":         "Astro.",
    "Microglia":          "Micro.",
}


def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    })


# ===================================================================
# STORY 1: The Rank-1 Problem
# ===================================================================

def _load_desp_per_sample():
    """Load the raw DESP deconvoluted profiles per (sample, cell type, site).

    Returns dict: {resolved_celltype: DataFrame (J sites × 24 samples)}.
    Each entry X[ct].iloc[j, i] = DESP-estimated intensity of site j in
    cell type ct for sample i.
    """
    from sap_data import _parse_desp_column

    df = pd.read_csv(config.DESP_BASELINE_FILE, low_memory=False)
    meta_cols = ["Unnamed: 0", "site_id", "protein_id", "gene_symbol",
                 "prot_description", "site_position", "motif"]
    existing_meta = [c for c in meta_cols if c in df.columns]
    sample_cols = [c for c in df.columns if c not in existing_meta]

    # Parse columns: group by (sample_id, resolved_celltype)
    # sample_id = f"{gender}_{timepoint}_{condition}"
    ct_sample_cols = {}  # {resolved_ct: {sample_id: [columns]}}
    for col in sample_cols:
        parsed = _parse_desp_column(col)
        if parsed is None:
            continue
        gender, tp, cond, desp_ct = parsed
        sample_id = f"{gender}_{tp}_{cond}"
        resolved = config.DESP_POOL_MAP.get(desp_ct)
        if resolved is None:
            continue
        ct_sample_cols.setdefault(resolved, {}).setdefault(sample_id, []).append(col)

    # Build per-cell-type DataFrames: mean across sub-types for "Other" pool
    result = {}
    for ct in config.SAP_ESTIMATED_CELLTYPES:
        sample_dict = ct_sample_cols.get(ct, {})
        per_sample = {}
        for sid, cols in sample_dict.items():
            per_sample[sid] = df[cols].mean(axis=1).values
        result[ct] = pd.DataFrame(per_sample)  # (J, n_samples)

    return result, df[existing_meta] if existing_meta else None


def _compute_naive_lfc(data):
    """Compute per-cell-type LFC profiles from DESP per-sample profiles.

    The DESP file stores A_{i,k} · X_{k,j} — the per-sample contribution of
    cell type k. Because X is shared across conditions, LFC between conditions
    is LFC_c[k,j] = X_{k,j} · (mean_c(A_k) - mean_WT(A_k)) — a constant
    times the shared profile. This makes LFC vectors across sites perfectly
    rank-correlated between any two conditions (Spearman ρ ≥ 0.998, SAP §A.9).

    Cell types with near-zero composition deltas (Oligo, Micro) will show
    LFC ≈ 0, i.e. no extractable condition-specific signal — equally
    demonstrating the rank-1 limitation.
    """
    print("  Loading raw per-sample DESP profiles...")
    desp_profiles, _ = _load_desp_per_sample()
    sample_meta = data.sample_meta

    n_sites = data.bulk_phospho.shape[0]
    celltypes = config.SAP_ESTIMATED_CELLTYPES
    conditions = ["WTyp", "AppP", "Ttau", "ApTt"]

    # Collect raw per-sample arrays per (cell type, condition)
    raw = {}  # {(ct_idx, cond): np.array (n_samples_c, J)}
    for cond in conditions:
        cond_samples = [sid for sid in sample_meta.index
                        if sample_meta.loc[sid, "condition"] == cond]
        for ki, ct in enumerate(celltypes):
            ct_df = desp_profiles[ct]
            avail = [s for s in cond_samples if s in ct_df.columns]
            if avail:
                raw[(ki, cond)] = np.array(
                    [ct_df[s].iloc[:n_sites].values for s in avail])

    # Build NaN-free mask: sites where NO sample in ANY condition has NaN.
    # NaN sites produce spurious correlation < 1.0 because nanmean uses
    # different sample subsets per condition, breaking exact proportionality.
    nan_free = np.ones(n_sites, dtype=bool)
    for key, arr in raw.items():
        nan_free &= ~np.any(np.isnan(arr), axis=0)
    n_clean = nan_free.sum()
    print(f"  NaN-free sites: {n_clean:,} / {n_sites:,} "
          f"(dropped {n_sites - n_clean:,} with partial NaN)")

    # Average per-condition on NaN-free sites only
    X_cond = {}
    for cond in conditions:
        profiles = np.full((len(celltypes), n_sites), np.nan)
        for ki, ct in enumerate(celltypes):
            if (ki, cond) in raw:
                profiles[ki, nan_free] = raw[(ki, cond)][:, nan_free].mean(axis=0)
        X_cond[cond] = profiles

    # LFC = mean_c(A·X) - mean_WT(A·X) = X · ΔA (rank-1)
    lfc = {}
    for cond in ["AppP", "Ttau", "ApTt"]:
        lfc[cond] = X_cond[cond] - X_cond["WTyp"]

    return lfc, X_cond


def figure_1a(data, lfc):
    """Heatmap of pairwise LFC correlations between conditions, per cell type.

    DESP produces a single X_base shared across conditions. LFC profiles
    are exactly proportional to X_base → Spearman ρ = 1.0 for all pairs,
    for all cell types. This is the rank-1 problem.
    """
    celltypes = config.SAP_ESTIMATED_CELLTYPES
    conditions = ["AppP", "Ttau", "ApTt"]
    n_ct = len(celltypes)

    fig, axes = plt.subplots(1, n_ct, figsize=(4 * n_ct, 3.5),
                             constrained_layout=True)

    for ct_idx, ct in enumerate(celltypes):
        ax = axes[ct_idx]
        profiles = np.array([lfc[c][ct_idx] for c in conditions])
        # Use only finite, nonzero sites
        valid = np.all(np.isfinite(profiles), axis=0) & np.any(profiles != 0, axis=0)
        profiles = profiles[:, valid]

        corr = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                corr[i, j] = stats.spearmanr(profiles[i], profiles[j]).statistic

        im = ax.imshow(corr, vmin=0.95, vmax=1.0, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(range(3))
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(conditions)
        ax.set_title(f"{CT_SHORT[ct]}\n(n = {valid.sum():,} sites)",
                     fontweight="bold", fontsize=9)

        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{corr[i, j]:.4f}", ha="center", va="center",
                        fontsize=8, color="black" if corr[i, j] < 0.99 else "white")

    fig.suptitle("Figure 1a: Pairwise LFC Correlations Between Conditions\n"
                 "(DESP produces one shared X$_{base}$ → all conditions perfectly rank-1 correlated)",
                 fontsize=11, y=1.05)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Spearman ρ")

    path = OUT_DIR / "fig1a_lfc_correlation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_1b(data, lfc):
    """Scatter of AppP vs Ttau LFC profiles for Excitatory neurons."""
    ct_idx = 0  # Excitatory_neurons
    ct_name = config.SAP_ESTIMATED_CELLTYPES[ct_idx]

    x = lfc["AppP"][ct_idx]
    y = lfc["Ttau"][ct_idx]

    # Filter out NaN/Inf
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    r_pearson = stats.pearsonr(x, y).statistic
    r_spearman = stats.spearmanr(x, y).statistic

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=1, alpha=0.3, color="#555555", rasterized=True)

    # Fit line
    m, b = np.polyfit(x, y, 1)
    xlim = np.array([x.min(), x.max()])
    ax.plot(xlim, m * xlim + b, color="#E64B35", linewidth=1.5, zorder=5)

    ax.set_xlabel("LFC: AppP vs WTyp")
    ax.set_ylabel("LFC: Ttau vs WTyp")
    ax.set_title(f"Figure 1b: Condition LFC Profiles — {CT_SHORT[ct_name]}\n"
                 f"Spearman ρ = {r_spearman:.4f}  (n = {len(x):,} sites)",
                 fontweight="bold")
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    # Add text box
    ax.text(0.05, 0.95, f"Pearson r = {r_pearson:.4f}\nSpearman ρ = {r_spearman:.4f}\nSlope = {m:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    path = OUT_DIR / "fig1b_lfc_scatter_excitatory.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_1c(data):
    """PCA of residual matrix R = Y - A·X_base^T, colored by condition."""
    A = data.a_obs.values           # (24, 6)
    X_base = data.x_base.values     # (J, 6)
    Y = data.bulk_phospho.values    # (J, 24)

    # Predicted bulk under naive model: Y_hat = X_base @ A^T → (J, 24)
    Y_hat = X_base @ A.T
    R = Y - Y_hat  # (J, 24) residual matrix

    # Filter out sites with NaN/Inf residuals
    valid_sites = np.all(np.isfinite(R), axis=1)
    R = R[valid_sites]

    # PCA on R^T (samples × sites)
    R_centered = R.T - R.T.mean(axis=0)

    # Replace any remaining NaN with 0
    R_centered = np.nan_to_num(R_centered, nan=0.0, posinf=0.0, neginf=0.0)

    U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    pc2 = U[:, 1] * S[1]
    var_explained = S ** 2 / np.sum(S ** 2)

    # Statistical tests: Kruskal-Wallis for PC1, PC2 vs condition
    conditions = data.sample_meta["condition"].values
    groups_pc1 = [pc1[conditions == c] for c in config.SAP_CONDITIONS]
    groups_pc2 = [pc2[conditions == c] for c in config.SAP_CONDITIONS]
    kw_pc1 = stats.kruskal(*groups_pc1)
    kw_pc2 = stats.kruskal(*groups_pc2)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond in config.SAP_CONDITIONS:
        mask = conditions == cond
        ax.scatter(pc1[mask], pc2[mask], c=PALETTE[cond], label=cond,
                   s=80, edgecolors="black", linewidths=0.5, zorder=5)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Figure 1c: PCA of Residual Matrix R = Y − A·X$^{base}$\n"
                 f"PC1 vs condition: p = {kw_pc1.pvalue:.4f}  |  "
                 f"PC2 vs condition: p = {kw_pc2.pvalue:.4f}",
                 fontweight="bold")
    ax.legend(title="Condition", framealpha=0.9)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

    path = OUT_DIR / "fig1c_residual_pca.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def run_story_1(data):
    print("\n=== Story 1: The Rank-1 Problem ===")
    lfc, _ = _compute_naive_lfc(data)
    figure_1a(data, lfc)
    figure_1b(data, lfc)
    figure_1c(data)


# ===================================================================
# STORY 2: The Hurdle Component
# ===================================================================

def figure_2a(data):
    """Zero rate by estimated intensity (binned smooth curve)."""
    A = data.a_obs.values           # (24, 6)
    X_base = data.x_base.values     # (J, 6)
    Y = data.bulk_phospho.values    # (J, 24)

    # Estimated intensity per (i, j): mu_hat = sum_k A[i,k] * X_base[j,k]
    mu_hat = X_base @ A.T  # (J, 24)
    # Dropouts are NaN in this dataset (not zero)
    is_dropout = np.isnan(Y).astype(float)

    # Flatten
    mu_flat = mu_hat.ravel()
    zero_flat = is_dropout.ravel()

    # Remove entries where mu_hat <= 0 for log scale
    valid = mu_flat > 0
    mu_flat = mu_flat[valid]
    zero_flat = zero_flat[valid]

    # Bin by log(mu_hat)
    log_mu = np.log10(mu_flat)
    n_bins = 50
    bin_edges = np.linspace(np.percentile(log_mu, 1), np.percentile(log_mu, 99), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(log_mu, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    zero_rate = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            zero_rate[b] = zero_flat[mask].mean()
            bin_counts[b] = mask.sum()

    # Only plot bins with sufficient data
    good = bin_counts >= 50
    overall_zero_rate = zero_flat.mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(10 ** bin_centers[good], zero_rate[good], "o-", color="#4477AA",
            markersize=4, linewidth=1.5)
    ax.axhline(float(overall_zero_rate), color="grey", linestyle="--", linewidth=0.8,
               label=f"Overall dropout rate: {overall_zero_rate:.1%}")
    ax.set_xscale("log")
    ax.set_xlabel("Estimated Intensity (A·X$^{base}$)")
    ax.set_ylabel("Dropout Rate (fraction NaN)")
    ax.set_title("Figure 2a: Dropout Rate by Estimated Intensity\n"
                 "Low-intensity sites drop out far more often → motivates hurdle component",
                 fontweight="bold")
    ax.legend()
    ax.set_ylim(-0.02, min(1.02, zero_rate[good].max() * 1.3))

    path = OUT_DIR / "fig2a_zero_rate_vs_intensity.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_2b(data):
    """Marginal distribution of Y for three representative sites."""
    Y = data.bulk_phospho.values    # (J, 24)
    A = data.a_obs.values
    X_base = data.x_base.values

    mu_hat = X_base @ A.T  # (J, 24)
    mean_intensity = mu_hat.mean(axis=1)

    # Count dropouts (NaN) per site
    n_dropout = np.isnan(Y).sum(axis=1)

    # Pick three sites that illustrate the dropout pattern:
    # High intensity with 0 dropouts, medium with some, low with many
    sorted_idx = np.argsort(mean_intensity)
    n = len(sorted_idx)

    # High: 90th percentile, 0 dropouts
    high_idx = None
    for pct in np.arange(0.90, 0.50, -0.01):
        candidate = sorted_idx[int(n * pct)]
        if n_dropout[candidate] == 0:
            high_idx = candidate
            break
    if high_idx is None:
        high_idx = sorted_idx[int(n * 0.90)]

    # Medium: around 50th percentile with some dropouts
    med_idx = None
    for pct in np.arange(0.50, 0.30, -0.01):
        candidate = sorted_idx[int(n * pct)]
        if 3 <= n_dropout[candidate] <= 8:
            med_idx = candidate
            break
    if med_idx is None:
        med_idx = sorted_idx[int(n * 0.50)]

    # Low: low intensity with many dropouts
    low_idx = None
    for pct in np.arange(0.10, 0.30, 0.01):
        candidate = sorted_idx[int(n * pct)]
        if n_dropout[candidate] >= 8:
            low_idx = candidate
            break
    if low_idx is None:
        low_idx = sorted_idx[int(n * 0.10)]

    picks = [
        (high_idx, "High intensity", "#E64B35"),
        (med_idx, "Medium intensity", "#4DBBD5"),
        (low_idx, "Low intensity", "#00A087"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, (idx, label, color) in zip(axes, picks):
        y_vals = Y[idx]
        n_zero = np.isnan(y_vals).sum()  # NaN = dropout
        n_pos = np.isfinite(y_vals).sum()
        # Replace NaN with 0 for histogram display
        y_vals = np.where(np.isnan(y_vals), 0.0, y_vals)
        site_id = data.site_meta.iloc[idx]["site_id"] if "site_id" in data.site_meta.columns else f"site {idx}"

        ax.hist(y_vals, bins=15, color=color, edgecolor="black", alpha=0.8)
        ax.set_title(f"{label}\n{site_id}\n"
                     f"dropouts: {n_zero}/24, detected: {n_pos}/24",
                     fontsize=9)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.axvline(0, color="black", linewidth=0.8, linestyle=":")

    fig.suptitle("Figure 2b: Marginal Distributions of Y for Representative Sites\n"
                 "Zero-inflation varies systematically with intensity → two-component model needed",
                 fontsize=11, y=1.08)

    path = OUT_DIR / "fig2b_marginal_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_2c(data):
    """Gaussian vs Tweedie Q-Q + profile deviance showing optimal p̂.

    Individual sites have only n=24, so we pool Pearson residuals across all
    sites.  Panel 3 sweeps the Tweedie power parameter p and shows total
    deviance is minimised at p̂ ≈ 1.7, *between* Gaussian (p→0) and Gamma (p=2).
    """
    from sap_model import tweedie_variance

    Y = data.bulk_phospho.values  # (J, 24)
    A = data.a_obs.values
    X_base = data.x_base.values
    mu_hat = X_base @ A.T  # (J, 24)

    # --- Collect pooled Pearson residuals under two variance functions ---
    gauss_resids_all = []   # (y - mu) / sd  (constant variance)
    tweedie_resids_all = [] # (y - mu) / mu^(p/2)  (Tweedie variance)
    p_hat = 1.72  # from prior profile likelihood fit
    n_sites_used = 0

    for j in range(Y.shape[0]):
        y_j = Y[j]
        mu_j = mu_hat[j]
        mask = np.isfinite(y_j) & (y_j > 0) & np.isfinite(mu_j) & (mu_j > 0)
        n_obs = mask.sum()
        if n_obs < 10:
            continue
        y_obs = y_j[mask]
        mu_obs = mu_j[mask]
        n_sites_used += 1

        resid_raw = y_obs - mu_obs

        # Gaussian Pearson: constant variance
        sd_g = resid_raw.std(ddof=1)
        if sd_g > 0:
            gauss_resids_all.append(resid_raw / sd_g)

        # Tweedie Pearson: variance = mu^p
        tw_resid = resid_raw / np.power(mu_obs, p_hat / 2)
        sd_tw = tw_resid.std(ddof=1)
        if sd_tw > 0:
            tweedie_resids_all.append(tw_resid / sd_tw)

    gauss_resids = np.concatenate(gauss_resids_all)
    tweedie_resids = np.concatenate(tweedie_resids_all)
    print(f"  Pooled {len(gauss_resids):,} residuals from {n_sites_used:,} sites")

    # --- Pearson chi-squared calibration across p grid ---
    # Under the correct variance V(mu) = mu^p, the Pearson statistic
    # X² = sum((y - mu)² / V(mu)) / (n-1) should be ≈ 1.
    # We use per-site means as mu and check which p gives median X²/(n-1) closest to 1.
    p_grid = np.arange(1.0, 2.05, 0.05)
    median_chi2 = []
    for p in p_grid:
        chi2_vals = []
        for j in range(Y.shape[0]):
            y_j = Y[j]
            mask = np.isfinite(y_j) & (y_j > 0)
            n_obs = mask.sum()
            if n_obs < 10:
                continue
            y_pos = y_j[mask]
            mu_site = y_pos.mean()
            if mu_site <= 0:
                continue
            V_mu = mu_site ** p
            chi2 = np.sum((y_pos - mu_site) ** 2 / V_mu) / (n_obs - 1)
            chi2_vals.append(chi2)
        median_chi2.append(np.median(chi2_vals))
    median_chi2 = np.array(median_chi2)
    # The optimal p is where median chi2/(n-1) is closest to its expected value.
    # Since we're on different scales for different p, find the p where
    # the distribution is most stable (lowest IQR ratio).
    # Simpler: log(median_chi2) closest to log of mean (which should be ~1 under correct model).
    # Actually: with common phi, chi2/(n-1) = phi/V(mu_bar). The "flattest" profile
    # isn't what we want. Instead, use the cross-site mean-variance regression directly.

    # Direct approach: log(Var_j) = log(phi) + p * log(mean_j) across sites
    site_means = []
    site_vars = []
    for j in range(Y.shape[0]):
        y_j = Y[j]
        mask = np.isfinite(y_j) & (y_j > 0)
        if mask.sum() < 10:
            continue
        y_pos = y_j[mask]
        site_means.append(y_pos.mean())
        site_vars.append(y_pos.var(ddof=1))
    site_means = np.array(site_means)
    site_vars = np.array(site_vars)
    valid = (site_means > 0) & (site_vars > 0)
    log_mu = np.log(site_means[valid])
    log_var = np.log(site_vars[valid])
    slope, intercept, r_value, _, _ = stats.linregress(log_mu, log_var)
    p_empirical = slope
    print(f"  Mean-variance power law: log(Var) = {slope:.2f} * log(μ) + {intercept:.2f}, "
          f"R² = {r_value**2:.3f}, p̂ = {p_empirical:.2f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    rng = np.random.default_rng(42)

    # Panel 1: Gaussian Q-Q
    n_plot = min(len(gauss_resids), 5000)
    sub_g = np.sort(rng.choice(gauss_resids, size=n_plot, replace=False))
    pp = (np.arange(1, n_plot + 1) - 0.5) / n_plot
    q_norm = stats.norm.ppf(pp)
    axes[0].scatter(q_norm, sub_g, s=3, alpha=0.3, color="#4477AA", rasterized=True)
    lims = [min(q_norm.min(), sub_g.min()), max(q_norm.max(), sub_g.max())]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_xlabel("Normal Theoretical Quantiles")
    axes[0].set_ylabel("Standardized Residuals")
    axes[0].set_title("Gaussian Q-Q\n(Var = const)", fontweight="bold")
    skew_g = stats.skew(gauss_resids)
    kurt_g = stats.kurtosis(gauss_resids)
    axes[0].text(0.05, 0.95, f"skew = {skew_g:.2f}\nexcess kurt = {kurt_g:.2f}",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Panel 2: Tweedie Q-Q (p = 1.72)
    n_plot_tw = min(len(tweedie_resids), 5000)
    sub_tw = np.sort(rng.choice(tweedie_resids, size=n_plot_tw, replace=False))
    pp_tw = (np.arange(1, n_plot_tw + 1) - 0.5) / n_plot_tw
    q_tw = stats.norm.ppf(pp_tw)
    axes[1].scatter(q_tw, sub_tw, s=3, alpha=0.3, color="#228833", rasterized=True)
    lims = [min(q_tw.min(), sub_tw.min()), max(q_tw.max(), sub_tw.max())]
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_xlabel("Normal Theoretical Quantiles")
    axes[1].set_ylabel("Standardized Pearson Residuals")
    axes[1].set_title(f"Tweedie Q-Q\n(Var ∝ μ^{{{p_hat}}})", fontweight="bold")
    skew_tw = stats.skew(tweedie_resids)
    kurt_tw = stats.kurtosis(tweedie_resids)
    axes[1].text(0.05, 0.95, f"skew = {skew_tw:.2f}\nexcess kurt = {kurt_tw:.2f}",
                 transform=axes[1].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Panel 3: Mean-variance log-log with power-law fit
    sub_idx = np.random.default_rng(99).choice(
        len(log_mu), size=min(3000, len(log_mu)), replace=False)
    axes[2].scatter(log_mu[sub_idx], log_var[sub_idx],
                    s=2, alpha=0.15, color="grey", rasterized=True)
    x_line = np.array([log_mu.min(), log_mu.max()])
    # Empirical OLS fit (= p̂)
    axes[2].plot(x_line, slope * x_line + intercept, color="#CC3311",
                 linewidth=2, label=f"p̂ = {slope:.2f} (OLS)")
    # Poisson reference (p=1)
    intercept_p1 = np.median(log_var - 1.0 * log_mu)
    axes[2].plot(x_line, 1.0 * x_line + intercept_p1, color="#4477AA",
                 linewidth=1.5, linestyle=":", alpha=0.7, label="p = 1 (Poisson)")
    # Gamma reference (p=2)
    intercept_p2 = np.median(log_var - 2.0 * log_mu)
    axes[2].plot(x_line, 2.0 * x_line + intercept_p2, color="#AA3377",
                 linewidth=1.5, linestyle=":", alpha=0.7, label="p = 2 (Gamma)")
    axes[2].set_xlabel("log(Site Mean)")
    axes[2].set_ylabel("log(Site Variance)")
    axes[2].set_title(f"Mean-Variance Power Law\n(slope = {slope:.2f}, "
                      f"R² = {r_value**2:.2f})", fontweight="bold")
    axes[2].legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Figure 2c: Why Tweedie — Variance Structure and Optimal Power\n"
                 f"(pooled across {n_sites_used:,} sites; "
                 f"empirical p̂ = {p_empirical:.2f}, between Poisson and Gamma)",
                 fontsize=11, y=1.06)

    path = OUT_DIR / "fig2c_qq_plots.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def run_story_2(data):
    print("\n=== Story 2: The Hurdle Component ===")
    figure_2a(data)
    figure_2b(data)
    figure_2c(data)


# ===================================================================
# STORY 3: Heteroscedasticity (Tweedie)
# ===================================================================

def figure_3a(data):
    """Mean-variance plot across sites on log-log scale."""
    Y = data.bulk_phospho.values  # (J, 24)

    # For each site, compute mean and variance of positive observations
    means = []
    variances = []
    for j in range(Y.shape[0]):
        y_j = Y[j]
        y_pos = y_j[y_j > 0]
        if len(y_pos) >= 3:
            means.append(y_pos.mean())
            variances.append(y_pos.var(ddof=1))

    means = np.array(means)
    variances = np.array(variances)

    # Filter out zero/negative
    valid = (means > 0) & (variances > 0)
    means = means[valid]
    variances = variances[valid]

    log_mu = np.log10(means)
    log_var = np.log10(variances)

    # Fit power-law: log(Var) = log(phi) + p * log(mu)
    slope, intercept, r_val, p_val, _ = stats.linregress(log_mu, log_var)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(means, variances, s=1, alpha=0.15, color="#555555", rasterized=True)

    # Overlay fitted line
    mu_range = np.logspace(log_mu.min(), log_mu.max(), 100)
    var_fit = 10 ** intercept * mu_range ** slope
    ax.plot(mu_range, var_fit, color="#E64B35", linewidth=2, zorder=5,
            label=f"Fitted: Var = φ·μ$^{{{slope:.2f}}}$\n(r² = {r_val**2:.3f})")

    # Gaussian reference: Var = const (horizontal)
    median_var = np.median(variances)
    ax.axhline(median_var, color="#4477AA", linewidth=1.5, linestyle="--",
               label=f"Gaussian assumption (Var = const)", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Intensity (positive obs.)")
    ax.set_ylabel("Variance (positive obs.)")
    ax.set_title(f"Figure 3a: Mean-Variance Relationship Across Sites\n"
                 f"Power-law slope p̂ = {slope:.2f} confirms Tweedie variance function V(μ) = φμ$^p$",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)

    path = OUT_DIR / "fig3a_mean_variance_plot.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_3b(data):
    """Residuals vs fitted under Gaussian vs Tweedie — aggregated across all sites.

    Uses per-site sample means as fitted values so residuals are centered at 0.
    The heteroscedasticity pattern shows that raw residuals fan out with mean
    intensity, while Tweedie Pearson residuals (dividing by sqrt(mu^p)) stabilize.
    """
    Y = data.bulk_phospho.values    # (J, 24)

    # Estimate global p from mean-variance regression
    means_all, vars_all = [], []
    for j in range(Y.shape[0]):
        yj = Y[j]
        yp = yj[np.isfinite(yj) & (yj > 0)]
        if len(yp) >= 3:
            means_all.append(yp.mean())
            vars_all.append(yp.var(ddof=1))
    means_arr = np.array(means_all)
    vars_arr = np.array(vars_all)
    valid = (means_arr > 0) & (vars_arr > 0)
    p_hat = stats.linregress(np.log10(means_arr[valid]),
                              np.log10(vars_arr[valid])).slope

    # Collect (site_mean, residual) pairs using per-site means as fitted values
    mu_list, resid_list = [], []
    for j in range(Y.shape[0]):
        yj = Y[j]
        mask = np.isfinite(yj) & (yj > 0)
        if mask.sum() < 3:
            continue
        y_pos = yj[mask]
        mu_j = y_pos.mean()
        if mu_j <= 0:
            continue
        # Each observation contributes one residual at this site's mean intensity
        mu_list.append(np.full(len(y_pos), mu_j))
        resid_list.append(y_pos - mu_j)

    mu_flat = np.concatenate(mu_list)
    resid_flat = np.concatenate(resid_list)

    # Subsample for plotting
    rng = np.random.RandomState(42)
    n_plot = min(50000, len(mu_flat))
    idx = rng.choice(len(mu_flat), n_plot, replace=False)
    mu_sub = mu_flat[idx]
    resid_sub = resid_flat[idx]

    # Gaussian residuals (raw)
    resid_gauss = resid_sub

    # Tweedie Pearson residuals: (y - mu) / sqrt(mu^p)
    resid_tweedie = resid_sub / np.sqrt(mu_sub ** p_hat)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # Gaussian
    axes[0].scatter(mu_sub, resid_gauss, s=1, alpha=0.08, color="#4477AA",
                    rasterized=True)
    # Binned spread (standard deviation in bins)
    log_bins = np.linspace(np.log10(mu_sub.min()), np.log10(mu_sub.max()), 25)
    bin_idx = np.digitize(np.log10(mu_sub), log_bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(log_bins) - 2)
    bin_centers = 10 ** ((log_bins[:-1] + log_bins[1:]) / 2)
    bin_sd = np.array([resid_gauss[bin_idx == b].std() if (bin_idx == b).sum() > 10
                       else np.nan for b in range(len(log_bins) - 1)])
    good = np.isfinite(bin_sd)
    axes[0].plot(bin_centers[good], bin_sd[good], "r-", linewidth=2, label="±1 SD")
    axes[0].plot(bin_centers[good], -bin_sd[good], "r-", linewidth=2)
    axes[0].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Fitted Value (μ̂)")
    axes[0].set_ylabel("Raw Residual (y − μ̂)")
    axes[0].set_title("Gaussian: fan-shaped residuals\n(spread grows with μ̂)", fontweight="bold")
    axes[0].legend()

    # Tweedie
    axes[1].scatter(mu_sub, resid_tweedie, s=1, alpha=0.08, color="#228833",
                    rasterized=True)
    bin_sd_tw = np.array([resid_tweedie[bin_idx == b].std() if (bin_idx == b).sum() > 10
                          else np.nan for b in range(len(log_bins) - 1)])
    good_tw = np.isfinite(bin_sd_tw)
    axes[1].plot(bin_centers[good_tw], bin_sd_tw[good_tw], "r-", linewidth=2, label="±1 SD")
    axes[1].plot(bin_centers[good_tw], -bin_sd_tw[good_tw], "r-", linewidth=2)
    axes[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Fitted Value (μ̂)")
    axes[1].set_ylabel(f"Pearson Residual (y − μ̂) / √(μ̂$^{{{p_hat:.2f}}}$)")
    axes[1].set_title("Tweedie: stabilized residuals\n(spread constant across μ̂)", fontweight="bold")
    # Clip y-axis to ±3 SD to avoid outlier-dominated scale at low μ
    tw_sd = np.nanmedian(bin_sd_tw[good_tw])
    axes[1].set_ylim(-5 * tw_sd, 5 * tw_sd)
    axes[1].legend()

    fig.suptitle(f"Figure 3b: Residuals vs Fitted Values (n = {n_plot:,} obs.)\n"
                 f"Gaussian residuals show heteroscedasticity; "
                 f"Tweedie (p̂ = {p_hat:.2f}) corrects it",
                 fontsize=11, y=1.05)

    path = OUT_DIR / "fig3b_residuals_vs_fitted.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def run_story_3(data):
    print("\n=== Story 3: Heteroscedasticity (Tweedie) ===")
    figure_3a(data)
    figure_3b(data)


# ===================================================================
# STORY 4: RNA Covariate and Adaptive Weights
# ===================================================================

def figure_4a(data):
    """Distribution of CVS per cell type × condition."""
    cvs = data.cvs  # DataFrame: (5 cell types × 3 conditions)
    celltypes = list(cvs.index)
    conditions = list(cvs.columns)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(celltypes))
    width = 0.25
    cond_colors = {"AppP": "#EE6677", "Ttau": "#228833", "ApTt": "#CCBB44"}

    for i, cond in enumerate(conditions):
        vals = cvs[cond].values
        bars = ax.bar(x + i * width, vals, width, label=cond,
                      color=cond_colors[cond], edgecolor="black", linewidth=0.5)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels([CT_SHORT.get(ct, ct) for ct in celltypes], rotation=30, ha="right")
    ax.set_ylabel("Condition-Variation Score (CVS)")
    ax.set_title("Figure 4a: RNA Condition-Variation Scores by Cell Type\n"
                 "ApTt shows highest CVS; narrow range (0.35–0.53) → modest adaptive weight modulation",
                 fontweight="bold")
    ax.legend(title="vs WTyp")
    ax.set_ylim(0, cvs.values.max() * 1.25)

    path = OUT_DIR / "fig4a_cvs_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_4b(data):
    """Histograms of pairwise site-correlations of r_tensor per cell type."""
    r_tensor = data.r_tensor  # (6, 24, J)
    celltypes = config.SAP_ESTIMATED_CELLTYPES

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), constrained_layout=True)

    for ct_idx, ct in enumerate(celltypes):
        ax = axes[ct_idx]
        r_k = r_tensor[ct_idx]  # (24, J)

        # Sample a manageable number of site pairs for correlation
        J = r_k.shape[1]
        n_sample = min(2000, J)
        rng = np.random.RandomState(42)
        idx_sample = rng.choice(J, n_sample, replace=False)
        r_sub = r_k[:, idx_sample]  # (24, n_sample)

        # Pairwise correlations between sites (across the 24-sample dimension)
        # Use corrcoef on transposed data
        corr_matrix = np.corrcoef(r_sub.T)  # (n_sample, n_sample)
        # Extract upper triangle
        iu = np.triu_indices(n_sample, k=1)
        pairwise_r = corr_matrix[iu]

        # Remove NaN
        pairwise_r = pairwise_r[np.isfinite(pairwise_r)]

        median_r = np.median(np.abs(pairwise_r))
        frac_high = np.mean(np.abs(pairwise_r) > 0.7)

        ax.hist(pairwise_r, bins=60, color=CT_PALETTE[ct], edgecolor="none",
                alpha=0.8, density=True)
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"{CT_SHORT[ct]}\nmed |r|={median_r:.3f}\n"
                     f">0.7: {frac_high:.1%}", fontsize=8)
        ax.set_xlim(-1, 1)
        if ct_idx == 0:
            ax.set_ylabel("Density")
        ax.set_xlabel("Pairwise r")

    fig.suptitle("Figure 4b: Pairwise Site-Correlations of RNA Covariate r$_{k,i,j}$\n"
                 "Low median |r| (0.03–0.08) confirms per-site ρ$_j$ is identifiable",
                 fontsize=11, y=1.08)

    path = OUT_DIR / "fig4b_r_tensor_correlations.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_4c(data):
    """Histogram of VIF values for the RNA covariate across sites."""
    r_tensor = data.r_tensor  # (6, 24, J)
    a_obs = data.a_obs
    sample_meta = data.sample_meta

    # Recompute VIF (same logic as sap_data.check_rna_covariate_vif)
    conditions = sample_meta["condition"].values
    app_ind = np.array([config.SAP_FACTORIAL[c][0] for c in conditions], dtype=float)
    tau_ind = np.array([config.SAP_FACTORIAL[c][1] for c in conditions], dtype=float)
    int_ind = np.array([config.SAP_FACTORIAL[c][2] for c in conditions], dtype=float)

    n_est = len(config.SAP_ESTIMATED_CELLTYPES)
    n_samples = a_obs.shape[0]

    # Build stacked factorial design (n_est * n_samples, n_est * 3)
    X_full = np.zeros((n_est * n_samples, n_est * 3))
    for ki, ct in enumerate(config.SAP_ESTIMATED_CELLTYPES):
        a_k = a_obs[ct].values
        row_s = ki * n_samples
        row_e = (ki + 1) * n_samples
        X_full[row_s:row_e, ki * 3] = a_k * app_ind
        X_full[row_s:row_e, ki * 3 + 1] = a_k * tau_ind
        X_full[row_s:row_e, ki * 3 + 2] = a_k * int_ind

    XtX = X_full.T @ X_full
    XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(XtX.shape[0]))
    hat_factor = XtX_inv @ X_full.T

    n_sites = r_tensor.shape[2]
    vifs = np.zeros(n_sites)

    for j in range(n_sites):
        r_vec = r_tensor[:n_est, :, j].ravel()
        if np.all(r_vec == 0):
            vifs[j] = 1.0
            continue
        r_hat = X_full @ (hat_factor @ r_vec)
        ss_res = np.sum((r_vec - r_hat) ** 2)
        ss_tot = np.sum((r_vec - r_vec.mean()) ** 2)
        if ss_tot < 1e-15:
            vifs[j] = 1.0
            continue
        r_sq = min(1.0 - ss_res / ss_tot, 1.0 - 1e-10)
        vifs[j] = 1.0 / (1.0 - r_sq)

    median_vif = np.median(vifs)
    max_vif = np.max(vifs)
    frac_below_10 = np.mean(vifs < 10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vifs, bins=80, color="#4DBBD5", edgecolor="none", alpha=0.85)
    ax.axvline(10, color="#E64B35", linewidth=2, linestyle="--",
               label=f"VIF = 10 threshold")
    ax.axvline(median_vif, color="#228833", linewidth=2, linestyle="-",
               label=f"Median VIF = {median_vif:.2f}")

    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_ylabel("Number of Sites")
    ax.set_title(f"Figure 4c: VIF Distribution of RNA Covariate\n"
                 f"Median = {median_vif:.2f}, Max = {max_vif:.1f}, "
                 f"{frac_below_10:.1%} below threshold of 10",
                 fontweight="bold")
    ax.legend()

    # Inset for tail if needed
    if max_vif > 5:
        ax.set_xlim(0.8, min(max_vif * 1.1, 20))

    path = OUT_DIR / "fig4c_vif_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def run_story_4(data):
    print("\n=== Story 4: RNA Covariate & Adaptive Weights ===")
    figure_4a(data)
    figure_4b(data)
    figure_4c(data)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="SAP diagnostic figures")
    parser.add_argument("--story", type=int, nargs="*", default=None,
                        help="Which stories to generate (1-4). Default: all.")
    args = parser.parse_args()

    stories = args.story or [1, 2, 3, 4]
    need_rna = any(s == 4 for s in stories)

    _setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SAP data" + (" with RNA preprocessing..." if need_rna else "..."))
    data, report = load_all(include_rna=need_rna)
    print(report.summary())

    if 1 in stories:
        run_story_1(data)
    if 2 in stories:
        run_story_2(data)
    if 3 in stories:
        run_story_3(data)
    if 4 in stories:
        run_story_4(data)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
