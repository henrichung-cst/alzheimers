"""
Deconvolution Infeasibility Analysis
=====================================

Investigates whether cell-type deconvolution of bulk phosphoproteomics is
fundamentally limited by the composition geometry of the experimental design,
independent of the deconvolution method used.

Approach:
    1. SVD analysis of the observed composition matrix (A_obs)
    2. Noise-amplification quantification from the pseudoinverse
    3. Synthetic ground-truth experiments:
       - Positive control: well-conditioned composition → recovery succeeds
       - Negative control: actual A_obs composition → recovery fails
    4. Multi-method comparison (OLS, Ridge, NNLS) on both compositions
    5. Condition-number sweep: parametric transition from recoverable to not

All figures are saved to outputs/reports/deconvolution_infeasibility/.

Usage:
    python alz/supplementary/deconvolution_infeasibility.py --run
    python alz/supplementary/deconvolution_infeasibility.py --svd
    python alz/supplementary/deconvolution_infeasibility.py --synthetic
    python alz/supplementary/deconvolution_infeasibility.py --sweep
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── paths ────────────────────────────────────────────────────────────────────

A_OBS_FILE = config.A_OBS_FILE
OUT_DIR = os.path.join(config.REPO_ROOT, "outputs", "reports", "deconvolution_infeasibility")
os.makedirs(OUT_DIR, exist_ok=True)

# ── shared parameters ────────────────────────────────────────────────────────

RNG_SEED = 42
N_SITES = 500          # phosphosites in synthetic experiments
N_TRIALS = 200         # Monte Carlo repetitions for recovery statistics
NOISE_SD = 0.3         # bulk measurement noise (log2 scale, ~MAD of real data)
EFFECT_RANGE = (0.5, 2.0)  # cell-type-specific LFC range (absolute)

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})


# =============================================================================
# §1  SVD analysis of composition matrix
# =============================================================================

def load_composition():
    """Load A_obs and return (DataFrame, numpy array, cell-type names)."""
    df = pd.read_csv(A_OBS_FILE, sep="\t", index_col=0)
    cell_types = list(df.columns)
    A = df.values.astype(np.float64)
    return df, A, cell_types


def svd_analysis(A, cell_types):
    """SVD of composition matrix; return singular values and components."""
    # Center columns (mean-subtract) for interpretability
    A_centered = A - A.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(A_centered, full_matrices=False)
    return U, s, Vt


def effective_condition_number(A):
    """
    Condition number excluding the trivial zero singular value.

    Because composition rows sum to 1 (simplex constraint), one column is
    linearly dependent → the last SV is always ~0.  The *effective* condition
    number uses σ_1 / σ_{k-1} where k = n_types.
    """
    s = np.linalg.svd(A - A.mean(0), compute_uv=False)
    s_eff = s[s > 1e-10]  # drop near-zero modes
    return s_eff[0] / s_eff[-1], s_eff


def figure_svd_spectrum(A, cell_types):
    """
    Figure 1: SVD spectrum + cumulative variance + noise amplification.
    Three-panel figure showing the fundamental rank structure.
    """
    U, s, Vt = svd_analysis(A, cell_types)
    # Separate effective SVs from the trivial zero mode (simplex constraint)
    s_eff = s[s > 1e-10]
    n_eff = len(s_eff)
    variance_explained = s_eff**2 / np.sum(s_eff**2)
    cumulative_var = np.cumsum(variance_explained)
    amplification = 1.0 / s_eff
    cond_eff = s_eff[0] / s_eff[-1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: singular values (log scale) — effective modes only
    ax = axes[0]
    ax.bar(range(1, n_eff + 1), s_eff, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value")
    ax.set_title(f"A. Singular value spectrum ({n_eff} effective modes)")
    ax.set_xticks(range(1, n_eff + 1))
    # annotate gap between top-2 and rest
    ax.axhline(y=s_eff[1], color="#C44E52", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(n_eff - 1, s_eff[1] * 1.3, f"σ₂ = {s_eff[1]:.3f}", color="#C44E52", fontsize=8)
    ax.text(n_eff - 1, s_eff[0] * 0.7, f"σ₁ = {s_eff[0]:.3f}", color="#4C72B0", fontsize=8)

    # Panel B: cumulative variance
    ax = axes[1]
    ax.plot(range(1, n_eff + 1), cumulative_var * 100, "o-", color="#4C72B0",
            markersize=5, linewidth=1.5)
    ax.axhline(y=95, color="#C44E52", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(n_eff - 2, 96, "95%", color="#C44E52", fontsize=8)
    ax.set_xlabel("Components retained")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("B. Cumulative variance explained")
    ax.set_xticks(range(1, n_eff + 1))
    ax.set_ylim(0, 105)
    n95 = int(np.searchsorted(cumulative_var, 0.95)) + 1
    ax.axvline(x=n95, color="gray", linestyle=":", alpha=0.4)
    ax.text(n95 + 0.2, 50, f"k={n95} for 95%", fontsize=8, color="gray")

    # Panel C: noise amplification
    ax = axes[2]
    colors = ["#4C72B0" if a < 10 else "#C44E52" for a in amplification]
    ax.bar(range(1, n_eff + 1), amplification, color=colors,
           edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Component")
    ax.set_ylabel("Amplification factor (1/σ)")
    ax.set_title("C. Noise amplification via pseudoinverse")
    ax.set_xticks(range(1, n_eff + 1))
    ax.axhline(y=10, color="#C44E52", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(1, 12, "10× amplification", color="#C44E52", fontsize=8)

    fig.suptitle(
        f"Composition matrix (A_obs) SVD — effective κ = {cond_eff:.1f}, "
        f"10th mode is zero (simplex constraint)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_svd_spectrum.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print numeric summary
    print(f"\n  Effective singular values: {np.array2string(s_eff, precision=4)}")
    print(f"  Effective condition number (κ): {cond_eff:.1f}")
    print(f"  Components with σ > 0.05: {np.sum(s_eff > 0.05)}")
    print(f"  Variance in top-2 components: {cumulative_var[1]*100:.1f}%")
    print(f"  Note: 10th SV ≈ 0 because composition rows sum to 1 (simplex constraint)")

    return s


def figure_composition_variation(A, cell_types, sample_ids):
    """
    Figure 2: Composition variation — mean, range, and SD per cell type.
    Two panels: (A) mean ± range strip, (B) SD barplot.
    """
    means = A.mean(axis=0)
    sds = A.std(axis=0)
    mins = A.min(axis=0)
    maxs = A.max(axis=0)
    ranges = maxs - mins

    # Sort by mean fraction descending
    order = np.argsort(-means)
    cell_types_sorted = [cell_types[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: mean ± full range (strip chart)
    ax = axes[0]
    y_pos = np.arange(len(cell_types_sorted))
    for i, idx in enumerate(order):
        # Plot all 24 individual points
        ax.scatter(A[:, idx], [i] * A.shape[0], s=12, alpha=0.4,
                   color="#4C72B0", zorder=2)
        # Mean marker
        ax.scatter(means[idx], i, s=60, color="#C44E52", marker="|",
                   linewidths=2, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cell_types_sorted, fontsize=9)
    ax.set_xlabel("Fraction of total cells")
    ax.set_title("A. Cell-type fractions across 24 groups\n"
                 "(dots = individual groups, red line = mean)")
    ax.set_xlim(-0.01, max(maxs) + 0.02)
    ax.invert_yaxis()

    # Panel B: SD and range barplot
    ax = axes[1]
    w = 0.35
    ax.barh(y_pos - w/2, sds[order], w, color="#4C72B0",
            label="SD across groups")
    ax.barh(y_pos + w/2, ranges[order], w, color="#C44E52",
            label="Full range (max − min)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cell_types_sorted, fontsize=9)
    ax.set_xlabel("Fraction")
    ax.set_title("B. Between-group variation per cell type")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()

    # Annotate SD values
    for i, idx in enumerate(order):
        ax.text(sds[idx] + 0.001, i - w/2, f"{sds[idx]:.3f}",
                va="center", fontsize=7, color="#4C72B0")
        ax.text(ranges[idx] + 0.001, i + w/2, f"{ranges[idx]:.3f}",
                va="center", fontsize=7, color="#C44E52")

    fig.suptitle("Composition matrix variation: cell-type fractions barely differ "
                 "across experimental groups",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_composition_variation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def figure_svd_factors(A, sample_ids):
    """
    Figure 2b: What drives the top SVD components?

    Regress SVD components 1-2 against sex, timepoint, and genotype
    to see if the composition variation is explained by experimental design.
    """
    A_centered = A - A.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(A_centered, full_matrices=False)

    # Parse sample_ids into factors
    sexes = []
    timepoints = []
    genotypes = []
    for sid in sample_ids:
        parts = sid.split("_")
        sexes.append(parts[0])        # fe or ma
        timepoints.append(parts[1])   # 2mo, 4mo, 6mo
        genotypes.append(parts[2])    # AppP, Ttau, ApTt, WTyp

    # PC scores (samples projected onto top components)
    scores = U * s  # (24, k)

    # Color maps for factors
    sex_colors = {"fe": "#E24A33", "ma": "#348ABD"}
    time_colors = {"2mo": "#FBC15E", "4mo": "#8EBA42", "6mo": "#988ED5"}
    geno_colors = {"WTyp": "#777777", "AppP": "#E24A33",
                   "Ttau": "#348ABD", "ApTt": "#988ED5"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    var_exp = s**2 / np.sum(s**2)

    for ax_idx, (factor_name, factor_vals, color_map) in enumerate([
        ("Sex", sexes, sex_colors),
        ("Timepoint", timepoints, time_colors),
        ("Genotype", genotypes, geno_colors),
    ]):
        ax = axes[ax_idx]
        for sid, fv, pc1, pc2 in zip(sample_ids, factor_vals,
                                      scores[:, 0], scores[:, 1]):
            ax.scatter(pc1, pc2, c=color_map[fv], s=50, alpha=0.7,
                       edgecolors="white", linewidth=0.5)
        # Add legend
        for label, color in color_map.items():
            ax.scatter([], [], c=color, s=40, label=label)
        ax.legend(fontsize=7, loc="best")
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
        ax.set_title(f"Colored by {factor_name}")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)

    # Compute R² for each factor via one-hot encoding
    from sklearn.linear_model import LinearRegression
    for comp_idx in range(2):
        pc = scores[:, comp_idx].reshape(-1, 1)
        for factor_name, factor_vals in [("Sex", sexes), ("Timepoint", timepoints),
                                          ("Genotype", genotypes)]:
            # One-hot encode
            unique = sorted(set(factor_vals))
            X = np.zeros((len(factor_vals), len(unique)))
            for i, v in enumerate(factor_vals):
                X[i, unique.index(v)] = 1
            model = LinearRegression(fit_intercept=True)
            model.fit(X, scores[:, comp_idx])
            r2 = model.score(X, scores[:, comp_idx])
            print(f"  PC{comp_idx+1} ~ {factor_name}: R² = {r2:.3f}")

    fig.suptitle("Top composition PCs colored by experimental factors",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2b_svd_factors.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# §2  Synthetic deconvolution experiments
# =============================================================================

def make_well_conditioned_composition(n_samples, n_types, rng):
    """
    Generate a composition matrix with near-uniform singular values.
    Uses Dirichlet with high concentration to ensure well-spread fractions,
    then adds structured variation to boost rank.
    """
    # Start with Dirichlet draws — gives valid compositions
    alpha = np.ones(n_types) * 2.0  # moderate concentration
    A = rng.dirichlet(alpha, size=n_samples)

    # Inject structured variation: each sample emphasizes a different type
    for i in range(min(n_samples, n_types)):
        A[i, i % n_types] += rng.uniform(0.1, 0.3)
    # Re-normalize rows to sum to 1
    A = A / A.sum(axis=1, keepdims=True)
    return A


def generate_ground_truth(n_sites, n_types, rng, sparsity=0.7):
    """
    Generate cell-type-specific effects (ground truth for recovery).

    Each site has effects in 1-3 cell types (sparse), rest are zero.
    Non-zero effects are drawn from a symmetric distribution.
    """
    B_true = np.zeros((n_sites, n_types))
    for i in range(n_sites):
        # How many cell types have a non-zero effect for this site
        n_active = rng.integers(1, min(4, n_types + 1))
        active = rng.choice(n_types, size=n_active, replace=False)
        signs = rng.choice([-1, 1], size=n_active)
        magnitudes = rng.uniform(EFFECT_RANGE[0], EFFECT_RANGE[1], size=n_active)
        B_true[i, active] = signs * magnitudes
    return B_true


def synthesize_bulk(A, B_true, noise_sd, rng):
    """
    Generate synthetic bulk measurements: Y = A @ B_true.T + noise.

    Y: (n_samples, n_sites) — what we'd observe in bulk.
    """
    Y = A @ B_true.T + rng.normal(0, noise_sd, size=(A.shape[0], B_true.shape[0]))
    return Y


def recover_ols(A, Y):
    """OLS recovery: B_hat = pinv(A) @ Y, transposed to (n_sites, n_types)."""
    B_hat = np.linalg.lstsq(A, Y, rcond=None)[0].T
    return B_hat


def recover_ridge(A, Y, alpha=1.0):
    """Ridge regression recovery."""
    model = Ridge(alpha=alpha, fit_intercept=False)
    # Y is (n_samples, n_sites); fit each site as a target
    model.fit(A, Y)
    B_hat = model.coef_  # (n_sites, n_types)
    return B_hat


def recover_nnls_signed(A, Y):
    """
    NNLS on positive and negative components separately, then combine.
    Handles signed effects by decomposing B = B+ - B-.
    """
    n_sites = Y.shape[1]
    n_types = A.shape[1]
    B_hat = np.zeros((n_sites, n_types))
    A_aug = np.hstack([A, A])  # doubled for pos/neg decomposition
    for j in range(n_sites):
        x, _ = nnls(A_aug, Y[:, j])
        B_hat[j, :] = x[:n_types] - x[n_types:]
    return B_hat


def site_correlations(B_true, B_hat):
    """Per-site Pearson correlation between true and recovered profiles."""
    n_sites = B_true.shape[0]
    corrs = np.zeros(n_sites)
    for i in range(n_sites):
        if np.std(B_true[i]) < 1e-10 or np.std(B_hat[i]) < 1e-10:
            corrs[i] = 0.0
        else:
            corrs[i] = np.corrcoef(B_true[i], B_hat[i])[0, 1]
    return corrs


def celltype_correlations(B_true, B_hat):
    """Per-cell-type Pearson correlation across all sites."""
    n_types = B_true.shape[1]
    corrs = np.zeros(n_types)
    for j in range(n_types):
        if np.std(B_true[:, j]) < 1e-10 or np.std(B_hat[:, j]) < 1e-10:
            corrs[j] = 0.0
        else:
            corrs[j] = np.corrcoef(B_true[:, j], B_hat[:, j])[0, 1]
    return corrs


def run_single_trial(A, rng, noise_sd=NOISE_SD):
    """Run one synthetic trial, return per-site correlations for OLS."""
    n_types = A.shape[1]
    B_true = generate_ground_truth(N_SITES, n_types, rng)
    Y = synthesize_bulk(A, B_true, noise_sd, rng)
    B_hat = recover_ols(A, Y)
    return site_correlations(B_true, B_hat)


def figure_positive_negative_control(A_obs):
    """
    Figure 3: Side-by-side recovery comparison.
    Left: well-conditioned composition (positive control).
    Right: actual A_obs composition (negative control).
    """
    rng = np.random.default_rng(RNG_SEED)
    n_types = A_obs.shape[1]
    n_samples = A_obs.shape[0]

    # Positive control: well-conditioned composition
    A_good = make_well_conditioned_composition(n_samples, n_types, rng)

    # Generate shared ground truth
    B_true = generate_ground_truth(N_SITES, n_types, rng)

    # Synthesize bulk for both compositions
    Y_good = synthesize_bulk(A_good, B_true, NOISE_SD, rng)
    Y_obs = synthesize_bulk(A_obs, B_true, NOISE_SD, rng)

    # Recover with OLS
    B_hat_good = recover_ols(A_good, Y_good)
    B_hat_obs = recover_ols(A_obs, Y_obs)

    # Per-site correlations
    corr_good = site_correlations(B_true, B_hat_good)
    corr_obs = site_correlations(B_true, B_hat_obs)

    # Per-cell-type correlations
    ct_corr_good = celltype_correlations(B_true, B_hat_good)
    ct_corr_obs = celltype_correlations(B_true, B_hat_obs)

    # Effective condition numbers (excluding simplex zero mode)
    cond_good, _ = effective_condition_number(A_good)
    cond_obs, _ = effective_condition_number(A_obs)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Top-left: scatter for well-conditioned
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(B_true.ravel(), B_hat_good.ravel(), s=1, alpha=0.1, color="#4C72B0")
    lims = [min(B_true.min(), B_hat_good.min()), max(B_true.max(), B_hat_good.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    r_all = np.corrcoef(B_true.ravel(), B_hat_good.ravel())[0, 1]
    ax.set_title(f"A. Positive control (κ = {cond_good:.0f})\nOverall r = {r_all:.3f}")
    ax.set_xlabel("True effect")
    ax.set_ylabel("Recovered effect")

    # Top-right: scatter for A_obs
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(B_true.ravel(), B_hat_obs.ravel(), s=1, alpha=0.1, color="#C44E52")
    lims_obs = [min(B_true.min(), B_hat_obs.min()), max(B_true.max(), B_hat_obs.max())]
    ax.plot([lims[0], lims[1]], [lims[0], lims[1]], "k--", linewidth=0.8, alpha=0.5)
    r_all_obs = np.corrcoef(B_true.ravel(), B_hat_obs.ravel())[0, 1]
    ax.set_title(f"B. Actual composition (κ = {cond_obs:.0f})\nOverall r = {r_all_obs:.3f}")
    ax.set_xlabel("True effect")
    ax.set_ylabel("Recovered effect")
    # Set same y-limits as positive control for fair comparison
    ax.set_xlim(lims)

    # Bottom-left: per-site correlation histograms
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-1, 1, 41)
    ax.hist(corr_good, bins=bins, alpha=0.7, color="#4C72B0",
            label=f"Well-conditioned (med={np.median(corr_good):.2f})", density=True)
    ax.hist(corr_obs, bins=bins, alpha=0.7, color="#C44E52",
            label=f"Actual A_obs (med={np.median(corr_obs):.2f})", density=True)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Per-site Pearson r (true vs recovered)")
    ax.set_ylabel("Density")
    ax.set_title("C. Per-site recovery correlation")
    ax.legend(fontsize=8)

    # Bottom-right: per-cell-type correlation comparison
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(n_types)
    w = 0.35
    ax.bar(x - w/2, ct_corr_good, w, color="#4C72B0", label="Well-conditioned")
    ax.bar(x + w/2, ct_corr_obs, w, color="#C44E52", label="Actual A_obs")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Cell type index")
    ax.set_ylabel("Pearson r (across sites)")
    ax.set_title("D. Per-cell-type recovery")
    ax.legend(fontsize=8)
    ax.set_xticks(x)

    fig.suptitle("Synthetic deconvolution: positive vs negative control",
                 fontsize=13, fontweight="bold", y=1.01)
    path = os.path.join(OUT_DIR, "fig3_positive_negative_control.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\n  Well-conditioned: overall r = {r_all:.3f}, "
          f"median site r = {np.median(corr_good):.3f}")
    print(f"  Actual A_obs:     overall r = {r_all_obs:.3f}, "
          f"median site r = {np.median(corr_obs):.3f}")

    return {
        "r_good": r_all, "r_obs": r_all_obs,
        "med_site_good": np.median(corr_good),
        "med_site_obs": np.median(corr_obs),
    }


def figure_multimethod(A_obs):
    """
    Figure 4: Multiple deconvolution methods on actual A_obs.
    Shows that OLS, Ridge (multiple alphas), and NNLS all fail.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    n_types = A_obs.shape[1]

    B_true = generate_ground_truth(N_SITES, n_types, rng)
    Y = synthesize_bulk(A_obs, B_true, NOISE_SD, rng)

    methods = {}

    # OLS
    B_ols = recover_ols(A_obs, Y)
    methods["OLS"] = B_ols

    # Ridge at several alphas
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        B_ridge = recover_ridge(A_obs, Y, alpha=alpha)
        methods[f"Ridge (α={alpha})"] = B_ridge

    # NNLS (signed)
    B_nnls = recover_nnls_signed(A_obs, Y)
    methods["NNLS (signed)"] = B_nnls

    # Compute overall and per-site correlations
    results = {}
    for name, B_hat in methods.items():
        r_overall = np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1]
        site_r = site_correlations(B_true, B_hat)
        results[name] = {"r_overall": r_overall, "med_site_r": np.median(site_r),
                         "site_r": site_r}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: overall correlation by method
    ax = axes[0]
    names = list(results.keys())
    r_vals = [results[n]["r_overall"] for n in names]
    colors = ["#C44E52"] * len(names)
    ax.barh(range(len(names)), r_vals, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Overall Pearson r (true vs recovered)")
    ax.set_title("A. Overall recovery by method")
    ax.set_xlim(-0.2, 1.0)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    # Annotate values
    for i, v in enumerate(r_vals):
        ax.text(max(v + 0.02, 0.02), i, f"{v:.3f}", va="center", fontsize=8)

    # Panel B: per-site correlation distributions
    ax = axes[1]
    bins = np.linspace(-1, 1, 41)
    for name in ["OLS", "Ridge (α=1.0)", "NNLS (signed)"]:
        ax.hist(results[name]["site_r"], bins=bins, alpha=0.5,
                label=f"{name} (med={results[name]['med_site_r']:.2f})", density=True)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Per-site Pearson r")
    ax.set_ylabel("Density")
    ax.set_title("B. Per-site recovery (method comparison)")
    ax.legend(fontsize=8)

    fig.suptitle("Method-agnostic failure: all methods on actual composition",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_multimethod_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    for name, res in results.items():
        print(f"  {name:20s}  r = {res['r_overall']:.3f}  "
              f"med_site_r = {res['med_site_r']:.3f}")

    return results


def figure_condition_sweep(A_obs):
    """
    Figure 5: Parametric sweep — interpolate between well-conditioned and
    actual composition, showing recovery degrades as condition number rises.
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    n_samples, n_types = A_obs.shape

    # Well-conditioned endpoint
    A_good = make_well_conditioned_composition(n_samples, n_types, rng)

    # Interpolation parameter: alpha=0 is A_good, alpha=1 is A_obs
    alphas = np.linspace(0, 1, 21)

    cond_numbers = []
    median_site_rs = []
    overall_rs = []

    B_true = generate_ground_truth(N_SITES, n_types, rng)

    for alpha in alphas:
        # Interpolate and re-normalize to valid compositions
        A_mix = (1 - alpha) * A_good + alpha * A_obs
        A_mix = A_mix / A_mix.sum(axis=1, keepdims=True)

        # Effective condition number (excluding simplex zero mode)
        cond, _ = effective_condition_number(A_mix)
        cond_numbers.append(cond)

        # Recovery (average over several noise draws)
        site_rs_all = []
        overall_rs_trial = []
        for trial in range(20):
            Y = synthesize_bulk(A_mix, B_true, NOISE_SD, rng)
            B_hat = recover_ols(A_mix, Y)
            sr = site_correlations(B_true, B_hat)
            site_rs_all.append(np.median(sr))
            overall_rs_trial.append(np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1])
        median_site_rs.append(np.mean(site_rs_all))
        overall_rs.append(np.mean(overall_rs_trial))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: recovery vs interpolation alpha
    ax = axes[0]
    ax.plot(alphas, overall_rs, "o-", color="#4C72B0", label="Overall r", markersize=4)
    ax.plot(alphas, median_site_rs, "s-", color="#C44E52", label="Median site r", markersize=4)
    ax.set_xlabel("Interpolation α (0 = well-conditioned, 1 = actual A_obs)")
    ax.set_ylabel("Pearson r")
    ax.set_title("A. Recovery degrades toward actual composition")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylim(-0.2, 1.05)

    # Panel B: recovery vs condition number
    ax = axes[1]
    ax.plot(cond_numbers, overall_rs, "o-", color="#4C72B0", label="Overall r", markersize=4)
    ax.plot(cond_numbers, median_site_rs, "s-", color="#C44E52", label="Median site r",
            markersize=4)
    ax.set_xlabel("Condition number (κ)")
    ax.set_xscale("log")
    ax.set_ylabel("Pearson r")
    ax.set_title("B. Recovery vs condition number")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylim(-0.2, 1.05)

    fig.suptitle("Parametric transition: well-conditioned → actual composition",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_condition_sweep.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def figure_monte_carlo_distribution(A_obs):
    """
    Figure 6: Monte Carlo — distribution of recovery across many trials
    for both compositions. Demonstrates the result is robust, not a fluke.
    """
    rng = np.random.default_rng(RNG_SEED + 3)
    n_samples, n_types = A_obs.shape
    A_good = make_well_conditioned_composition(n_samples, n_types, rng)

    overall_rs_good = []
    overall_rs_obs = []

    for trial in range(N_TRIALS):
        B_true = generate_ground_truth(N_SITES, n_types, rng)

        Y_good = synthesize_bulk(A_good, B_true, NOISE_SD, rng)
        B_hat_good = recover_ols(A_good, Y_good)
        overall_rs_good.append(np.corrcoef(B_true.ravel(), B_hat_good.ravel())[0, 1])

        Y_obs = synthesize_bulk(A_obs, B_true, NOISE_SD, rng)
        B_hat_obs = recover_ols(A_obs, Y_obs)
        overall_rs_obs.append(np.corrcoef(B_true.ravel(), B_hat_obs.ravel())[0, 1])

    overall_rs_good = np.array(overall_rs_good)
    overall_rs_obs = np.array(overall_rs_obs)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.3, 1.0, 51)
    ax.hist(overall_rs_good, bins=bins, alpha=0.7, color="#4C72B0", density=True,
            label=f"Well-conditioned (μ={overall_rs_good.mean():.3f} ± {overall_rs_good.std():.3f})")
    ax.hist(overall_rs_obs, bins=bins, alpha=0.7, color="#C44E52", density=True,
            label=f"Actual A_obs (μ={overall_rs_obs.mean():.3f} ± {overall_rs_obs.std():.3f})")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Overall Pearson r (true vs recovered)")
    ax.set_ylabel("Density")
    ax.set_title(f"Monte Carlo recovery distribution ({N_TRIALS} trials)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_monte_carlo.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\n  Well-conditioned: {overall_rs_good.mean():.3f} ± {overall_rs_good.std():.3f}")
    print(f"  Actual A_obs:     {overall_rs_obs.mean():.3f} ± {overall_rs_obs.std():.3f}")
    print(f"  Non-overlap: {np.mean(overall_rs_good > overall_rs_obs.max()):.1%} of "
          f"well-conditioned trials exceed best A_obs trial")


def figure_noise_sensitivity(A_obs):
    """
    Figure 7: Recovery as a function of noise level for both compositions.
    Even at very low noise, A_obs fails due to ill-conditioning.
    """
    rng = np.random.default_rng(RNG_SEED + 4)
    n_samples, n_types = A_obs.shape
    A_good = make_well_conditioned_composition(n_samples, n_types, rng)
    B_true = generate_ground_truth(N_SITES, n_types, rng)

    noise_levels = np.logspace(-3, 0.5, 20)  # 0.001 to ~3.2
    rs_good = []
    rs_obs = []

    for sd in noise_levels:
        rs_g = []
        rs_o = []
        for _ in range(10):
            Y_good = synthesize_bulk(A_good, B_true, sd, rng)
            Y_obs = synthesize_bulk(A_obs, B_true, sd, rng)
            B_hat_g = recover_ols(A_good, Y_good)
            B_hat_o = recover_ols(A_obs, Y_obs)
            rs_g.append(np.corrcoef(B_true.ravel(), B_hat_g.ravel())[0, 1])
            rs_o.append(np.corrcoef(B_true.ravel(), B_hat_o.ravel())[0, 1])
        rs_good.append(np.mean(rs_g))
        rs_obs.append(np.mean(rs_o))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, rs_good, "o-", color="#4C72B0", label="Well-conditioned",
            markersize=4)
    ax.plot(noise_levels, rs_obs, "s-", color="#C44E52", label="Actual A_obs",
            markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Noise σ (log₂ scale)")
    ax.set_ylabel("Overall Pearson r")
    ax.set_title("Recovery vs measurement noise")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=NOISE_SD, color="gray", linestyle="--", alpha=0.4)
    ax.text(NOISE_SD * 1.1, 0.1, f"σ = {NOISE_SD}\n(empirical)", fontsize=8, color="gray")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_noise_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# §3  Numerical summary table
# =============================================================================

def write_summary_table(A_obs, results_ctrl, results_methods):
    """Write a CSV summarizing all key numbers for the report."""
    cond_eff, s_eff = effective_condition_number(A_obs)

    rows = []
    rows.append({
        "metric": "composition_matrix_shape",
        "value": f"{A_obs.shape[0]} × {A_obs.shape[1]}",
    })
    rows.append({
        "metric": "effective_condition_number",
        "value": f"{cond_eff:.1f}",
    })
    rows.append({
        "metric": "effective_rank_gt_0.05",
        "value": str(np.sum(s_eff > 0.05)),
    })
    rows.append({
        "metric": "effective_singular_values",
        "value": ", ".join(f"{v:.4f}" for v in s_eff),
    })
    rows.append({
        "metric": "variance_top2_pct",
        "value": f"{np.sum(s_eff[:2]**2)/np.sum(s_eff**2)*100:.1f}",
    })
    if results_ctrl:
        rows.append({
            "metric": "positive_control_overall_r",
            "value": f"{results_ctrl['r_good']:.3f}",
        })
        rows.append({
            "metric": "negative_control_overall_r",
            "value": f"{results_ctrl['r_obs']:.3f}",
        })
    if results_methods:
        for name, res in results_methods.items():
            rows.append({
                "metric": f"method_{name}_overall_r",
                "value": f"{res['r_overall']:.3f}",
            })

    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "summary_table.csv")
    df.to_csv(path, index=False)
    print(f"\n  Summary table saved: {path}")


# =============================================================================
# §4  Cell-type reduction analysis
# =============================================================================

def find_best_subset(A, k):
    """
    Find the k-column subset of A that maximizes the minimum singular value.

    This selects the k cell types whose compositional profiles are most
    distinguishable across the 24 groups.  Uses brute-force search for small
    n_types (combinatorially feasible for choose(10, k)).
    """
    from itertools import combinations
    n_types = A.shape[1]
    best_min_sv = -1
    best_cols = None
    for cols in combinations(range(n_types), k):
        A_sub = A[:, cols]
        # Re-normalize rows so they sum to 1 within this subset
        row_sums = A_sub.sum(axis=1, keepdims=True)
        A_sub_norm = A_sub / row_sums
        s = np.linalg.svd(A_sub_norm - A_sub_norm.mean(0), compute_uv=False)
        s_eff = s[s > 1e-10]
        if len(s_eff) > 0 and s_eff[-1] > best_min_sv:
            best_min_sv = s_eff[-1]
            best_cols = cols
    return best_cols, best_min_sv


def figure_celltype_reduction(A_obs, cell_types):
    """
    Figure 8: Recovery as a function of the number of cell types retained.

    For each k = 2..9, select the k cell types with maximum compositional
    diversity, then run synthetic recovery.
    """
    rng = np.random.default_rng(RNG_SEED + 5)
    n_samples = A_obs.shape[0]

    ks = list(range(2, A_obs.shape[1]))  # 2 through 9
    results = []

    for k in ks:
        best_cols, best_min_sv = find_best_subset(A_obs, k)
        A_sub = A_obs[:, list(best_cols)]
        A_sub = A_sub / A_sub.sum(axis=1, keepdims=True)  # re-normalize

        cond, s_eff = effective_condition_number(A_sub)
        type_names = [cell_types[c] for c in best_cols]

        # Run recovery trials
        overall_rs = []
        site_rs_med = []
        for _ in range(50):
            B_true = generate_ground_truth(N_SITES, k, rng)
            Y = synthesize_bulk(A_sub, B_true, NOISE_SD, rng)
            B_hat = recover_ols(A_sub, Y)
            overall_rs.append(np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1])
            site_rs_med.append(np.median(site_correlations(B_true, B_hat)))

        results.append({
            "k": k,
            "cols": best_cols,
            "types": type_names,
            "cond": cond,
            "min_sv": best_min_sv,
            "r_overall": np.mean(overall_rs),
            "r_site_med": np.mean(site_rs_med),
        })

        print(f"  k={k}: κ={cond:.1f}, r={np.mean(overall_rs):.3f}, "
              f"types={type_names}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ks_plot = [r["k"] for r in results]
    rs_overall = [r["r_overall"] for r in results]
    rs_site = [r["r_site_med"] for r in results]
    conds = [r["cond"] for r in results]

    # Panel A: recovery vs k
    ax = axes[0]
    ax.plot(ks_plot, rs_overall, "o-", color="#4C72B0", label="Overall r",
            markersize=6, linewidth=1.5)
    ax.plot(ks_plot, rs_site, "s-", color="#C44E52", label="Median site r",
            markersize=6, linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Number of cell types retained")
    ax.set_ylabel("Pearson r (true vs recovered)")
    ax.set_title("A. Recovery vs cell-type count")
    ax.legend(fontsize=8)
    ax.set_xticks(ks_plot)
    ax.set_ylim(-0.1, 0.8)

    # Panel B: condition number vs k
    ax = axes[1]
    ax.plot(ks_plot, conds, "D-", color="#55A868", markersize=6, linewidth=1.5)
    ax.set_xlabel("Number of cell types retained")
    ax.set_ylabel("Effective condition number (κ)")
    ax.set_title("B. Conditioning vs cell-type count")
    ax.set_xticks(ks_plot)

    # Panel C: which types are selected
    ax = axes[2]
    # Build a presence matrix
    all_types = cell_types
    presence = np.zeros((len(results), len(all_types)))
    for i, r in enumerate(results):
        for c in r["cols"]:
            presence[i, c] = 1
    im = ax.imshow(presence.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"k={r['k']}" for r in results], fontsize=8)
    ax.set_yticks(range(len(all_types)))
    ax.set_yticklabels(all_types, fontsize=7)
    ax.set_title("C. Selected cell types per subset")
    ax.set_xlabel("Subset size")

    fig.suptitle("Cell-type reduction: selecting maximally distinguishable subsets",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig8_celltype_reduction.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Save table
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "celltype_reduction.csv"), index=False)

    return results


# =============================================================================
# §5  Sample and diversity threshold analysis
# =============================================================================

def figure_recovery_frontier(A_obs):
    """
    Figure 9: How many samples and how much diversity are needed?

    Two panels:
    A) Fix n_types=10, vary n_samples from 24 up to 200 (with A_obs composition)
    B) Fix n_samples=24, vary diversity by blending A_obs toward well-conditioned
       — show the minimum diversity needed for r > 0.3, r > 0.5
    """
    rng = np.random.default_rng(RNG_SEED + 6)
    n_types = A_obs.shape[1]

    # --- Panel A: more samples, same composition structure ---
    # Simulate adding more groups with compositions drawn from the same
    # distribution as A_obs (bootstrap resampling with small jitter)
    sample_counts = [24, 36, 48, 72, 96, 120, 160, 200, 300, 500]
    rs_by_n = []

    for n in sample_counts:
        # Resample rows from A_obs with replacement, add small jitter
        idx = rng.choice(A_obs.shape[0], size=n, replace=True)
        A_n = A_obs[idx, :].copy()
        # Add jitter proportional to observed variation
        jitter = rng.normal(0, 0.005, size=A_n.shape)
        A_n = np.clip(A_n + jitter, 1e-6, None)
        A_n = A_n / A_n.sum(axis=1, keepdims=True)

        trial_rs = []
        for _ in range(30):
            B_true = generate_ground_truth(N_SITES, n_types, rng)
            Y = synthesize_bulk(A_n, B_true, NOISE_SD, rng)
            B_hat = recover_ols(A_n, Y)
            trial_rs.append(np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1])
        rs_by_n.append(np.mean(trial_rs))

    # --- Panel B: same 24 samples, vary diversity ---
    A_good = make_well_conditioned_composition(24, n_types, rng)
    alphas = np.linspace(0, 1, 21)  # 1=A_obs (bad), 0=A_good
    # Reverse: fraction of "good" composition
    diversity_fracs = 1 - alphas  # 0=A_obs, 1=A_good
    rs_by_div = []

    for alpha in alphas:
        A_mix = (1 - alpha) * A_good + alpha * A_obs
        A_mix = A_mix / A_mix.sum(axis=1, keepdims=True)

        trial_rs = []
        for _ in range(30):
            B_true = generate_ground_truth(N_SITES, n_types, rng)
            Y = synthesize_bulk(A_mix, B_true, NOISE_SD, rng)
            B_hat = recover_ols(A_mix, Y)
            trial_rs.append(np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1])
        rs_by_div.append(np.mean(trial_rs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A
    ax = axes[0]
    ax.plot(sample_counts, rs_by_n, "o-", color="#4C72B0", markersize=5,
            linewidth=1.5)
    ax.axhline(y=0.3, color="#C44E52", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(sample_counts[-1] * 0.7, 0.32, "r = 0.3", color="#C44E52", fontsize=8)
    ax.axhline(y=0.5, color="#55A868", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(sample_counts[-1] * 0.7, 0.52, "r = 0.5", color="#55A868", fontsize=8)
    ax.axvline(x=24, color="gray", linestyle=":", alpha=0.4)
    ax.text(26, rs_by_n[0] + 0.02, "actual\n(n=24)", fontsize=7, color="gray")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Overall Pearson r")
    ax.set_title("A. More samples (same composition structure)")
    ax.set_ylim(-0.05, 0.7)

    # Panel B
    ax = axes[1]
    ax.plot(diversity_fracs, rs_by_div, "o-", color="#4C72B0", markersize=5,
            linewidth=1.5)
    ax.axhline(y=0.3, color="#C44E52", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.5, color="#55A868", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
    ax.text(0.02, 0.1, "actual\nA_obs", fontsize=7, color="gray")
    ax.set_xlabel("Compositional diversity\n(0 = actual A_obs, 1 = ideal)")
    ax.set_ylabel("Overall Pearson r")
    ax.set_title("B. More diversity (same 24 samples)")
    ax.set_ylim(-0.05, 0.7)

    fig.suptitle("Recovery thresholds: how many samples or how much diversity is needed?",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig9_recovery_frontier.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print key numbers
    print(f"\n  With A_obs-like composition:")
    for n, r in zip(sample_counts, rs_by_n):
        print(f"    n={n:4d}: r = {r:.3f}")

    # Estimate n needed for r > 0.3
    for threshold, label in [(0.3, "r > 0.3"), (0.5, "r > 0.5")]:
        above = [(n, r) for n, r in zip(sample_counts, rs_by_n) if r >= threshold]
        if above:
            print(f"  First n achieving {label}: {above[0][0]}")
        else:
            print(f"  {label} not achieved even at n={sample_counts[-1]}")

    return sample_counts, rs_by_n


# =============================================================================
# §6  Established deconvolution package equivalence
# =============================================================================

def recover_cibersort_style(A, Y):
    """
    CIBERSORTx-style constrained regression.

    CIBERSORTx uses nu-SVR (support vector regression), but its core
    mathematical operation is equivalent to constrained least squares with
    non-negativity.  We implement the closest standard equivalent:
    NNLS per-sample, which matches the CIBERSORTx "absolute mode."

    For signed effects (which deconvolution of condition effects requires),
    we use the same pos/neg decomposition as our NNLS method.
    """
    return recover_nnls_signed(A, Y)


def recover_music_style(A, Y, cell_type_variances=None):
    """
    MuSiC-style weighted least squares.

    MuSiC (Wang et al., 2019) uses cell-type-specific gene variance weights
    from scRNA-seq to down-weight noisy genes.  The core operation is
    weighted least squares: (A^T W A)^{-1} A^T W y.

    Here we simulate the effect by using inverse-variance weights derived
    from the composition matrix itself (since we don't have per-gene
    variance from scRNA-seq in this synthetic setup).
    """
    n_samples, n_types = A.shape
    if cell_type_variances is None:
        # Use per-sample composition entropy as a proxy for reliability
        # Samples with more uniform composition are less informative
        entropy = -np.sum(A * np.log(A + 1e-10), axis=1)
        cell_type_variances = entropy + 1e-6
    # W is n_samples x n_samples diagonal weight matrix
    W = np.diag(1.0 / cell_type_variances)
    # WLS: (A^T W A)^{-1} A^T W Y
    AtWA = A.T @ W @ A
    try:
        AtWA_inv = np.linalg.inv(AtWA + 1e-8 * np.eye(n_types))
        B_hat = (AtWA_inv @ A.T @ W @ Y).T
    except np.linalg.LinAlgError:
        B_hat = recover_ols(A, Y)
    return B_hat


def recover_bisque_style(A, Y):
    """
    BisqueRNA-style reference-based decomposition.

    BisqueRNA (Jew et al., 2020) uses a reference scRNA-seq profile to
    transform bulk data into a regression framework.  The mathematical
    core is OLS after reference-based transformation.  In our synthetic
    setup (where A is already the composition matrix), this reduces to
    OLS with a column-scaling step.
    """
    # Scale columns by their mean (simulates reference normalization)
    col_means = A.mean(axis=0, keepdims=True) + 1e-8
    A_scaled = A / col_means
    B_hat = recover_ols(A_scaled, Y)
    # Unscale
    B_hat = B_hat / col_means.ravel()
    return B_hat


def figure_package_equivalence(A_obs):
    """
    Figure 10: Recovery by method, mapped to published packages.

    Shows that the methods we test correspond to the mathematical cores
    of widely-used deconvolution tools.
    """
    rng = np.random.default_rng(RNG_SEED + 7)
    n_types = A_obs.shape[1]

    B_true = generate_ground_truth(N_SITES, n_types, rng)
    Y = synthesize_bulk(A_obs, B_true, NOISE_SD, rng)

    # Methods mapped to packages
    methods = {
        "OLS\n(bulk regression)": recover_ols(A_obs, Y),
        "NNLS\n(CIBERSORTx, EPIC,\nquanTIseq)": recover_cibersort_style(A_obs, Y),
        "WLS\n(MuSiC)": recover_music_style(A_obs, Y),
        "Scaled OLS\n(BisqueRNA)": recover_bisque_style(A_obs, Y),
        "Ridge α=0.1\n(Bayesian priors)": recover_ridge(A_obs, Y, alpha=0.1),
        "Ridge α=1.0\n(strong prior)": recover_ridge(A_obs, Y, alpha=1.0),
    }

    results = {}
    for name, B_hat in methods.items():
        r_overall = np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1]
        sr = site_correlations(B_true, B_hat)
        results[name] = {
            "r_overall": r_overall,
            "med_site_r": np.median(sr),
            "site_r": sr,
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: overall and per-site r by method
    ax = axes[0]
    names = list(results.keys())
    r_overall = [results[n]["r_overall"] for n in names]
    r_site = [results[n]["med_site_r"] for n in names]
    y_pos = np.arange(len(names))
    w = 0.35
    ax.barh(y_pos - w/2, r_overall, w, color="#4C72B0", label="Overall r")
    ax.barh(y_pos + w/2, r_site, w, color="#C44E52", label="Median site r")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Pearson r (true vs recovered)")
    ax.set_title("A. Recovery by method (mapped to packages)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.1, 0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    # Annotate overall r values
    for i, v in enumerate(r_overall):
        ax.text(max(v + 0.01, 0.01), i - w/2, f"{v:.3f}", va="center",
                fontsize=7, color="#4C72B0")

    # Panel B: per-site distributions for 3 representative methods
    ax = axes[1]
    bins = np.linspace(-1, 1, 41)
    rep_methods = [n for n in names if "OLS\n(bulk" in n or "NNLS" in n or "WLS" in n]
    colors = ["#4C72B0", "#C44E52", "#55A868"]
    for method, color in zip(rep_methods, colors):
        short = method.split("\n")[0]
        med = results[method]["med_site_r"]
        ax.hist(results[method]["site_r"], bins=bins, alpha=0.5,
                color=color, density=True,
                label=f"{short} (med={med:.2f})")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Per-site Pearson r")
    ax.set_ylabel("Density")
    ax.set_title("B. Per-site recovery distributions")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Published deconvolution methods: mathematical equivalents on actual composition",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig10_package_equivalence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    for name, res in results.items():
        short = name.replace("\n", " ")
        print(f"  {short:45s}  r = {res['r_overall']:.3f}  "
              f"med_site = {res['med_site_r']:.3f}")

    return results


# =============================================================================
# §7  BMIND + fine-grained cluster analysis
# =============================================================================

CLUSTERSIZE_FILE = os.path.join(
    config.REPO_ROOT, "data", "incytr_collections", "song",
    "method_records", "legacy_deconvolution_20250721", "inputs",
    "yuyu_clustersize.csv",
)

# The 10-type A_obs categories and which clusters they aggregate
# (from config.CLUSTERSIZE_POOL_MAP + the A_obs column names)
CLUSTER_TO_AOBS = {
    "Astrocytes": "Astrocytes",
    "Ptprz1-protoplasmic-astrocytes": "Astrocytes",
    "Endothelial-cell": "Endothelial cells",
    "Pericyte": "Endothelial cells",
    "Excitatory-Rorb": "Excitatory neurons",
    "Excitatory-Pyramidal": "Excitatory neurons",
    "Excitatory-Pyramidal-Satb2-Cux2": "Excitatory neurons",
    "Foxp2-Excitatory-Neurons-layers-6-and-2-3": "Excitatory neurons",
    "glutamatergic-excitatory-neurons": "Excitatory neurons",
    "Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons": "Excitatory neurons",
    "Excitatory principal neurons in the hippocampal dentate gyrus": "Excitatory neurons",
    "Excitatory-neurons": "Excitatory neurons",
    "Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin": "Excitatory neurons",
    "Inhibitory-Neurons": "Interneurons",
    "Erbb4-VIP-inhibitory-neurons": "Interneurons",
    "Erbb4-inhibitory-neurons": "Interneurons",
    "VIP-positive-interneuron": "Interneurons",
    "GABAergic inhibitory interneurons": "Interneurons",
    "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4": "Interneurons",
    "GABAergic-inhibitory-interneurons-VIP-positive": "Interneurons",
    "Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic": "Interneurons",
    "Reln-neurons": "Interneurons",
    "Basal-Ganglia-GABAergic-Neurons": "Interneurons",
    "Striatal-medium-spiny-neuron": "Medium spiny neurons",
    "Microglia": "Microglia",
    "Oligodendrocytes": "Oligodendrocytes",
    "OPC": "OPCs",
    "Cholinergic-Neurons": "Other",
    "Choroid-Plexus-Epithelial-Cells": "Other",
    "Ependymal-cell": "Other",
    "Vascular-Leptomeningeal-Cells": "Other",
}


def load_cluster_counts():
    """Load 46-cluster count matrix (clusters x samples)."""
    df = pd.read_csv(CLUSTERSIZE_FILE, index_col=0)
    # Transpose so rows = samples (24), columns = clusters (46)
    return df.T


def cluster_to_fractions(counts_df):
    """Convert counts to fractions (rows sum to 1)."""
    totals = counts_df.sum(axis=1)
    return counts_df.div(totals, axis=0)


def recover_bmind_style(A, Y):
    """
    BMIND-style Bayesian mixed-effects deconvolution.

    BMIND (Wang et al., 2024) uses a Bayesian framework where cell-type
    effects are treated as random effects with a shared prior covariance.
    The mathematical core is empirical Bayes linear mixed model:

        b ~ N(0, Sigma)
        y | b ~ N(Ab, sigma^2 I)

    The posterior mean is: b_hat = Sigma A^T (A Sigma A^T + sigma^2 I)^{-1} y

    With an uninformative prior (Sigma = tau^2 I), this simplifies to
    Ridge regression with tau^2/sigma^2 as the effective regularization.
    BMIND estimates Sigma from cross-individual variation, but in our
    synthetic framework with fixed composition, this reduces to Ridge
    with an empirically chosen penalty.
    """
    # BMIND's effective behavior: Ridge with data-driven regularization
    # We estimate sigma^2 from residuals of an initial OLS fit
    B_ols = recover_ols(A, Y)
    residuals = Y - A @ B_ols.T
    sigma2 = np.mean(residuals ** 2)
    # tau^2 estimated from variance of OLS estimates
    tau2 = max(np.var(B_ols), sigma2 * 0.1)
    alpha_eff = sigma2 / tau2
    return recover_ridge(A, Y, alpha=alpha_eff)


def figure_package_equivalence_v2(A_obs):
    """
    Figure 10 (revised): Recovery by method, now including BMIND.
    """
    rng = np.random.default_rng(RNG_SEED + 7)
    n_types = A_obs.shape[1]

    B_true = generate_ground_truth(N_SITES, n_types, rng)
    Y = synthesize_bulk(A_obs, B_true, NOISE_SD, rng)

    methods = {
        "OLS\n(bulk regression)": recover_ols(A_obs, Y),
        "NNLS\n(CIBERSORTx, EPIC,\nquanTIseq)": recover_cibersort_style(A_obs, Y),
        "WLS\n(MuSiC)": recover_music_style(A_obs, Y),
        "Scaled OLS\n(BisqueRNA)": recover_bisque_style(A_obs, Y),
        "Ridge α=0.1\n(Bayesian priors)": recover_ridge(A_obs, Y, alpha=0.1),
        "Mixed-effects\n(BMIND)": recover_bmind_style(A_obs, Y),
        "Ridge α=1.0\n(strong prior)": recover_ridge(A_obs, Y, alpha=1.0),
    }

    results = {}
    for name, B_hat in methods.items():
        r_overall = np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1]
        sr = site_correlations(B_true, B_hat)
        results[name] = {
            "r_overall": r_overall,
            "med_site_r": np.median(sr),
            "site_r": sr,
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel A: overall and per-site r by method
    ax = axes[0]
    names = list(results.keys())
    r_overall = [results[n]["r_overall"] for n in names]
    r_site = [results[n]["med_site_r"] for n in names]
    y_pos = np.arange(len(names))
    w = 0.35
    ax.barh(y_pos - w/2, r_overall, w, color="#4C72B0", label="Overall r")
    ax.barh(y_pos + w/2, r_site, w, color="#C44E52", label="Median site r")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Pearson r (true vs recovered)")
    ax.set_title("A. Recovery by method (mapped to packages)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.1, 0.5)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    for i, v in enumerate(r_overall):
        ax.text(max(v + 0.01, 0.01), i - w/2, f"{v:.3f}", va="center",
                fontsize=7, color="#4C72B0")

    # Panel B: per-site distributions
    ax = axes[1]
    bins = np.linspace(-1, 1, 41)
    rep_methods = [n for n in names if any(k in n for k in ["OLS\n(bulk", "NNLS", "WLS", "BMIND"])]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    for method, color in zip(rep_methods, colors):
        short = method.split("\n")[0]
        med = results[method]["med_site_r"]
        ax.hist(results[method]["site_r"], bins=bins, alpha=0.4,
                color=color, density=True,
                label=f"{short} (med={med:.2f})")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Per-site Pearson r")
    ax.set_ylabel("Density")
    ax.set_title("B. Per-site recovery distributions")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Published deconvolution methods: mathematical equivalents on actual composition",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig10_package_equivalence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    for name, res in results.items():
        short = name.replace("\n", " ")
        print(f"  {short:45s}  r = {res['r_overall']:.3f}  "
              f"med_site = {res['med_site_r']:.3f}")

    return results


def figure_aggregation_analysis():
    """
    Figure 11: Does finer-grained clustering improve deconvolution?

    Tests multiple aggregation strategies:
    1. Original 10-type A_obs (baseline)
    2. Full 46 clusters (maximum resolution)
    3. Split-neuronal: keep excitatory/inhibitory subtypes separate
    4. Optimal-k: for each k, find best aggregation from 46 clusters
    """
    rng = np.random.default_rng(RNG_SEED + 8)

    # Load raw cluster counts
    counts = load_cluster_counts()
    fracs_full = cluster_to_fractions(counts)
    cluster_names = list(fracs_full.columns)

    # Remove very rare clusters (< 0.5% mean fraction) — these add noise
    mean_fracs = fracs_full.mean(axis=0)
    kept = mean_fracs[mean_fracs >= 0.005].index.tolist()
    rare = mean_fracs[mean_fracs < 0.005].index.tolist()

    print(f"  Total clusters: {len(cluster_names)}")
    print(f"  Clusters with mean fraction >= 0.5%: {len(kept)}")
    print(f"  Rare clusters dropped: {len(rare)}")

    # ── Strategy 1: Original 10-type A_obs ──
    df_obs, A_obs, _ = load_composition()

    # ── Strategy 2: Full resolution (kept clusters only) ──
    fracs_kept = fracs_full[kept].copy()
    # Re-normalize
    fracs_kept = fracs_kept.div(fracs_kept.sum(axis=1), axis=0)
    A_full = fracs_kept.values

    # ── Strategy 3: Split neuronal subtypes ──
    # Keep excitatory subtypes and inhibitory subtypes separate
    # but merge other small categories as in A_obs
    split_map = {}
    for cluster in kept:
        if cluster in CLUSTER_TO_AOBS:
            parent = CLUSTER_TO_AOBS[cluster]
            if parent in ("Excitatory neurons", "Interneurons"):
                split_map[cluster] = cluster  # keep separate
            else:
                split_map[cluster] = parent
        elif cluster.startswith("cluster-"):
            split_map[cluster] = "Other/unnamed"
        else:
            split_map[cluster] = "Other/unnamed"

    split_types = sorted(set(split_map.values()))
    A_split = np.zeros((24, len(split_types)))
    for j, stype in enumerate(split_types):
        member_cols = [c for c, s in split_map.items() if s == stype]
        A_split[:, j] = fracs_full[member_cols].sum(axis=1).values
    A_split = A_split / A_split.sum(axis=1, keepdims=True)

    # ── Strategy 4: Greedy optimal grouping ──
    # Start from finest resolution, greedily merge the two most similar
    # clusters (highest correlation) until we reach target k
    # Test recovery at each k

    def greedy_merge_to_k(fracs_df, target_k):
        """Merge clusters greedily until target_k groups remain."""
        groups = {c: [c] for c in fracs_df.columns}

        while len(groups) > target_k:
            # Compute current group fractions
            group_names = sorted(groups.keys())
            group_fracs = pd.DataFrame(index=fracs_df.index)
            for gname in group_names:
                group_fracs[gname] = fracs_df[groups[gname]].sum(axis=1)

            # Find pair with highest correlation (most similar → merge)
            best_corr = -2
            best_pair = None
            for i, g1 in enumerate(group_names):
                for g2 in group_names[i+1:]:
                    corr = np.corrcoef(group_fracs[g1], group_fracs[g2])[0, 1]
                    if corr > best_corr:
                        best_corr = corr
                        best_pair = (g1, g2)

            # Merge
            g1, g2 = best_pair
            merged_name = f"{g1}+{g2}" if len(g1) < 40 else f"merged_{len(groups)}"
            groups[merged_name] = groups.pop(g1) + groups.pop(g2)

        # Build final matrix
        group_names = sorted(groups.keys())
        A_merged = np.zeros((len(fracs_df), len(group_names)))
        for j, gname in enumerate(group_names):
            A_merged[:, j] = fracs_df[groups[gname]].sum(axis=1).values
        A_merged = A_merged / A_merged.sum(axis=1, keepdims=True)
        return A_merged, group_names

    # Alternatively: greedy merge that maximizes min SV at each step
    def greedy_merge_maxsv(fracs_df, target_k):
        """Merge clusters greedily to maximize min singular value.
        Returns (A_merged, group_names, groups_dict)."""
        groups = {c: [c] for c in fracs_df.columns}

        while len(groups) > target_k:
            group_names = sorted(groups.keys())
            best_min_sv = -1
            best_pair = None

            for i, g1 in enumerate(group_names):
                for g2 in group_names[i+1:]:
                    trial = dict(groups)
                    merged = trial.pop(g1) + trial.pop(g2)
                    trial[f"{g1}+{g2}"] = merged

                    tnames = sorted(trial.keys())
                    A_trial = np.zeros((len(fracs_df), len(tnames)))
                    for j, tn in enumerate(tnames):
                        A_trial[:, j] = fracs_df[trial[tn]].sum(axis=1).values
                    A_trial = A_trial / A_trial.sum(axis=1, keepdims=True)

                    s = np.linalg.svd(A_trial - A_trial.mean(0), compute_uv=False)
                    s_eff = s[s > 1e-10]
                    if len(s_eff) > 0 and s_eff[-1] > best_min_sv:
                        best_min_sv = s_eff[-1]
                        best_pair = (g1, g2)

            g1, g2 = best_pair
            merged_name = f"{g1}+{g2}" if len(g1) + len(g2) < 50 else f"merged_{len(groups)}"
            groups[merged_name] = groups.pop(g1) + groups.pop(g2)

        group_names = sorted(groups.keys())
        A_merged = np.zeros((len(fracs_df), len(group_names)))
        for j, gname in enumerate(group_names):
            A_merged[:, j] = fracs_df[groups[gname]].sum(axis=1).values
        A_merged = A_merged / A_merged.sum(axis=1, keepdims=True)
        return A_merged, group_names, groups

    # ── Run recovery across strategies ──
    strategies = {}

    # 1. Original 10 types
    cond_10, _ = effective_condition_number(A_obs)
    strategies["Original 10 types"] = {"A": A_obs, "k": A_obs.shape[1],
                                        "cond": cond_10}

    # 2. Split neuronal (many types)
    cond_split, _ = effective_condition_number(A_split)
    strategies[f"Split neuronal ({A_split.shape[1]} types)"] = {
        "A": A_split, "k": A_split.shape[1], "cond": cond_split,
    }

    # 3. Full resolution (kept clusters)
    cond_full, _ = effective_condition_number(A_full)
    strategies[f"Full clusters ({A_full.shape[1]} types)"] = {
        "A": A_full, "k": A_full.shape[1], "cond": cond_full,
    }

    # 4. Greedy-optimal at various k
    for target_k in [5, 8, 10, 15]:
        if target_k >= len(kept):
            continue
        A_opt, opt_names, opt_groups = greedy_merge_maxsv(fracs_kept, target_k)
        cond_opt, _ = effective_condition_number(A_opt)
        strategies[f"Optimized {target_k} groups"] = {
            "A": A_opt, "k": target_k, "cond": cond_opt,
            "group_names": opt_names, "group_members": opt_groups,
        }

    # Save optimized 5-group membership table
    if "Optimized 5 groups" in strategies:
        gm = strategies["Optimized 5 groups"]["group_members"]
        rows = []
        for i, (gname, members) in enumerate(sorted(gm.items()), 1):
            mean_frac = fracs_kept[members].sum(axis=1).mean()
            rows.append({
                "group": i,
                "mean_fraction": f"{mean_frac:.3f}",
                "n_clusters": len(members),
                "clusters": ", ".join(members),
            })
        df_groups = pd.DataFrame(rows)
        path = os.path.join(OUT_DIR, "optimized_5groups.csv")
        df_groups.to_csv(path, index=False)
        print(f"\n  Optimized 5-group membership saved: {path}")
        for _, row in df_groups.iterrows():
            print(f"    Group {row['group']} ({row['mean_fraction']}, "
                  f"{row['n_clusters']} clusters): {row['clusters']}")

    # Run recovery for each strategy
    for name, strat in strategies.items():
        A = strat["A"]
        k = strat["k"]
        trial_rs = []
        site_rs = []
        for _ in range(50):
            B_true = generate_ground_truth(N_SITES, k, rng)
            Y = synthesize_bulk(A, B_true, NOISE_SD, rng)
            B_hat = recover_ols(A, Y)
            trial_rs.append(np.corrcoef(B_true.ravel(), B_hat.ravel())[0, 1])
            site_rs.append(np.median(site_correlations(B_true, B_hat)))
        strat["r_overall"] = np.mean(trial_rs)
        strat["r_site"] = np.mean(site_rs)
        print(f"  {name:35s}  k={k:2d}  κ={strat['cond']:6.1f}  "
              f"r={strat['r_overall']:.3f}  site_r={strat['r_site']:.3f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: recovery by strategy
    ax = axes[0]
    names_list = list(strategies.keys())
    rs = [strategies[n]["r_overall"] for n in names_list]
    rs_site = [strategies[n]["r_site"] for n in names_list]
    ks = [strategies[n]["k"] for n in names_list]
    y_pos = np.arange(len(names_list))
    w = 0.35
    ax.barh(y_pos - w/2, rs, w, color="#4C72B0", label="Overall r")
    ax.barh(y_pos + w/2, rs_site, w, color="#C44E52", label="Median site r")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n}\n(k={k})" for n, k in zip(names_list, ks)], fontsize=7)
    ax.set_xlabel("Pearson r (true vs recovered)")
    ax.set_title("A. Recovery by aggregation strategy")
    ax.legend(fontsize=8)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlim(-0.05, 0.7)
    for i, v in enumerate(rs):
        ax.text(v + 0.01, i - w/2, f"{v:.3f}", va="center", fontsize=7,
                color="#4C72B0")

    # Panel B: recovery vs k for all strategies
    ax = axes[1]
    for name, strat in strategies.items():
        marker = "o" if "Optimized" in name else "s"
        color = "#55A868" if "Optimized" in name else (
            "#C44E52" if "Original" in name else "#4C72B0")
        ax.plot(strat["k"], strat["r_overall"], marker, color=color,
                markersize=8, label=name)
    ax.set_xlabel("Number of cell-type groups (k)")
    ax.set_ylabel("Overall Pearson r")
    ax.set_title("B. Recovery vs number of groups")
    ax.legend(fontsize=6, loc="upper right")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)

    fig.suptitle("Effect of cell-type aggregation on deconvolution feasibility",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig11_aggregation_analysis.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    return strategies


# =============================================================================
# §8  Entry points
# =============================================================================

def step_svd():
    """§1: SVD analysis and composition heatmap."""
    print("\n=== §1: SVD Analysis ===")
    df, A, cell_types = load_composition()
    figure_svd_spectrum(A, cell_types)
    figure_composition_variation(A, cell_types, list(df.index))
    figure_svd_factors(A, list(df.index))
    return A, cell_types


def step_synthetic():
    """§2: Synthetic positive/negative control + multi-method + Monte Carlo."""
    print("\n=== §2: Synthetic Deconvolution Experiments ===")
    _, A, cell_types = load_composition()

    print("\n--- Positive vs Negative Control ---")
    results_ctrl = figure_positive_negative_control(A)

    print("\n--- Multi-Method Comparison ---")
    results_methods = figure_multimethod(A)

    print("\n--- Monte Carlo Distribution ---")
    figure_monte_carlo_distribution(A)

    print("\n--- Noise Sensitivity ---")
    figure_noise_sensitivity(A)

    return results_ctrl, results_methods


def step_sweep():
    """§3: Condition number sweep."""
    print("\n=== §3: Condition Number Sweep ===")
    _, A, cell_types = load_composition()
    figure_condition_sweep(A)


def step_reduction():
    """§4: Cell-type reduction analysis."""
    print("\n=== §4: Cell-Type Reduction ===")
    df, A, cell_types = load_composition()
    figure_celltype_reduction(A, cell_types)


def step_frontier():
    """§5: Sample and diversity threshold analysis."""
    print("\n=== §5: Recovery Frontier ===")
    _, A, cell_types = load_composition()
    figure_recovery_frontier(A)


def step_packages():
    """§6: Published package equivalence (with BMIND)."""
    print("\n=== §6: Package Equivalence ===")
    _, A, cell_types = load_composition()
    figure_package_equivalence_v2(A)


def step_aggregation():
    """§7: Aggregation analysis."""
    print("\n=== §7: Aggregation Analysis ===")
    figure_aggregation_analysis()


def run_all():
    """Run everything."""
    A, cell_types = step_svd()
    results_ctrl, results_methods = step_synthetic()
    step_sweep()
    step_reduction()
    step_frontier()
    step_packages()
    step_aggregation()

    _, A_obs, _ = load_composition()
    write_summary_table(A_obs, results_ctrl, results_methods)

    print("\n" + "=" * 60)
    print("All figures saved to:", OUT_DIR)
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deconvolution infeasibility analysis")
    parser.add_argument("--svd", action="store_true", help="SVD analysis only")
    parser.add_argument("--synthetic", action="store_true", help="Synthetic experiments only")
    parser.add_argument("--sweep", action="store_true", help="Condition sweep only")
    parser.add_argument("--reduction", action="store_true", help="Cell-type reduction only")
    parser.add_argument("--frontier", action="store_true", help="Recovery frontier only")
    parser.add_argument("--packages", action="store_true", help="Package equivalence only")
    parser.add_argument("--aggregation", action="store_true", help="Aggregation analysis only")
    parser.add_argument("--run", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    if args.run:
        run_all()
    elif args.svd:
        step_svd()
    elif args.synthetic:
        step_synthetic()
    elif args.sweep:
        step_sweep()
    elif args.reduction:
        step_reduction()
    elif args.frontier:
        step_frontier()
    elif args.packages:
        step_packages()
    elif args.aggregation:
        step_aggregation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
