#!/usr/bin/env python3
"""Module 3: Two-compartment (neuronal vs glial) deconvolution.

Collapses 5 cell types to Neuronal (Excitatory + GABAergic) vs Glial
(Oligodendrocytes + Astrocytes + Microglia + Other). Hard-gated on
synthetic validation.

See sap_rescue.md Module 3 for specification.

Usage:
    python code/sap_model_2comp.py --collapse    # diagnostics only
    python code/sap_model_2comp.py --validate    # synthetic validation (go/no-go)
    python code/sap_model_2comp.py --fit         # fit on real data (gated)
    python code/sap_model_2comp.py --summary     # print cached results
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config
import sap_data
from sap_validate import _safe_pearson, _safe_slope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join("outputs", "reports", "module3_two_compartment")
COLLAPSE_FILE = os.path.join(OUTPUT_DIR, "composition_svd.json")
VALIDATION_FILE = os.path.join(OUTPUT_DIR, "synthetic_validation.json")
BETA_FILE = os.path.join(OUTPUT_DIR, "beta_estimates.csv")
META_FILE = os.path.join(OUTPUT_DIR, "module3_summary.json")

COMPARTMENTS = ["Neuronal", "Glial"]
NEURONAL_CTS = ["Excitatory_neurons", "GABAergic_neurons"]
GLIAL_CTS = ["Oligodendrocytes", "Astrocytes", "Microglia", "Other"]

K = 2  # number of compartments
N_FACTORIAL = 3  # App, Tau, Int
N_NUISANCE = 3  # alpha_gen, alpha_time_4mo, alpha_time_6mo
N_THETA = N_NUISANCE + K * N_FACTORIAL  # 3 + 6 = 9

# IRLS settings
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-6
OUTER_MAX_ITER = 10
OUTER_TOL = 1e-5

# Synthetic validation
SYNTH_PASS_R = 0.50


# ---------------------------------------------------------------------------
# Tweedie utilities (copied from sap_model.py to avoid import coupling)
# ---------------------------------------------------------------------------


def _tw_variance(mu: np.ndarray, p: float) -> np.ndarray:
    return np.power(np.maximum(mu, 1e-10), p)


def _tw_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> np.ndarray:
    mu = np.maximum(mu, 1e-10)
    y = np.maximum(y, 0.0)
    a, b = 1.0 - p, 2.0 - p
    t1 = np.where(y > 0, np.power(y, b) / (a * b), 0.0)
    t2 = y * np.power(mu, a) / a
    t3 = np.power(mu, b) / b
    return 2.0 * (t1 - t2 + t3)


def _tw_total_deviance(y: np.ndarray, mu: np.ndarray, p: float) -> float:
    return float(np.sum(_tw_deviance(y, mu, p)))


def _tw_loglik(y: np.ndarray, mu: np.ndarray, phi: float, p: float) -> float:
    mu = np.maximum(mu, 1e-10)
    is_zero = y <= 0
    is_pos = ~is_zero
    ll = 0.0
    if np.any(is_zero):
        ll += np.sum(-np.power(mu[is_zero], 2.0 - p) / (phi * (2.0 - p)))
    if np.any(is_pos):
        y_pos, mu_pos = y[is_pos], mu[is_pos]
        d = _tw_deviance(y_pos, mu_pos, p)
        v_y = _tw_variance(y_pos, p)
        ll += np.sum(-d / (2.0 * phi) - 0.5 * np.log(2.0 * np.pi * phi * v_y))
    return float(ll)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TwoCompData:
    """Collapsed two-compartment data."""
    a_obs: np.ndarray        # (24, 2) composition fractions
    x_base: np.ndarray       # (J, 2) DESP baselines
    y: np.ndarray            # (J, 24) bulk phospho
    sample_meta: pd.DataFrame
    site_meta: pd.DataFrame
    a_bar: np.ndarray        # (2,) mean fraction per compartment
    # Cached sample indicators
    female_ind: np.ndarray   # (24,)
    time_4mo: np.ndarray     # (24,)
    time_6mo: np.ndarray     # (24,)
    fact_ind: np.ndarray     # (24, 3)


@dataclass
class SiteParams2:
    """Parameters for one site in the 2-compartment model."""
    beta: np.ndarray       # (2, 3): compartment × [App, Tau, Int]
    alpha_gen: float
    alpha_time: np.ndarray  # (2,): [4mo, 6mo]
    phi: float
    converged: bool = False


# ---------------------------------------------------------------------------
# Data collapsing
# ---------------------------------------------------------------------------


def collapse_to_two_compartments(data: sap_data.SAPData) -> TwoCompData:
    """Collapse 6 cell types to Neuronal vs Glial.

    Neuronal = Excitatory + GABAergic
    Glial = Oligodendrocytes + Astrocytes + Microglia + Other
    """
    ct_names = config.SAP_CELLTYPES  # 6 cell types in order

    neuro_idx = [ct_names.index(ct) for ct in NEURONAL_CTS]
    glia_idx = [ct_names.index(ct) for ct in GLIAL_CTS]

    # Composition: sum fractions
    a_full = data.a_obs[ct_names].values  # (24, 6)
    a_2 = np.column_stack([
        a_full[:, neuro_idx].sum(axis=1),  # Neuronal
        a_full[:, glia_idx].sum(axis=1),   # Glial
    ])  # (24, 2)

    # Baselines: least-squares collapse for exact conservation.
    # For each site j, we need x_neuro_j and x_glia_j such that for all samples:
    #   A_neuro_i * x_neuro_j + A_glia_i * x_glia_j ≈ sum_k A_{i,k} * x_base_{k,j}
    # Since A_neuro + A_glia = 1 (exactly), the system is rank-1 after centering.
    # The exact solution per site: x_comp_j = sum_{k in comp} A_{i,k} * x_base_{k,j} / A_comp_i
    # is sample-dependent. We solve via OLS: x_2 = (A_2^T A_2)^{-1} A_2^T (A_full @ x_base)
    xb_full = data.x_base.values  # (J, 6)
    J = xb_full.shape[0]

    # Target: (24, J) = A_full @ x_base^T per sample
    target = a_full @ xb_full.T  # (24, J)

    # OLS solve: x_2 = (A_2^T A_2)^{-1} A_2^T target, per site
    AtA_inv = np.linalg.inv(a_2.T @ a_2)  # (2, 2)
    xb_2 = (AtA_inv @ a_2.T @ target).T  # (J, 2)

    # Verify conservation: A_2 @ x_base_2 ≈ A_full @ x_base
    n_check = min(J, 100)
    check_idx = np.linspace(0, J - 1, n_check, dtype=int)
    max_err_all = 0.0
    for j in check_idx:
        orig = a_full @ xb_full[j]  # (24,)
        comp = a_2 @ xb_2[j]        # (24,)
        max_err_all = max(max_err_all, np.max(np.abs(orig - comp)))
    print(f"  Conservation check ({n_check} sites): max error = {max_err_all:.2e}")

    a_bar = a_2.mean(axis=0)

    # Sample indicators
    sm = data.sample_meta
    female_ind = (sm["gender"].values == "fe").astype(float)
    time_4mo = (sm["timepoint"].values == "4mo").astype(float)
    time_6mo = (sm["timepoint"].values == "6mo").astype(float)
    fact_ind = np.array([config.SAP_FACTORIAL[c] for c in sm["condition"].values], dtype=float)

    y = data.bulk_phospho.values  # (J, 24)

    return TwoCompData(
        a_obs=a_2, x_base=xb_2, y=y,
        sample_meta=sm, site_meta=data.site_meta,
        a_bar=a_bar,
        female_ind=female_ind, time_4mo=time_4mo,
        time_6mo=time_6mo, fact_ind=fact_ind,
    )


def svd_diagnostics(data2: TwoCompData) -> Dict:
    """SVD analysis of collapsed composition matrix.

    Note: since Neuronal + Glial = 1, the centered matrix is rank 1.
    We report SVD of both the raw and centered matrices.
    """
    A = data2.a_obs  # (24, 2)
    _, s_raw, _ = np.linalg.svd(A, full_matrices=False)
    A_centered = A - A.mean(axis=0)
    _, s_centered, _ = np.linalg.svd(A_centered, full_matrices=False)

    # The informative singular value is s_centered[0] — this is the
    # variation in neuronal vs glial proportions across samples.
    # Compare to the 5-CT case: the original leading SV was 1.37.
    return {
        "sv_raw": s_raw.tolist(),
        "sv_centered": s_centered.tolist(),
        "informative_sv": float(s_centered[0]),
        "mean_fractions": data2.a_bar.tolist(),
        "neuronal_range": [float(A[:, 0].min()), float(A[:, 0].max())],
        "compartments": COMPARTMENTS,
    }


# ---------------------------------------------------------------------------
# Mean model computation
# ---------------------------------------------------------------------------


def _compute_mu(
    theta: np.ndarray,  # (9,)
    data2: TwoCompData,
    x_base_j: np.ndarray,  # (2,)
) -> np.ndarray:
    """Compute mu_i = sum_k A_{i,k} * S_{k,i} for all 24 samples.

    theta layout: [alpha_gen, alpha_4mo, alpha_6mo, delta_neuro_App, delta_neuro_Tau,
                   delta_neuro_Int, delta_glia_App, delta_glia_Tau, delta_glia_Int]
    """
    n = data2.a_obs.shape[0]  # 24

    # S_{k,i} = x_base_k + alpha_gen*female + alpha_time + delta_{k,c(i)} / a_bar_k
    # In delta-space: beta_{k,c} = delta_{k,c} / a_bar_k
    alpha_gen = theta[0]
    alpha_time = theta[1:3]  # (2,)
    delta = theta[3:].reshape(K, N_FACTORIAL)  # (2, 3)

    # Nuisance contribution (shared across compartments)
    nuisance = (alpha_gen * data2.female_ind
                + alpha_time[0] * data2.time_4mo
                + alpha_time[1] * data2.time_6mo)  # (24,)

    # S matrix: (2, 24)
    S = np.zeros((K, n))
    for k in range(K):
        S[k] = x_base_j[k] + nuisance
        # Factorial contribution: delta_{k,c} / a_bar_k * factorial_indicators
        if data2.a_bar[k] > 1e-10:
            beta_k = delta[k] / data2.a_bar[k]  # (3,)
            S[k] += data2.fact_ind @ beta_k  # (24,)

    # Non-negativity
    np.maximum(S, 0.0, out=S)

    # mu_i = sum_k A_{i,k} * S_{k,i}
    mu = np.sum(data2.a_obs.T * S, axis=0)  # (24,)
    return np.maximum(mu, 1e-10)


def _compute_jacobian(
    theta: np.ndarray,
    data2: TwoCompData,
    x_base_j: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mu and Jacobian dmu/dtheta (24 × 9).

    Active-set: if S_{k,i} = 0 (clamped), dS_{k,i}/dtheta = 0.
    """
    n = data2.a_obs.shape[0]
    alpha_gen = theta[0]
    alpha_time = theta[1:3]
    delta = theta[3:].reshape(K, N_FACTORIAL)

    nuisance = (alpha_gen * data2.female_ind
                + alpha_time[0] * data2.time_4mo
                + alpha_time[1] * data2.time_6mo)

    S = np.zeros((K, n))
    for k in range(K):
        S[k] = x_base_j[k] + nuisance
        if data2.a_bar[k] > 1e-10:
            beta_k = delta[k] / data2.a_bar[k]
            S[k] += data2.fact_ind @ beta_k

    active = S > 0  # (2, 24)
    np.maximum(S, 0.0, out=S)
    mu = np.sum(data2.a_obs.T * S, axis=0)
    mu = np.maximum(mu, 1e-10)

    # Jacobian (24 × 9)
    A_active = data2.a_obs * active.T  # (24, 2): A_{i,k} if active
    a_sum = A_active.sum(axis=1)  # (24,)

    J = np.zeros((n, N_THETA))
    J[:, 0] = a_sum * data2.female_ind
    J[:, 1] = a_sum * data2.time_4mo
    J[:, 2] = a_sum * data2.time_6mo

    for k in range(K):
        if data2.a_bar[k] > 1e-10:
            a_ratio = A_active[:, k] / data2.a_bar[k]
        else:
            a_ratio = A_active[:, k]
        J[:, N_NUISANCE + k * N_FACTORIAL: N_NUISANCE + (k + 1) * N_FACTORIAL] = (
            a_ratio[:, np.newaxis] * data2.fact_ind
        )

    return mu, J


# ---------------------------------------------------------------------------
# IRLS fitting for one site
# ---------------------------------------------------------------------------


def fit_site_2comp(
    j: int,
    data2: TwoCompData,
    p: float,
    lambda_ridge: float,
) -> SiteParams2:
    """Fit 2-compartment Hurdle-Tweedie for one site via penalized IRLS."""
    y = data2.y[j]  # (24,)
    x_base_j = data2.x_base[j]  # (2,)
    pos_mask = y > 0

    if not np.any(pos_mask):
        return SiteParams2(
            beta=np.zeros((K, N_FACTORIAL)),
            alpha_gen=0.0, alpha_time=np.zeros(2),
            phi=1.0, converged=True,
        )

    theta = np.zeros(N_THETA)

    # Initial phi from method-of-moments
    y_pos = y[pos_mask]
    mu_init = max(y_pos.mean(), 1e-10)
    var_est = max(y_pos.var(), 1e-10)
    phi = max(var_est / (mu_init ** p), 1e-6)

    prev_dev = np.inf
    converged = False

    for outer_iter in range(OUTER_MAX_ITER):
        # IRLS inner loop
        prev_dev_inner = np.inf
        for irls_iter in range(IRLS_MAX_ITER):
            mu, J = _compute_jacobian(theta, data2, x_base_j)

            w = np.power(mu, 2.0 - p) / phi
            w[~pos_mask] = 0.0

            score = np.where(pos_mask, (y - mu) / (phi * np.power(mu, p - 1.0)), 0.0)
            grad = -J.T @ score  # (9,)

            # Ridge penalty gradient on delta terms only
            ridge_grad = np.zeros(N_THETA)
            ridge_grad[N_NUISANCE:] = lambda_ridge * theta[N_NUISANCE:]
            grad += ridge_grad

            JtWJ = J.T @ (w[:, np.newaxis] * J)
            # Ridge regularization on delta terms
            ridge_diag = np.zeros(N_THETA)
            ridge_diag[N_NUISANCE:] = lambda_ridge
            JtWJ += np.diag(ridge_diag) + 1e-8 * np.eye(N_THETA)

            try:
                step = np.linalg.solve(JtWJ, -grad)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(JtWJ, grad, rcond=None)[0]

            # Backtracking line search
            alpha_ls = 1.0
            for _ in range(10):
                theta_new = theta + alpha_ls * step
                mu_new = _compute_mu(theta_new, data2, x_base_j)
                dev_new = _tw_total_deviance(y[pos_mask], mu_new[pos_mask], p)
                if dev_new < prev_dev_inner or alpha_ls < 1e-4:
                    break
                alpha_ls *= 0.5

            theta = theta_new
            mu_new = _compute_mu(theta, data2, x_base_j)
            dev = _tw_total_deviance(y[pos_mask], mu_new[pos_mask], p)

            if prev_dev_inner < np.inf:
                rel_change = abs(prev_dev_inner - dev) / (abs(prev_dev_inner) + 1e-10)
                if rel_change < IRLS_TOL:
                    break
            prev_dev_inner = dev

        # Update phi (Pearson estimator)
        mu_curr = _compute_mu(theta, data2, x_base_j)
        n_active = int(np.sum(np.abs(theta[N_NUISANCE:]) > 1e-10))
        n_eff_params = N_NUISANCE + n_active
        pearson_X2 = np.sum(np.where(
            pos_mask,
            (y - mu_curr) ** 2 / _tw_variance(mu_curr, p),
            0.0,
        ))
        df_resid = max(pos_mask.sum() - n_eff_params, 1)
        phi = max(pearson_X2 / df_resid, 1e-6)

        # Check outer convergence
        dev = _tw_total_deviance(y[pos_mask], mu_curr[pos_mask], p)
        if prev_dev < np.inf:
            rel_change = abs(prev_dev - dev) / (abs(prev_dev) + 1e-10)
            if rel_change < OUTER_TOL:
                converged = True
                break
        prev_dev = dev

    # Convert theta to params
    delta = theta[N_NUISANCE:].reshape(K, N_FACTORIAL)
    beta = np.zeros((K, N_FACTORIAL))
    for k in range(K):
        if data2.a_bar[k] > 1e-10:
            beta[k] = delta[k] / data2.a_bar[k]

    return SiteParams2(
        beta=beta,
        alpha_gen=theta[0],
        alpha_time=theta[1:3].copy(),
        phi=phi,
        converged=converged,
    )


# Worker for parallel fitting
def _fit_site_worker(args):
    j, data2_dict, p, lam = args
    data2 = TwoCompData(**data2_dict)
    return j, fit_site_2comp(j, data2, p, lam)


def fit_all_sites(
    data2: TwoCompData,
    p: float = 1.2,
    lambda_ridge: float = 1.0,
    n_workers: int = 1,
    site_subset: Optional[np.ndarray] = None,
) -> List[SiteParams2]:
    """Fit all sites (or a subset). Returns list of SiteParams2."""
    J = data2.y.shape[0]
    indices = site_subset if site_subset is not None else np.arange(J)

    print(f"\nFitting {len(indices)} sites (p={p}, lambda={lambda_ridge})...")
    t0 = time.time()

    results = [None] * J
    for idx_pos, j in enumerate(indices):
        sp = fit_site_2comp(int(j), data2, p, lambda_ridge)
        results[j] = sp
        if (idx_pos + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (idx_pos + 1) / elapsed
            print(f"  {idx_pos + 1}/{len(indices)} sites ({rate:.0f} sites/s)")

    elapsed = time.time() - t0
    n_conv = sum(1 for r in results if r is not None and r.converged)
    n_fit = sum(1 for r in results if r is not None)
    print(f"  Done: {n_fit} sites in {elapsed:.1f}s, {n_conv}/{n_fit} converged")

    return results


# ---------------------------------------------------------------------------
# Synthetic validation
# ---------------------------------------------------------------------------


def generate_delta_true(
    data2: TwoCompData,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic effects for 2 compartments.

    Returns: (2, J, 3) array of true delta values.
    """
    J = data2.y.shape[0]
    rng = np.random.RandomState(seed)
    delta_true = np.zeros((K, J, N_FACTORIAL))

    # Effect magnitude: calibrate to ~2x the noise floor
    # Use App and Tau effects only (no interaction)
    n_active = int(0.10 * J)  # 10% of sites
    active_sites = rng.choice(J, n_active, replace=False)
    signs = rng.choice([-1, 1], size=n_active)

    # Assign to compartments: some neuronal-only, some glial-only, some both
    for i, j in enumerate(active_sites):
        # Magnitude scaled by 1/a_bar for each compartment
        mag = 0.5  # modest effect
        pattern = rng.choice(3)  # 0=neuro only, 1=glia only, 2=both
        if pattern == 0:
            delta_true[0, j, 0] = signs[i] * mag  # App effect in neuronal
        elif pattern == 1:
            delta_true[1, j, 0] = signs[i] * mag  # App effect in glial
        else:
            delta_true[0, j, 0] = signs[i] * mag
            delta_true[1, j, 0] = signs[i] * mag * 0.5  # weaker in glial

    return delta_true


def _generate_synthetic_bulk(
    data2: TwoCompData,
    delta_true: np.ndarray,  # (2, J, 3)
    seed: int = 42,
) -> np.ndarray:
    """Permuted-residual injection for 2-compartment model.

    Returns: (J, 24) synthetic bulk data.
    """
    J, n = data2.y.shape
    rng = np.random.RandomState(seed)

    # Null model prediction: mu_null_i = A_2 @ x_base
    mu_null = data2.a_obs @ data2.x_base.T  # (24, J) → transpose needed
    mu_null = mu_null.T  # (J, 24)

    # Residuals from null model
    residuals = data2.y - mu_null

    # Permute residuals per site (destroy condition structure)
    for j in range(J):
        valid = ~np.isnan(data2.y[j])
        if valid.sum() > 1:
            perm_idx = rng.permutation(np.where(valid)[0])
            residuals[j, valid] = residuals[j, perm_idx]

    # Synthetic signal
    y_syn = np.zeros((J, n))
    for j in range(J):
        xb = data2.x_base[j]
        for i in range(n):
            S = np.zeros(K)
            for k in range(K):
                S[k] = xb[k]
                if data2.a_bar[k] > 1e-10:
                    beta_true = delta_true[k, j] / data2.a_bar[k]
                    S[k] += data2.fact_ind[i] @ beta_true
            S = np.maximum(S, 0.0)
            y_syn[j, i] = data2.a_obs[i] @ S

    # Add permuted residuals
    y_syn += residuals
    y_syn = np.maximum(y_syn, 0.0)

    # Preserve NaN mask
    y_syn[np.isnan(data2.y)] = 0.0

    return y_syn


def validate_synthetic(
    data2: TwoCompData,
    p: float = 1.2,
    lambda_ridge: float = 1.0,
    seed: int = 42,
) -> Dict:
    """Synthetic validation: inject known effects, refit, measure recovery."""
    print("\n" + "=" * 60)
    print("Synthetic Validation: Two-Compartment Model")
    print("=" * 60)

    # Generate true effects
    delta_true = generate_delta_true(data2, seed=seed)
    n_active = int(np.any(delta_true != 0, axis=(0, 2)).sum())
    print(f"  Injected effects: {n_active} active sites")

    # Generate synthetic bulk
    print("  Generating synthetic bulk data...")
    y_syn = _generate_synthetic_bulk(data2, delta_true, seed=seed)

    # Create synthetic data object
    data2_syn = TwoCompData(
        a_obs=data2.a_obs, x_base=data2.x_base, y=y_syn,
        sample_meta=data2.sample_meta, site_meta=data2.site_meta,
        a_bar=data2.a_bar,
        female_ind=data2.female_ind, time_4mo=data2.time_4mo,
        time_6mo=data2.time_6mo, fact_ind=data2.fact_ind,
    )

    # Refit on synthetic data (subsample for speed)
    J = data2.y.shape[0]
    n_assess = min(J, 3000)
    rng = np.random.RandomState(seed + 1)
    assess_idx = rng.choice(J, n_assess, replace=False)

    print(f"  Refitting on synthetic data ({n_assess} sites)...")
    results = fit_all_sites(data2_syn, p=p, lambda_ridge=lambda_ridge,
                            site_subset=assess_idx)

    # Measure recovery per compartment
    print("\n  Recovery metrics:")
    recovery = {}
    for k, comp_name in enumerate(COMPARTMENTS):
        d_hat = np.array([
            results[j].beta[k] if results[j] is not None else np.zeros(N_FACTORIAL)
            for j in assess_idx
        ])  # (n_assess, 3)
        d_true = delta_true[k, assess_idx, :]  # (n_assess, 3)

        # Convert back from beta to delta for comparison
        # delta_hat = a_bar_k * beta_hat_k
        d_hat_delta = d_hat * data2.a_bar[k]

        r = _safe_pearson(d_hat_delta.flatten(), d_true.flatten())
        slope = _safe_slope(d_hat_delta.flatten(), d_true.flatten())
        r_app = _safe_pearson(d_hat_delta[:, 0], d_true[:, 0])

        recovery[comp_name] = {
            "r": float(r),
            "slope": float(slope),
            "r_App": float(r_app),
            "pass": r > SYNTH_PASS_R,
        }
        pass_str = "PASS" if r > SYNTH_PASS_R else "FAIL"
        print(f"    {comp_name:12s}  r={r:+.4f}  slope={slope:.3f}  r_App={r_app:+.4f}  [{pass_str}]")

    overall_pass = all(v["pass"] for v in recovery.values())
    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'} (criterion: r > {SYNTH_PASS_R} for both)")

    return {
        "recovery": recovery,
        "overall_pass": overall_pass,
        "n_sites_assessed": n_assess,
        "n_active_sites": n_active,
        "p": p,
        "lambda_ridge": lambda_ridge,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_collapse_diagnostics() -> None:
    """Run composition collapse and SVD diagnostics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    data, _ = sap_data.load_all(include_rna=False)

    print("\nCollapsing to two compartments...")
    data2 = collapse_to_two_compartments(data)

    print("\nSVD diagnostics:")
    svd = svd_diagnostics(data2)
    print(f"  Raw singular values: {svd['sv_raw']}")
    print(f"  Centered singular values: {svd['sv_centered']}")
    print(f"  Informative SV (composition variation): {svd['informative_sv']:.4f}")
    print(f"  Mean fractions: Neuronal={svd['mean_fractions'][0]:.3f}, "
          f"Glial={svd['mean_fractions'][1]:.3f}")
    print(f"  Neuronal range: [{svd['neuronal_range'][0]:.3f}, {svd['neuronal_range'][1]:.3f}]")

    with open(COLLAPSE_FILE, "w") as f:
        json.dump(svd, f, indent=2)
    print(f"\n  Saved: {COLLAPSE_FILE}")


def run_validation() -> None:
    """Run synthetic validation (go/no-go gate)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    data, _ = sap_data.load_all(include_rna=False)

    print("\nCollapsing to two compartments...")
    data2 = collapse_to_two_compartments(data)

    svd = svd_diagnostics(data2)
    print(f"  SVD: informative SV = {svd['informative_sv']:.4f}, "
          f"fractions = {svd['mean_fractions']}")

    result = validate_synthetic(data2)

    with open(VALIDATION_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {VALIDATION_FILE}")


def run_fit() -> None:
    """Fit on real data (gated on validation passing)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check validation gate
    if os.path.exists(VALIDATION_FILE):
        with open(VALIDATION_FILE) as f:
            val = json.load(f)
        if not val.get("overall_pass", False):
            print("GATE FAILED: Synthetic validation did not pass.")
            print("Two-compartment model is not identifiable. Module 3 abandoned.")
            return
    else:
        print("No validation results found. Run --validate first.")
        return

    print("Loading data...")
    data, _ = sap_data.load_all(include_rna=False)

    print("\nCollapsing to two compartments...")
    data2 = collapse_to_two_compartments(data)

    results = fit_all_sites(data2, p=1.2, lambda_ridge=1.0)

    # Save beta estimates
    rows = []
    for j, sp in enumerate(results):
        if sp is None:
            continue
        site_id = data2.site_meta.iloc[j].get("site_id", f"site_{j}")
        for k, comp in enumerate(COMPARTMENTS):
            rows.append({
                "site_id": site_id,
                "site_idx": j,
                "compartment": comp,
                "beta_App": float(sp.beta[k, 0]),
                "beta_Tau": float(sp.beta[k, 1]),
                "beta_Int": float(sp.beta[k, 2]),
                "converged": sp.converged,
            })

    df = pd.DataFrame(rows)
    df.to_csv(BETA_FILE, index=False)
    print(f"\n  Saved: {BETA_FILE} ({len(df)} rows)")


def print_summary() -> None:
    """Print cached results."""
    for f in [COLLAPSE_FILE, VALIDATION_FILE]:
        if os.path.exists(f):
            print(f"\n--- {os.path.basename(f)} ---")
            with open(f) as fh:
                print(json.dumps(json.load(fh), indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Module 3: Two-compartment deconvolution")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collapse", action="store_true", help="Composition collapse diagnostics")
    group.add_argument("--validate", action="store_true", help="Synthetic validation (go/no-go)")
    group.add_argument("--fit", action="store_true", help="Fit real data (gated)")
    group.add_argument("--summary", action="store_true", help="Print cached results")
    args = parser.parse_args()

    if args.collapse:
        run_collapse_diagnostics()
    elif args.validate:
        run_validation()
    elif args.fit:
        run_fit()
    elif args.summary:
        print_summary()


if __name__ == "__main__":
    main()
