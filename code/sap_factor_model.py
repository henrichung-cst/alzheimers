"""
Joint Kinase-Activity Factor Model (sap_extension.md §3).

Replaces per-site condition effects β_{k,j,c} with kinase-level activity
parameters θ_{m,k,c}, sharing information across all substrates of each
kinase via the kinase-library PSSM scoring matrix W.

Two-block coordinate descent:
  Block A: Per-site nuisance (α_gen, α_time, φ, γ₀) with fixed θ
  Block B: Global θ update via Ridge-regularized Newton

Usage:
    python code/sap_factor_model.py --smoke-test     # gradient verification (3 sites)
    python code/sap_factor_model.py --fit             # full model fit
    python code/sap_factor_model.py --validate        # kinase-level synthetic validation
    python code/sap_factor_model.py --summary         # print cached results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import sap_data
import sap_model
from sap_model import (
    SampleArrays,
    tweedie_deviance,
    tweedie_log_likelihood,
    tweedie_variance,
    fit_hurdle_logistic,
    update_phi,
    compute_neff,
)
from sap_preflight import build_W_matrix, build_R_matrix, threshold_W
from analysis_utils import get_expression_cache


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
FACTOR_MODEL_DIR = os.path.join("outputs", "reports", "factor_model")
FACTOR_MODEL_FILE = os.path.join(
    config.SONG_ANALYSIS_CACHE_DIR, "sap_factor_model_fit.npz",
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
W_PERCENTILE = 25                # best from pre-flight §4.2
RIDGE_LAMBDA_DEFAULT = 1.0       # base Ridge penalty on θ
P_INIT = 1.2                     # initial Tweedie power (from production)
MAX_OUTER_ITER = 30              # two-block outer iterations
OUTER_TOL = 1e-4                 # relative change in objective
NUISANCE_MAX_ITER = 10           # Newton steps for per-site nuisance
NUISANCE_TOL = 1e-6
N_FACTORIAL = 2                  # App, Tau (no interaction)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorModelParams:
    """Parameters of the joint kinase-activity factor model."""
    theta: np.ndarray           # (P,) where P = 2 * n_active_pairs
    alpha_gen: np.ndarray       # (J,)
    alpha_time: np.ndarray      # (J, 2)
    phi: np.ndarray             # (J,)
    gamma0: np.ndarray          # (J,)
    gamma1_by_stratum: np.ndarray  # (Q,) pooled hurdle slopes


@dataclass
class FactorModelFit:
    """Complete fitted factor model."""
    params: FactorModelParams
    p: float
    active_map: List[Tuple[int, int]]  # (kinase_idx, celltype_idx) per active pair
    kinase_names: List[str]
    ridge_lambda: float
    converged: bool
    convergence_history: List[float]
    n_outer_iter: int
    wall_time: float


# ---------------------------------------------------------------------------
# Core: Δ from θ
# ---------------------------------------------------------------------------

def compute_delta_from_theta(
    theta: np.ndarray,
    W: np.ndarray,
    active_map: List[Tuple[int, int]],
    n_celltypes: int = 5,
) -> np.ndarray:
    """Compute per-site condition contributions from kinase activities.

    Δ_{k,c,j} = Σ_{m: active(m,k)} W[m,j] · θ_{idx(m,k), c}

    Args:
        theta: (P,) flattened — P = n_active_pairs * N_FACTORIAL
        W: (M, J) kinase-substrate scoring matrix
        active_map: list of (kinase_idx, celltype_idx) tuples

    Returns:
        delta: (K, J, N_FACTORIAL) condition contributions
    """
    J = W.shape[1]
    n_active = len(active_map)
    theta_2d = theta.reshape(n_active, N_FACTORIAL)  # (n_active, 2)
    delta = np.zeros((n_celltypes, J, N_FACTORIAL))

    for a_idx, (m_idx, k_idx) in enumerate(active_map):
        # W[m_idx, :] is (J,), theta_2d[a_idx, :] is (2,)
        # Contribution: outer product W[m,:] × θ[a,:] added to delta[k,:,:]
        delta[k_idx] += np.outer(W[m_idx], theta_2d[a_idx])

    return delta


# ---------------------------------------------------------------------------
# Core: μ from nuisance + Δ
# ---------------------------------------------------------------------------

def compute_mu_all(
    alpha_gen: np.ndarray,
    alpha_time: np.ndarray,
    delta: np.ndarray,
    x_base: np.ndarray,
    sa: SampleArrays,
) -> np.ndarray:
    """Compute bulk predicted intensities for all sites.

    S_{k,c(i),j} = X^base_{k,j} + α_gen_j·female_i + α_time_j·time_i + Δ_{k,c(i),j}
    μ_{i,j} = Σ_k A_{i,k} · max(S_{k,c(i),j}, 0)

    Args:
        alpha_gen: (J,)
        alpha_time: (J, 2)
        delta: (K, J, 2) — condition contributions
        x_base: (J, 6) — DESP baselines
        sa: SampleArrays

    Returns:
        mu: (J, N) where N = n_samples
    """
    J = x_base.shape[0]
    N = sa.A.shape[0]  # 24
    K = 5  # estimated cell types
    fact_ind = sa.fact_indicators[:, :N_FACTORIAL]  # (N, 2) App and Tau only

    mu = np.zeros((J, N))

    # Vectorize: pre-compute global covariate adjustment (J, N)
    # global_adj[j, i] = alpha_gen[j]*female[i] + alpha_time[j,0]*4mo[i] + alpha_time[j,1]*6mo[i]
    global_adj = (
        np.outer(alpha_gen, sa.female_ind)
        + np.outer(alpha_time[:, 0], sa.time_4mo)
        + np.outer(alpha_time[:, 1], sa.time_6mo)
    )  # (J, N)

    # For each cell type k, compute S_{k,i,j} and accumulate into μ
    x_base_vals = x_base.values if hasattr(x_base, 'values') else x_base

    for k in range(6):
        # Base signal for cell type k: (J,) broadcast to (J, N)
        S_k = x_base_vals[:, k][:, np.newaxis] + global_adj  # (J, N)

        if k < K:
            # Add condition contribution: delta[k, :, :] @ fact_ind.T → (J, N)
            S_k = S_k + delta[k] @ fact_ind.T

        # Non-negativity
        np.maximum(S_k, 0.0, out=S_k)

        # Accumulate: μ_{i,j} += A_{i,k} * S_{k,i,j}
        mu += S_k * sa.A[:, k]  # broadcast: (J, N) * (N,)

    return mu


def compute_mu_site(
    j: int,
    alpha_gen_j: float,
    alpha_time_j: np.ndarray,
    delta_j: np.ndarray,
    x_base_j: np.ndarray,
    sa: SampleArrays,
) -> np.ndarray:
    """Compute μ for a single site. Returns (N,)."""
    N = sa.A.shape[0]
    fact_ind = sa.fact_indicators[:, :N_FACTORIAL]

    global_adj = (
        alpha_gen_j * sa.female_ind
        + alpha_time_j[0] * sa.time_4mo
        + alpha_time_j[1] * sa.time_6mo
    )  # (N,)

    mu = np.zeros(N)
    for k in range(6):
        S_k = x_base_j[k] + global_adj  # (N,)
        if k < 5:
            S_k = S_k + fact_ind @ delta_j[k]  # delta_j[k] is (2,)
        S_k = np.maximum(S_k, 0.0)
        mu += sa.A[:, k] * S_k

    return mu


# ---------------------------------------------------------------------------
# Block A: Per-site nuisance fitting
# ---------------------------------------------------------------------------

def fit_site_nuisance(
    j: int,
    y_j: np.ndarray,
    delta_j: np.ndarray,
    x_base_j: np.ndarray,
    sa: SampleArrays,
    p: float,
    alpha_gen_init: float,
    alpha_time_init: np.ndarray,
    phi_init: float,
    gamma0_init: float,
    gamma1_fixed: float,
) -> Tuple[float, np.ndarray, float, float]:
    """Fit per-site nuisance parameters with condition contribution fixed.

    Newton-Raphson for (α_gen, α_time[0], α_time[1]) — 3 parameters.
    Then update φ and γ₀.

    Returns: (alpha_gen, alpha_time, phi, gamma0)
    """
    N = len(y_j)
    pos_mask = y_j > 0
    n_pos = int(pos_mask.sum())

    alpha_gen = alpha_gen_init
    alpha_time = alpha_time_init.copy()
    phi = phi_init

    fact_ind = sa.fact_indicators[:, :N_FACTORIAL]

    for _ in range(NUISANCE_MAX_ITER):
        # Compute μ
        mu = compute_mu_site(j, alpha_gen, alpha_time, delta_j, x_base_j, sa)
        mu = np.maximum(mu, 1e-10)

        if n_pos == 0:
            break

        # Tweedie score: (y - μ) / (φ · μ^{p-1})
        score = np.where(pos_mask, (y_j - mu) / (phi * np.power(mu, p - 1)), 0.0)

        # Fisher weights: μ^{2-p} / φ
        w = np.where(pos_mask, np.power(mu, 2.0 - p) / phi, 0.0)

        # Jacobian dμ/dθ_nuisance for θ = [α_gen, α_time_4mo, α_time_6mo]
        # dμ/dα_gen = Σ_k A_{i,k} * female_i * I[S_{k,i}>0]
        # We approximate by ignoring active-set (all S > 0)
        # since nuisance effects are small relative to baselines
        J_nuis = np.column_stack([
            sa.A.sum(axis=1) * sa.female_ind,
            sa.A.sum(axis=1) * sa.time_4mo,
            sa.A.sum(axis=1) * sa.time_6mo,
        ])  # (N, 3)

        grad = -J_nuis.T @ score  # (3,)
        H = J_nuis.T @ (w[:, np.newaxis] * J_nuis)  # (3, 3)
        H += 1e-8 * np.eye(3)

        try:
            step = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break

        # Damped step
        step_size = 1.0
        for _ in range(5):
            alpha_gen_new = alpha_gen + step_size * step[0]
            alpha_time_new = alpha_time + step_size * step[1:]
            mu_new = compute_mu_site(
                j, alpha_gen_new, alpha_time_new, delta_j, x_base_j, sa,
            )
            mu_new = np.maximum(mu_new, 1e-10)
            dev_new = tweedie_deviance(y_j[pos_mask], mu_new[pos_mask], p).sum()
            dev_old = tweedie_deviance(y_j[pos_mask], mu[pos_mask], p).sum()
            if dev_new < dev_old + 1e-10:
                alpha_gen = alpha_gen_new
                alpha_time = alpha_time_new
                break
            step_size *= 0.5

        if np.linalg.norm(step * step_size) < NUISANCE_TOL:
            break

    # Update phi
    mu_final = compute_mu_site(j, alpha_gen, alpha_time, delta_j, x_base_j, sa)
    mu_final = np.maximum(mu_final, 1e-10)
    phi = update_phi(y_j, mu_final, p, n_eff_params=3.0)

    # Update gamma0 (given gamma1 fixed by stratum)
    gamma0, _ = fit_hurdle_logistic(y_j, mu_final)

    return float(alpha_gen), alpha_time, float(phi), float(gamma0)


# ---------------------------------------------------------------------------
# Block B: Global θ update
# ---------------------------------------------------------------------------

def compute_theta_gradient_and_hessian(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    p: float,
    W: np.ndarray,
    active_map: List[Tuple[int, int]],
    sa: SampleArrays,
    ridge_lambda: float,
    theta: np.ndarray,
    delta: np.ndarray,
    x_base_vals: np.ndarray,
    alpha_gen: np.ndarray,
    alpha_time: np.ndarray,
    ridge_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gradient and Fisher information for θ across all sites.

    Accounts for non-negativity active set: design columns are zeroed
    for (cell_type, sample) pairs where S_{k,c(i),j} ≤ 0.

    Args:
        y: (J, N) bulk intensities
        mu: (J, N) predicted intensities
        phi: (J,) per-site dispersion
        delta: (K, J, 2) condition contributions
        x_base_vals: (J, 6) DESP baselines as numpy array
        alpha_gen, alpha_time: per-site nuisance params
        ridge_scale: (P,) optional per-parameter Ridge scaling
    """
    J, N = y.shape
    P = len(active_map) * N_FACTORIAL
    fact_ind = sa.fact_indicators[:, :N_FACTORIAL]  # (N, 2)

    grad = np.zeros(P)
    hess = np.zeros((P, P))

    mu_safe = np.maximum(mu, 1e-10)

    for j in range(J):
        pos = y[j] > 0
        if not np.any(pos):
            continue

        # Compute S for active-set determination
        # S_{k,i,j} = x_base + global_adj + delta_contribution
        global_adj = (
            alpha_gen[j] * sa.female_ind
            + alpha_time[j, 0] * sa.time_4mo
            + alpha_time[j, 1] * sa.time_6mo
        )  # (N,)
        # S_active[k, i] = True if S_{k,c(i),j} > 0
        S_active = np.ones((6, N), dtype=bool)
        for k in range(6):
            S_k = x_base_vals[j, k] + global_adj
            if k < 5:
                S_k = S_k + fact_ind @ delta[k, j]
            S_active[k] = S_k > 0

        # Negative log-likelihood gradient w.r.t. μ: (μ-y)/(φ·μ^p)
        mu_p = np.power(mu_safe[j], p)
        score_j = np.where(pos, (mu_safe[j] - y[j]) / (phi[j] * mu_p), 0.0)
        # Fisher weight: 1/(φ·μ^p)
        w_j = np.where(pos, 1.0 / (phi[j] * mu_p), 0.0)

        # Build design matrix d_j (N, P) with active-set masking
        d_j = np.zeros((N, P))
        for a_idx, (m_idx, k_idx) in enumerate(active_map):
            w_mj = W[m_idx, j]
            if abs(w_mj) < 1e-15:
                continue
            base = a_idx * N_FACTORIAL
            # Mask by S_active[k_idx, i]
            a_k_masked = sa.A[:, k_idx] * S_active[k_idx]
            for c in range(N_FACTORIAL):
                d_j[:, base + c] = w_mj * a_k_masked * fact_ind[:, c]

        # Gradient contribution
        grad += d_j.T @ score_j

        # Hessian contribution: d_j.T @ diag(w_j) @ d_j
        wd_j = w_j[:, np.newaxis] * d_j
        hess += d_j.T @ wd_j

    # Ridge penalty
    if ridge_scale is not None:
        grad += ridge_lambda * ridge_scale * theta
        hess += ridge_lambda * np.diag(ridge_scale)
    else:
        grad += ridge_lambda * theta
        hess += ridge_lambda * np.eye(P)

    return grad, hess


def theta_ridge_step(
    theta: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    p: float,
    W: np.ndarray,
    active_map: List[Tuple[int, int]],
    alpha_gen: np.ndarray,
    alpha_time: np.ndarray,
    x_base: np.ndarray,
    sa: SampleArrays,
    ridge_lambda: float,
    ridge_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Ridge-regularized Newton step with backtracking line search.

    Returns: (theta_new, step_norm)
    """
    hess_reg = hess + 1e-8 * np.eye(len(theta))
    try:
        step = np.linalg.solve(hess_reg, -grad)
    except np.linalg.LinAlgError:
        step = np.linalg.lstsq(hess_reg, -grad, rcond=None)[0]

    # Current objective
    delta_cur = compute_delta_from_theta(theta, W, active_map)
    mu_cur = compute_mu_all(alpha_gen, alpha_time, delta_cur, x_base, sa)
    mu_cur = np.maximum(mu_cur, 1e-10)
    obj_cur = compute_objective(y, mu_cur, phi, p, theta, ridge_lambda, ridge_scale)

    # Backtracking line search
    step_size = 1.0
    for _ in range(10):
        theta_new = theta + step_size * step
        delta_new = compute_delta_from_theta(theta_new, W, active_map)
        mu_new = compute_mu_all(alpha_gen, alpha_time, delta_new, x_base, sa)
        mu_new = np.maximum(mu_new, 1e-10)
        obj_new = compute_objective(y, mu_new, phi, p, theta_new, ridge_lambda, ridge_scale)
        if obj_new < obj_cur - 1e-4 * step_size * np.dot(grad, step):
            break
        step_size *= 0.5

    theta_new = theta + step_size * step
    return theta_new, float(np.linalg.norm(step_size * step))


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def compute_objective(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    p: float,
    theta: np.ndarray,
    ridge_lambda: float,
    ridge_scale: Optional[np.ndarray] = None,
) -> float:
    """Total objective: negative Tweedie log-likelihood + Ridge penalty.

    Uses the log-likelihood (not D/φ) so that φ updates are absorbed
    consistently — D/φ is not a proper joint objective when φ is re-estimated.
    """
    J, N = y.shape
    total_nll = 0.0
    for j in range(J):
        pos = y[j] > 0
        if np.any(pos):
            total_nll -= tweedie_log_likelihood(y[j, pos], mu[j, pos], phi[j], p)

    penalty = 0.5 * ridge_lambda * np.sum(
        (ridge_scale * theta ** 2) if ridge_scale is not None else theta ** 2
    )

    return total_nll + penalty


# ---------------------------------------------------------------------------
# Main fitting loop
# ---------------------------------------------------------------------------

def fit_factor_model(
    data: sap_data.SAPData,
    W: np.ndarray,
    R: np.ndarray,
    kinase_names: List[str],
    active_map: List[Tuple[int, int]],
    ridge_lambda: float = RIDGE_LAMBDA_DEFAULT,
    p: float = P_INIT,
    max_outer_iter: int = MAX_OUTER_ITER,
    ct_ridge_scale: bool = True,
) -> FactorModelFit:
    """Fit the joint kinase-activity factor model.

    Two-block coordinate descent:
      Block A: Per-site nuisance (α_gen, α_time, φ, γ₀)
      Block B: Global θ via Ridge-regularized Newton
    """
    t0 = time.time()
    J = data.bulk_phospho.shape[0]
    N = data.a_obs.shape[0]
    n_active = len(active_map)
    P = n_active * N_FACTORIAL

    sa = SampleArrays.from_data(data.sample_meta, data.a_obs)
    y = data.bulk_phospho.values.copy()
    y[np.isnan(y)] = 0.0
    x_base = data.x_base

    # Cell-type-specific Ridge scaling (Ā_max / Ā_k per active pair)
    ridge_scale = None
    if ct_ridge_scale:
        a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values
        a_bar_max = a_bar.max()
        ridge_scale = np.ones(P)
        for a_idx, (m_idx, k_idx) in enumerate(active_map):
            scale = a_bar_max / max(a_bar[k_idx], 1e-6)
            base = a_idx * N_FACTORIAL
            ridge_scale[base:base + N_FACTORIAL] = scale

    # Pre-fit pooled hurdle slopes by stratum
    print("Pre-fitting hurdle slopes by stratum...")
    strata = data.intensity_strata
    if strata is None:
        strata = np.zeros(J, dtype=int)
    Q = config.N_INTENSITY_STRATA
    gamma1_by_stratum = np.zeros(Q)
    x_base_vals = x_base.values if hasattr(x_base, 'values') else x_base
    # Use baseline-only μ for initial hurdle fit
    mu_baseline = x_base_vals @ sa.A.T  # (J, N) — approximate
    mu_baseline = np.maximum(mu_baseline, 1e-10)
    for q in range(Q):
        mask_q = strata == q
        if not np.any(mask_q):
            continue
        y_q = y[mask_q].ravel()
        mu_q = mu_baseline[mask_q].ravel()
        _, g1 = fit_hurdle_logistic(y_q, mu_q)
        gamma1_by_stratum[q] = g1

    # Initialize parameters
    theta = np.zeros(P)
    alpha_gen = np.zeros(J)
    alpha_time = np.zeros((J, 2))
    phi = np.ones(J)
    gamma0 = np.zeros(J)

    # Initialize phi from baseline
    for j in range(J):
        pos = y[j] > 0
        if np.any(pos):
            phi[j] = update_phi(y[j], mu_baseline[j], p, n_eff_params=1.0)

    convergence_history = []
    converged = False

    print(f"\nFitting factor model: {n_active} active pairs, "
          f"{P} parameters, {J} sites, {N} samples")
    print(f"Ridge λ = {ridge_lambda}, p = {p}")

    nuisance_freq = 5  # run Block A every N iterations after the first
    for outer_iter in range(max_outer_iter):
        iter_t0 = time.time()

        # --- Block A: Per-site nuisance update (run on iter 0 and every nuisance_freq) ---
        delta = compute_delta_from_theta(theta, W, active_map)

        run_nuisance = (outer_iter == 0) or (outer_iter % nuisance_freq == 0)
        if run_nuisance:
            for j in range(J):
                delta_j = delta[:, j, :]
                x_base_j = x_base_vals[j]
                alpha_gen[j], alpha_time[j], phi[j], gamma0[j] = fit_site_nuisance(
                    j, y[j], delta_j, x_base_j, sa, p,
                    alpha_gen[j], alpha_time[j], phi[j], gamma0[j],
                    gamma1_by_stratum[strata[j]],
                )

        # --- Block B: Global θ update ---
        mu = compute_mu_all(alpha_gen, alpha_time, delta, x_base, sa)
        mu = np.maximum(mu, 1e-10)

        grad, hess = compute_theta_gradient_and_hessian(
            y, mu, phi, p, W, active_map, sa, ridge_lambda, theta,
            delta, x_base_vals, alpha_gen, alpha_time, ridge_scale,
        )

        theta_new, step_norm = theta_ridge_step(
            theta, grad, hess,
            y, phi, p, W, active_map,
            alpha_gen, alpha_time, x_base, sa,
            ridge_lambda, ridge_scale,
        )
        theta = theta_new

        # Recompute objective with accepted θ
        delta = compute_delta_from_theta(theta, W, active_map)
        mu = compute_mu_all(alpha_gen, alpha_time, delta, x_base, sa)
        mu = np.maximum(mu, 1e-10)
        obj = compute_objective(y, mu, phi, p, theta, ridge_lambda, ridge_scale)
        convergence_history.append(obj)

        iter_time = time.time() - iter_t0

        nuis_tag = " [+nuis]" if run_nuisance else ""
        if outer_iter > 0:
            rel_change = abs(obj - convergence_history[-2]) / (abs(convergence_history[-2]) + 1e-10)
            print(f"  Iter {outer_iter:3d}: obj={obj:.2f}, "
                  f"Δobj={rel_change:.2e}, ‖step‖={step_norm:.2e}, "
                  f"time={iter_time:.1f}s{nuis_tag}")
            if rel_change < OUTER_TOL:
                converged = True
                print(f"  Converged at iteration {outer_iter}")
                break
        else:
            print(f"  Iter {outer_iter:3d}: obj={obj:.2f}, "
                  f"‖step‖={step_norm:.2e}, time={iter_time:.1f}s{nuis_tag}")

    wall_time = time.time() - t0
    print(f"\nFitting complete: {outer_iter + 1} iterations, "
          f"{'converged' if converged else 'NOT converged'}, "
          f"{wall_time:.1f}s total")

    # θ summary
    theta_2d = theta.reshape(n_active, N_FACTORIAL)
    print(f"\nθ summary:")
    print(f"  App component: mean={theta_2d[:, 0].mean():.4f}, "
          f"std={theta_2d[:, 0].std():.4f}, "
          f"|max|={np.abs(theta_2d[:, 0]).max():.4f}")
    print(f"  Tau component: mean={theta_2d[:, 1].mean():.4f}, "
          f"std={theta_2d[:, 1].std():.4f}, "
          f"|max|={np.abs(theta_2d[:, 1]).max():.4f}")
    n_nonzero = int(np.sum(np.abs(theta) > 1e-6))
    print(f"  Nonzero (|θ| > 1e-6): {n_nonzero}/{P}")

    params = FactorModelParams(
        theta=theta,
        alpha_gen=alpha_gen,
        alpha_time=alpha_time,
        phi=phi,
        gamma0=gamma0,
        gamma1_by_stratum=gamma1_by_stratum,
    )

    return FactorModelFit(
        params=params,
        p=p,
        active_map=active_map,
        kinase_names=kinase_names,
        ridge_lambda=ridge_lambda,
        converged=converged,
        convergence_history=convergence_history,
        n_outer_iter=outer_iter + 1,
        wall_time=wall_time,
    )


# ---------------------------------------------------------------------------
# Gradient verification (smoke test)
# ---------------------------------------------------------------------------

def verify_gradient(
    data: sap_data.SAPData,
    W: np.ndarray,
    R: np.ndarray,
    active_map: List[Tuple[int, int]],
    n_sites: int = 3,
    eps: float = 1e-5,
) -> bool:
    """Verify analytical gradient matches finite differences on a few sites."""
    print("\n--- Gradient Verification ---")
    sa = SampleArrays.from_data(data.sample_meta, data.a_obs)
    J_full = data.bulk_phospho.shape[0]
    rng = np.random.default_rng(42)
    site_idx = rng.choice(J_full, size=min(n_sites, J_full), replace=False)

    # Use small subset
    y_sub = data.bulk_phospho.values[site_idx].copy()
    y_sub[np.isnan(y_sub)] = 0.0
    x_base_sub = data.x_base.values[site_idx]
    W_sub = W[:, site_idx]

    n_active = len(active_map)
    P = n_active * N_FACTORIAL
    theta = rng.normal(0, 0.01, size=P)
    phi = np.ones(n_sites)
    p = P_INIT
    alpha_gen = np.zeros(n_sites)
    alpha_time = np.zeros((n_sites, 2))

    # Analytical gradient
    delta = compute_delta_from_theta(theta, W_sub, active_map)
    mu = compute_mu_all(alpha_gen, alpha_time, delta,
                        pd.DataFrame(x_base_sub, columns=data.x_base.columns), sa)
    mu = np.maximum(mu, 1e-10)
    grad, _ = compute_theta_gradient_and_hessian(
        y_sub, mu, phi, p, W_sub, active_map, sa, RIDGE_LAMBDA_DEFAULT, theta,
        delta, x_base_sub, alpha_gen, alpha_time,
    )

    # Finite differences
    fd_grad = np.zeros(P)
    n_check = min(20, P)  # check first 20 components
    for idx in range(n_check):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[idx] += eps
        theta_m[idx] -= eps

        delta_p = compute_delta_from_theta(theta_p, W_sub, active_map)
        mu_p = compute_mu_all(alpha_gen, alpha_time, delta_p,
                              pd.DataFrame(x_base_sub, columns=data.x_base.columns), sa)
        mu_p = np.maximum(mu_p, 1e-10)
        obj_p = compute_objective(y_sub, mu_p, phi, p, theta_p, RIDGE_LAMBDA_DEFAULT)

        delta_m = compute_delta_from_theta(theta_m, W_sub, active_map)
        mu_m = compute_mu_all(alpha_gen, alpha_time, delta_m,
                              pd.DataFrame(x_base_sub, columns=data.x_base.columns), sa)
        mu_m = np.maximum(mu_m, 1e-10)
        obj_m = compute_objective(y_sub, mu_m, phi, p, theta_m, RIDGE_LAMBDA_DEFAULT)

        fd_grad[idx] = (obj_p - obj_m) / (2 * eps)

    # Compare
    analytic = grad[:n_check]
    finite = fd_grad[:n_check]

    max_abs_diff = np.max(np.abs(analytic - finite))
    rel_diff = np.abs(analytic - finite) / (np.abs(analytic) + np.abs(finite) + 1e-10)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)

    print(f"  Checked {n_check}/{P} gradient components on {n_sites} sites")
    print(f"  Max absolute diff: {max_abs_diff:.2e}")
    print(f"  Max relative diff: {max_rel_diff:.2e}")
    print(f"  Mean relative diff: {mean_rel_diff:.2e}")

    passed = max_rel_diff < 0.05
    print(f"  {'PASS' if passed else 'FAIL'} (threshold: max rel diff < 0.05)")
    return passed


# ---------------------------------------------------------------------------
# Synthetic validation (extension §3.7)
# ---------------------------------------------------------------------------

def generate_kinase_effects(
    active_map: List[Tuple[int, int]],
    R: np.ndarray,
    n_test: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Generate θ^true for synthetic validation.

    For n_test kinases: nonzero in 1-3 cell types (following R), magnitude
    drawn from N(0, 0.3) for App, N(0, 0.2) for Tau.

    Returns: theta_true (P,) in same layout as model θ.
    """
    rng = np.random.default_rng(seed)
    n_active = len(active_map)
    P = n_active * N_FACTORIAL
    theta_true = np.zeros(P)

    # Group active pairs by kinase
    kinase_pairs: Dict[int, List[int]] = {}
    for a_idx, (m_idx, k_idx) in enumerate(active_map):
        kinase_pairs.setdefault(m_idx, []).append(a_idx)

    # Select test kinases (those with ≥2 active cell types for interesting decomposition)
    eligible = [m for m, pairs in kinase_pairs.items() if len(pairs) >= 2]
    if len(eligible) < n_test:
        eligible = list(kinase_pairs.keys())
    test_kinases = rng.choice(eligible, size=min(n_test, len(eligible)), replace=False)

    for m_idx in test_kinases:
        pairs = kinase_pairs[m_idx]
        # Activate 1-3 cell types
        n_ct = min(rng.integers(1, 4), len(pairs))
        active_pairs = rng.choice(pairs, size=n_ct, replace=False)
        for a_idx in active_pairs:
            base = a_idx * N_FACTORIAL
            theta_true[base] = rng.normal(0, 0.3)      # App
            theta_true[base + 1] = rng.normal(0, 0.2)   # Tau

    n_nonzero = int(np.sum(np.abs(theta_true) > 1e-10))
    print(f"  Generated θ^true: {len(test_kinases)} test kinases, "
          f"{n_nonzero}/{P} nonzero components")

    return theta_true


def validate_factor_model(
    data: sap_data.SAPData,
    W: np.ndarray,
    R: np.ndarray,
    kinase_names: List[str],
    active_map: List[Tuple[int, int]],
    ridge_lambda: float = RIDGE_LAMBDA_DEFAULT,
    seed: int = 42,
) -> Dict:
    """Kinase-level synthetic validation (extension §3.7).

    1. Generate θ^true → compute Δ^true → generate synthetic bulk
    2. Refit factor model on synthetic data
    3. Measure r(θ̂, θ^true) per cell type
    """
    print("\n" + "=" * 60)
    print("Synthetic Validation: Joint Factor Model (§3.7)")
    print("=" * 60)

    n_active = len(active_map)
    sa = SampleArrays.from_data(data.sample_meta, data.a_obs)
    rng = np.random.default_rng(seed)

    # Generate true kinase effects
    print("\nGenerating kinase-level true effects...")
    theta_true = generate_kinase_effects(active_map, R, seed=seed)

    # Compute implied per-site Δ^true
    delta_true = compute_delta_from_theta(theta_true, W, active_map)  # (5, J, 2)

    # Generate synthetic bulk via permuted-residual injection
    print("Generating synthetic bulk data (permuted-residual injection)...")
    J = data.bulk_phospho.shape[0]
    N = data.a_obs.shape[0]
    y_real = data.bulk_phospho.values.copy()
    nan_mask = np.isnan(y_real)
    y_real[nan_mask] = 0.0

    # Baseline prediction (no condition effects)
    x_base_vals = data.x_base.values if hasattr(data.x_base, 'values') else data.x_base
    mu_base = x_base_vals @ sa.A.T  # (J, N)
    mu_base = np.maximum(mu_base, 1e-10)

    # Residuals and permutation
    e = y_real - mu_base
    y_syn = np.zeros_like(y_real)

    fact_ind = sa.fact_indicators[:, :N_FACTORIAL]

    for j in range(J):
        # Permute residuals (destroy condition structure)
        obs = ~nan_mask[j]
        obs_idx = np.where(obs)[0]
        e_perm = np.zeros(N)
        if len(obs_idx) > 1:
            e_obs = e[j, obs_idx].copy()
            rng.shuffle(e_obs)
            e_perm[obs_idx] = e_obs

        # Synthetic signal: baseline + Δ^true
        S_syn = np.broadcast_to(x_base_vals[j][:, np.newaxis], (6, N)).copy()
        delta_contrib = delta_true[:, j, :] @ fact_ind.T  # (5, N)
        S_syn[:5] += delta_contrib
        np.maximum(S_syn, 0.0, out=S_syn)
        mu_syn = np.sum(sa.A.T * S_syn, axis=0)  # (N,)

        y_syn[j] = mu_syn + e_perm
        y_syn[j, nan_mask[j]] = np.nan
        y_syn[j] = np.maximum(y_syn[j], 0.0)
        y_syn[j, nan_mask[j]] = np.nan

    # Refit factor model on synthetic data
    print("\nRefitting factor model on synthetic data...")
    data_syn = sap_data.SAPData(
        a_obs=data.a_obs,
        x_base=data.x_base,
        bulk_phospho=pd.DataFrame(y_syn, columns=data.bulk_phospho.columns),
        sample_meta=data.sample_meta,
        site_meta=data.site_meta,
        n_sites_raw=data.n_sites_raw,
        n_sites_filtered=data.n_sites_filtered,
        intensity_strata=data.intensity_strata,
    )

    model_syn = fit_factor_model(
        data_syn, W, R, kinase_names, active_map,
        ridge_lambda=ridge_lambda, max_outer_iter=15,
    )

    # Measure recovery
    print("\n--- Recovery Metrics ---")
    theta_hat = model_syn.params.theta
    theta_true_2d = theta_true.reshape(n_active, N_FACTORIAL)
    theta_hat_2d = theta_hat.reshape(n_active, N_FACTORIAL)

    results = {}
    cell_types = config.SAP_ESTIMATED_CELLTYPES

    # Per-cell-type recovery
    for k_idx, ct in enumerate(cell_types):
        # Indices of active pairs for this cell type
        ct_indices = [a_idx for a_idx, (m, k) in enumerate(active_map) if k == k_idx]
        if not ct_indices:
            continue

        true_ct = theta_true_2d[ct_indices]   # (n_ct, 2)
        hat_ct = theta_hat_2d[ct_indices]     # (n_ct, 2)

        # Overall correlation
        r_all = _safe_pearson(hat_ct.ravel(), true_ct.ravel())
        slope = _safe_slope(hat_ct.ravel(), true_ct.ravel())

        # Per-component
        r_app = _safe_pearson(hat_ct[:, 0], true_ct[:, 0])
        r_tau = _safe_pearson(hat_ct[:, 1], true_ct[:, 1])

        results[ct] = {
            "pearson_all": r_all,
            "slope_all": slope,
            "pearson_App": r_app,
            "pearson_Tau": r_tau,
            "n_pairs": len(ct_indices),
        }

        ct_short = ct.replace("_neurons", "").replace("_", " ")
        print(f"  {ct_short:<20} r={r_all:+.4f}  slope={slope:.3f}  "
              f"r_App={r_app:+.4f}  r_Tau={r_tau:+.4f}  "
              f"(n={len(ct_indices)} pairs)")

    # Overall
    r_global = _safe_pearson(theta_hat, theta_true)
    results["overall"] = {"pearson": r_global, "n_params": len(theta_hat)}
    print(f"\n  Overall r(θ̂, θ^true) = {r_global:+.4f}")

    # Pass criteria (extension §3.7)
    # r > 0.50 for top 3 cell types by composition
    a_bar = data.a_obs[cell_types].mean(axis=0).sort_values(ascending=False)
    top3 = list(a_bar.index[:3])
    pass_r = all(results.get(ct, {}).get("pearson_all", 0) > 0.50 for ct in top3)
    results["pass"] = pass_r
    print(f"\n  Top-3 cell types: {[ct.split('_')[0] for ct in top3]}")
    print(f"  Pass criterion (r > 0.50 for top 3): {'PASS' if pass_r else 'FAIL'}")

    return results


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_slope(hat: np.ndarray, true: np.ndarray) -> float:
    denom = np.sum(true ** 2)
    if denom < 1e-10:
        return 0.0
    return float(np.sum(hat * true) / denom)


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_factor_model_inputs(
    data: sap_data.SAPData,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int]]]:
    """Load W, R, kinase_names, and build active_map."""
    print("\n--- Loading factor model inputs ---")
    W, kinase_names, site_mask = build_W_matrix(data.site_meta)
    W = threshold_W(W, W_PERCENTILE)

    # Build R matrix
    allen_cache = get_expression_cache(config.ALLEN_EXPRESSION_CACHE_FILE)
    R, R_ann = build_R_matrix(kinase_names, data, allen_cache)

    # Filter W to matched sites only
    if not np.all(site_mask):
        print(f"  Warning: {(~site_mask).sum()} sites not matched — filling with zeros")
        W_full = np.zeros((W.shape[0], len(site_mask)))
        W_full[:, site_mask] = W
        W = W_full

    # Build active map: (kinase_idx, celltype_idx) where R=1
    active_map = []
    for m in range(R.shape[0]):
        for k in range(R.shape[1]):
            if R[m, k] == 1:
                active_map.append((m, k))

    print(f"  Active pairs: {len(active_map)} "
          f"({len(active_map) * N_FACTORIAL} parameters)")

    return W, R, kinase_names, active_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Joint kinase-activity factor model (sap_extension.md §3).",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Gradient verification on 3 sites")
    parser.add_argument("--fit", action="store_true",
                        help="Full model fit")
    parser.add_argument("--validate", action="store_true",
                        help="Kinase-level synthetic validation (§3.7)")
    parser.add_argument("--ridge-lambda", type=float, default=RIDGE_LAMBDA_DEFAULT,
                        help=f"Ridge penalty (default: {RIDGE_LAMBDA_DEFAULT})")
    parser.add_argument("--max-iter", type=int, default=MAX_OUTER_ITER,
                        help=f"Max outer iterations (default: {MAX_OUTER_ITER})")
    parser.add_argument("--summary", action="store_true",
                        help="Print cached results")
    args = parser.parse_args()

    if args.summary:
        summary_path = os.path.join(FACTOR_MODEL_DIR, "factor_model_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print(f"No cached results at {summary_path}")
        return

    # Load data
    print("Loading data (Phase 0 + Phase 1)...")
    data, diag_report = sap_data.load_all(include_rna=True)
    W, R, kinase_names, active_map = load_factor_model_inputs(data)

    if args.smoke_test:
        passed = verify_gradient(data, W, R, active_map)
        sys.exit(0 if passed else 1)

    if args.fit:
        os.makedirs(FACTOR_MODEL_DIR, exist_ok=True)
        model = fit_factor_model(
            data, W, R, kinase_names, active_map,
            ridge_lambda=args.ridge_lambda,
            max_outer_iter=args.max_iter,
        )
        # Save summary
        summary = {
            "converged": model.converged,
            "n_outer_iter": model.n_outer_iter,
            "wall_time": model.wall_time,
            "ridge_lambda": model.ridge_lambda,
            "n_active_pairs": len(active_map),
            "n_params": len(active_map) * N_FACTORIAL,
            "final_objective": model.convergence_history[-1] if model.convergence_history else None,
        }
        with open(os.path.join(FACTOR_MODEL_DIR, "factor_model_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {FACTOR_MODEL_DIR}/factor_model_summary.json")

    if args.validate:
        os.makedirs(FACTOR_MODEL_DIR, exist_ok=True)
        results = validate_factor_model(
            data, W, R, kinase_names, active_map,
            ridge_lambda=args.ridge_lambda,
        )
        with open(os.path.join(FACTOR_MODEL_DIR, "validation_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved validation results to {FACTOR_MODEL_DIR}/validation_results.json")


if __name__ == "__main__":
    main()
