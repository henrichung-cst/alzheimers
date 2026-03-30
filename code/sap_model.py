"""
SAP Hurdle-Tweedie model: estimation engine for condition-specific deconvolution.

Phase 2 of the SAP implementation. Provides:
- Tweedie distribution utilities (deviance, variance, log-likelihood)
- Parameter structures (SiteParams, ModelFit)
- Mean model computation with non-negativity projection (§3.2)
- Penalized IRLS solver with Group Lasso (§3.5, §4.1)
- Profile likelihood outer loop (§3.5)
- LOCO-CV for hyperparameter selection (§4.2)

Usage:
    python code/sap_model.py --fit              # full LOCO-CV + fit
    python code/sap_model.py --fit-fast         # coarse grid, site subsampling
    python code/sap_model.py --fit-site 42      # single site debug
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


# ============================================================================
# Tweedie distribution utilities (§3.1)
# ============================================================================

def tweedie_variance(mu: np.ndarray, p: float) -> np.ndarray:
    """Tweedie variance function V(mu) = mu^p."""
    return np.power(np.maximum(mu, 1e-10), p)


def tweedie_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    p: float,
) -> np.ndarray:
    """Unit deviance d(y, mu) for Tweedie with power p in (1, 2).

    d(y, mu) = 2 * [y^{2-p}/((1-p)(2-p)) - y*mu^{1-p}/(1-p) + mu^{2-p}/(2-p)]

    Returns per-observation deviance (non-negative). Handles y=0 case
    (valid for Tweedie p in (1,2) since the distribution has a point mass at 0).
    """
    mu = np.maximum(mu, 1e-10)
    y = np.maximum(y, 0.0)

    a = 1.0 - p
    b = 2.0 - p

    # y^{2-p} term: when y=0 and p<2, this is 0
    term1 = np.where(y > 0, np.power(y, b) / (a * b), 0.0)
    term2 = y * np.power(mu, a) / a
    term3 = np.power(mu, b) / b

    return 2.0 * (term1 - term2 + term3)


def tweedie_deviance_residuals(
    y: np.ndarray,
    mu: np.ndarray,
    p: float,
) -> np.ndarray:
    """Signed deviance residuals: sign(y - mu) * sqrt(d(y, mu))."""
    d = tweedie_deviance(y, mu, p)
    return np.sign(y - mu) * np.sqrt(np.maximum(d, 0.0))


def tweedie_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    phi: float,
    p: float,
) -> float:
    """Tweedie log-likelihood using the saddlepoint approximation.

    For the Tweedie with power p in (1,2), the exact density involves an
    infinite series (Dunn & Smyth, 2005). The saddlepoint approximation
    (Dunn & Smyth, 2001) is accurate and numerically stable:

        log f(y; mu, phi, p) ≈ -(1/2) * [d(y,mu)/phi + log(2*pi*phi*V(y))]
                                 + adjustment for y=0

    For y=0 (point mass): P(Y=0) = exp(-mu^{2-p} / (phi*(2-p)))

    This approximation is sufficient for profile likelihood optimization
    over p and phi.
    """
    mu = np.maximum(mu, 1e-10)
    n = len(y)

    # Separate zero and positive observations
    is_zero = y <= 0
    is_pos = ~is_zero

    ll = 0.0

    # Zero observations: log P(Y=0) = -mu^{2-p} / (phi*(2-p))
    if np.any(is_zero):
        ll += np.sum(-np.power(mu[is_zero], 2.0 - p) / (phi * (2.0 - p)))

    # Positive observations: saddlepoint approximation
    if np.any(is_pos):
        y_pos = y[is_pos]
        mu_pos = mu[is_pos]

        d = tweedie_deviance(y_pos, mu_pos, p)
        v_y = tweedie_variance(y_pos, p)

        # Saddlepoint: log f ≈ -d/(2*phi) - 0.5*log(2*pi*phi*v_y)
        ll += np.sum(-d / (2.0 * phi) - 0.5 * np.log(2.0 * np.pi * phi * v_y))

    return float(ll)


def tweedie_total_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    p: float,
) -> float:
    """Total (summed) Tweedie deviance across all observations."""
    return float(np.sum(tweedie_deviance(y, mu, p)))


# ============================================================================
# Parameter structures (§3.2–§3.5)
# ============================================================================

@dataclass
class SiteParams:
    """Parameters for one phosphosite j."""

    # Factorial condition effects: (5 cell types × 3 factorial terms)
    # beta[k, :] = [beta^App, beta^Tau, beta^Int] for cell type k
    beta: np.ndarray  # shape (5, 3)

    # Global covariates
    alpha_gen: float = 0.0          # gender effect (female indicator)
    alpha_time: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [4mo, 6mo]

    # RNA coupling coefficient (§3.2.1)
    rho: float = 0.0

    # Hurdle component (§3.1)
    gamma0: float = 0.0             # dropout intercept
    gamma1: float = 0.0             # dropout slope on log(mu_hat)

    # Tweedie dispersion (site-specific)
    phi: float = 1.0

    # Intensity stratum index (0 to Q-1)
    stratum: int = 0

    @staticmethod
    def zeros(stratum: int = 0) -> "SiteParams":
        """Create zero-initialized parameters."""
        return SiteParams(
            beta=np.zeros((5, 3)),
            alpha_gen=0.0,
            alpha_time=np.zeros(2),
            rho=0.0,
            gamma0=0.0,
            gamma1=0.0,
            phi=1.0,
            stratum=stratum,
        )


@dataclass
class CVResult:
    """LOCO-CV hyperparameter selection result (§4.2)."""

    best_lambda: np.ndarray        # shape (Q,): per-stratum Group Lasso penalties
    best_lambda_rho: float         # ridge penalty on rho_j
    best_eta: float                # interaction penalty multiplier
    best_gamma: float              # CVS adaptive weight exponent
    selected_p: float              # global Tweedie power parameter
    cv_scores: Optional[pd.DataFrame] = None  # full grid search results


@dataclass
class ModelFit:
    """Complete fitted model across all sites."""

    site_params: List[SiteParams]
    p: float                       # global Tweedie power parameter
    cv_result: Optional[CVResult] = None
    converged: Optional[np.ndarray] = None    # (n_sites,) bool
    binding_freq: Optional[pd.DataFrame] = None  # non-negativity binding stats


@dataclass
class SampleArrays:
    """Pre-extracted numpy arrays from sample_meta and a_obs.

    These are constant for a given dataset split (full data or CV fold)
    and should be computed once, then passed through the call chain to
    avoid repeated pandas DataFrame extractions in the hot path.
    """

    female_ind: np.ndarray       # (n_samples,) float: 1.0 if female
    time_4mo: np.ndarray         # (n_samples,) float: 1.0 if 4mo
    time_6mo: np.ndarray         # (n_samples,) float: 1.0 if 6mo
    fact_indicators: np.ndarray  # (n_samples, 3) float: factorial coding
    A: np.ndarray                # (n_samples, 6) float: a_obs composition matrix

    @staticmethod
    def from_data(
        sample_meta: pd.DataFrame,
        a_obs: pd.DataFrame,
    ) -> "SampleArrays":
        """Extract constant arrays from pandas DataFrames."""
        assert list(a_obs.columns) == config.SAP_CELLTYPES, (
            f"a_obs columns {list(a_obs.columns)} != {config.SAP_CELLTYPES}"
        )
        conditions = sample_meta["condition"].values
        return SampleArrays(
            female_ind=(sample_meta["gender"].values == "fe").astype(float),
            time_4mo=(sample_meta["timepoint"].values == "4mo").astype(float),
            time_6mo=(sample_meta["timepoint"].values == "6mo").astype(float),
            fact_indicators=np.array(
                [config.SAP_FACTORIAL[c] for c in conditions], dtype=float,
            ),
            A=a_obs.values.copy(),
        )

    @staticmethod
    def ensure(
        cached: Optional["SampleArrays"],
        sample_meta: pd.DataFrame,
        a_obs: pd.DataFrame,
    ) -> "SampleArrays":
        """Return cached arrays if available, otherwise extract from DataFrames."""
        if cached is not None:
            return cached
        return SampleArrays.from_data(sample_meta, a_obs)


# ============================================================================
# Model serialization
# ============================================================================

def save_model(model: ModelFit, path: Optional[str] = None) -> str:
    """Serialize a ModelFit to disk (.npz arrays + .json metadata).

    Returns the path to the .npz file.
    """
    if path is None:
        path = config.SAP_MODEL_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n = len(model.site_params)
    beta_all = np.array([sp.beta for sp in model.site_params])       # (n, 5, 3)
    alpha_gen = np.array([sp.alpha_gen for sp in model.site_params])  # (n,)
    alpha_time = np.array([sp.alpha_time for sp in model.site_params])  # (n, 2)
    rho = np.array([sp.rho for sp in model.site_params])              # (n,)
    gamma0 = np.array([sp.gamma0 for sp in model.site_params])        # (n,)
    gamma1 = np.array([sp.gamma1 for sp in model.site_params])        # (n,)
    phi = np.array([sp.phi for sp in model.site_params])              # (n,)
    strata = np.array([sp.stratum for sp in model.site_params])       # (n,)
    converged = model.converged if model.converged is not None else np.zeros(n, dtype=bool)

    np.savez_compressed(
        path,
        beta_all=beta_all, alpha_gen=alpha_gen, alpha_time=alpha_time,
        rho=rho, gamma0=gamma0, gamma1=gamma1, phi=phi,
        strata=strata, converged=converged,
    )

    # Sidecar JSON for scalar metadata
    json_path = path.replace(".npz", ".json")
    meta: Dict = {
        "schema_version": 1,
        "p": model.p,
        "n_sites": n,
    }
    if model.cv_result is not None:
        meta["cv_result"] = {
            "best_lambda": model.cv_result.best_lambda.tolist(),
            "best_lambda_rho": model.cv_result.best_lambda_rho,
            "best_eta": model.cv_result.best_eta,
            "best_gamma": model.cv_result.best_gamma,
            "selected_p": model.cv_result.selected_p,
        }
    if model.binding_freq is not None:
        meta["binding_freq"] = model.binding_freq.to_dict()
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved: {path} ({n} sites)")
    return path


def load_model(path: Optional[str] = None) -> ModelFit:
    """Deserialize a ModelFit from disk."""
    if path is None:
        path = config.SAP_MODEL_FILE

    data = np.load(path)
    json_path = path.replace(".npz", ".json")
    with open(json_path) as f:
        meta = json.load(f)

    n = meta["n_sites"]
    site_params: List[SiteParams] = []
    for i in range(n):
        site_params.append(SiteParams(
            beta=data["beta_all"][i],
            alpha_gen=float(data["alpha_gen"][i]),
            alpha_time=data["alpha_time"][i],
            rho=float(data["rho"][i]),
            gamma0=float(data["gamma0"][i]),
            gamma1=float(data["gamma1"][i]),
            phi=float(data["phi"][i]),
            stratum=int(data["strata"][i]),
        ))

    cv_result = None
    if "cv_result" in meta:
        cv = meta["cv_result"]
        cv_result = CVResult(
            best_lambda=np.array(cv["best_lambda"]),
            best_lambda_rho=cv["best_lambda_rho"],
            best_eta=cv["best_eta"],
            best_gamma=cv["best_gamma"],
            selected_p=cv["selected_p"],
        )

    binding_freq = None
    if "binding_freq" in meta:
        binding_freq = pd.DataFrame(meta["binding_freq"])

    return ModelFit(
        site_params=site_params,
        p=meta["p"],
        cv_result=cv_result,
        converged=data["converged"],
        binding_freq=binding_freq,
    )


# ============================================================================
# Mean model computation (§3.2)
# ============================================================================

def _compute_s_core(
    params: SiteParams,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    project: bool,
    cached_arrays: Optional["SampleArrays"] = None,
) -> np.ndarray:
    """Core S-matrix computation shared by compute_s_matrix and _compute_s_unconstrained.

    Args:
        project: If True, apply non-negativity projection max(S, 0).
        cached_arrays: Pre-extracted numpy arrays. If None, extracts from DataFrames.
    """
    if cached_arrays is not None:
        female_ind = cached_arrays.female_ind
        time_4mo = cached_arrays.time_4mo
        time_6mo = cached_arrays.time_6mo
        fact_indicators = cached_arrays.fact_indicators
        n_samples = len(female_ind)
    else:
        n_samples = len(sample_meta)
        female_ind = (sample_meta["gender"].values == "fe").astype(float)
        time_4mo = (sample_meta["timepoint"].values == "4mo").astype(float)
        time_6mo = (sample_meta["timepoint"].values == "6mo").astype(float)
        conditions = sample_meta["condition"].values
        fact_indicators = np.array(
            [config.SAP_FACTORIAL[c] for c in conditions], dtype=float,
        )

    S = np.tile(x_base_j, (n_samples, 1)).T  # (6, n_samples)

    global_adj = (params.alpha_gen * female_ind
                  + params.alpha_time[0] * time_4mo
                  + params.alpha_time[1] * time_6mo)
    S += global_adj[np.newaxis, :]

    S += params.rho * r_slice_j

    S[:5, :] += params.beta @ fact_indicators.T

    if project:
        np.maximum(S, 0.0, out=S)

    return S


def compute_s_matrix(
    params: SiteParams,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    cached_arrays: Optional["SampleArrays"] = None,
) -> np.ndarray:
    """Compute S_{k,c(i),j} for all samples with non-negativity projection (§3.2).

    Returns: (6, n_samples) matrix of cell-type-specific signals.
    """
    return _compute_s_core(params, sample_meta, x_base_j, r_slice_j,
                           project=True, cached_arrays=cached_arrays)


def _compute_mu_and_s(
    params: SiteParams,
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    cached_arrays: Optional["SampleArrays"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute both mu (n_samples,) and S (6, n_samples) in one pass.

    Avoids recomputing the S-matrix when both mu and S are needed (e.g.,
    in update_rho which needs S for active-set logic and mu for scoring).
    """
    sa = SampleArrays.ensure(cached_arrays, sample_meta, a_obs)
    S = compute_s_matrix(params, sample_meta, x_base_j, r_slice_j,
                         cached_arrays=sa)
    mu = np.sum(sa.A * S.T, axis=1)
    return np.maximum(mu, 1e-10), S


def compute_mu(
    params: SiteParams,
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    cached_arrays: Optional["SampleArrays"] = None,
) -> np.ndarray:
    """Compute predicted bulk intensity mu_{i,j} for all 24 samples.

    mu_{i,j} = sum_k A_{i,k} * S_{k,c(i),j}

    Returns: (n_samples,) array of predicted intensities.
    """
    mu, _ = _compute_mu_and_s(params, a_obs, sample_meta, x_base_j, r_slice_j,
                              cached_arrays=cached_arrays)
    return mu


def compute_dropout_prob(
    gamma0: float,
    gamma1: float,
    log_mu: np.ndarray,
) -> np.ndarray:
    """Dropout probability pi_{i,j} via logistic model (§3.1).

    logit(pi) = gamma_0 + gamma_1 * log(mu_hat)
    """
    logit = gamma0 + gamma1 * log_mu
    # Clip for numerical stability
    logit = np.clip(logit, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-logit))


# ============================================================================
# Hurdle-Tweedie log-likelihood (§3.1)
# ============================================================================

def hurdle_tweedie_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    phi: float,
    p: float,
    gamma0: float,
    gamma1: float,
) -> float:
    """Full Hurdle-Tweedie log-likelihood for one site across 24 samples.

    L = sum_i [ I(y_i=0)*log(pi_i) + I(y_i>0)*log(1-pi_i)
                + I(y_i>0)*log(f_Tweedie^+(y_i; mu_i, phi, p)) ]

    where pi is the dropout probability and f^+ is the zero-truncated Tweedie.
    """
    log_mu = np.log(np.maximum(mu, 1e-10))
    pi = compute_dropout_prob(gamma0, gamma1, log_mu)
    pi = np.clip(pi, 1e-10, 1.0 - 1e-10)

    is_zero = y <= 0
    is_pos = ~is_zero

    ll = 0.0

    # Zero observations: log(pi)
    if np.any(is_zero):
        ll += np.sum(np.log(pi[is_zero]))

    # Positive observations: log(1 - pi) + log(f^+)
    if np.any(is_pos):
        ll += np.sum(np.log(1.0 - pi[is_pos]))

        y_pos = y[is_pos]
        mu_pos = mu[is_pos]

        # Zero-truncated Tweedie: f^+(y) = f(y) / P(Y > 0)
        # log f^+(y) = log f(y) - log P(Y > 0)
        # P(Y=0) = exp(-mu^{2-p} / (phi*(2-p)))
        # log P(Y>0) = log(1 - exp(-mu^{2-p} / (phi*(2-p))))

        d = tweedie_deviance(y_pos, mu_pos, p)
        v_y = tweedie_variance(y_pos, p)

        # Saddlepoint log f(y)
        log_f = -d / (2.0 * phi) - 0.5 * np.log(2.0 * np.pi * phi * v_y)

        # log P(Y>0) for truncation correction
        neg_lambda = -np.power(mu_pos, 2.0 - p) / (phi * (2.0 - p))
        log_p_pos = np.log(np.maximum(1.0 - np.exp(neg_lambda), 1e-30))

        ll += np.sum(log_f - log_p_pos)

    return float(ll)


# ============================================================================
# Group Lasso proximal operator (§4.1, §4.4)
# ============================================================================

def group_lasso_prox(
    delta_k: np.ndarray,
    lambda_q: float,
    omega_k: float,
    eta: float = 1.0,
    step_size: float = 1.0,
) -> np.ndarray:
    """Proximal operator for one Group Lasso group delta_{k,j} in R^3.

    Operates in delta-space (bulk-contribution units) where
    delta_{k,c,j} = Ā_k · beta_{k,c,j}.

    prox(delta) = delta * max(1 - tau / ||delta||_2, 0)

    where tau = step_size * lambda_q * omega_k * sqrt(3).
    No proportion factor — the reparametrization to delta-space makes the
    penalty directly threshold bulk-space effect sizes, which are comparable
    across cell types regardless of their mixing proportions.

    The interaction term (index 2) gets an additional eta multiplier.

    Args:
        delta_k: (3,) array [delta^App, delta^Tau, delta^Int].
        lambda_q: Group Lasso penalty for this site's stratum.
        omega_k: Adaptive weight for this cell type.
        eta: Interaction penalty multiplier (>= 2.0).
        step_size: IRLS step size.

    Returns:
        Proximal-updated delta_k (3,).
    """
    # Apply additional interaction penalty by scaling the interaction component
    # in the norm computation and shrinkage
    delta_scaled = delta_k.copy()
    delta_scaled[2] *= np.sqrt(eta)  # inflate interaction for norm

    norm = np.linalg.norm(delta_scaled)
    tau = step_size * lambda_q * omega_k * np.sqrt(3.0)

    if norm <= tau:
        return np.zeros(3)

    shrink = 1.0 - tau / norm
    result = delta_scaled * shrink
    # Undo the interaction scaling
    result[2] /= np.sqrt(eta)
    return result


def compute_adaptive_weights(
    cvs: pd.DataFrame,
    gamma: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Adaptive Group Lasso weights omega_k (§4.3).

    omega_k = 1 / (CVS_k^gamma + epsilon)

    where CVS_k = max_c CVS_{k,c} is the maximum CVS across conditions.

    Args:
        cvs: (5 cell types × 3 conditions) DataFrame.
        gamma: Exponent controlling RNA guidance strength.
        epsilon: Floor to prevent division by zero.

    Returns:
        (5,) array of adaptive weights, one per estimated cell type.
    """
    if gamma == 0:
        # Non-adaptive: uniform weights
        return np.ones(len(config.SAP_ESTIMATED_CELLTYPES))

    cvs_max = cvs.max(axis=1).values  # (5,): max CVS per cell type
    return 1.0 / (np.power(cvs_max, gamma) + epsilon)


def compute_lambda_max(
    data,  # SAPData
    omega: np.ndarray,
    a_bar: np.ndarray,
    p: float = 1.5,
    n_sites_sample: int = 100,
    seed: int = 42,
) -> float:
    """Compute the data-scale-calibrated lambda_max in δ-space.

    lambda_max is the smallest Group Lasso penalty that would shrink ALL
    cell-type groups to zero for all sampled sites. Computed from the
    deviance gradient w.r.t. δ_k at δ=0:

        lambda_max_j = max_k ||grad_{δ_k} D|_{δ=0}|| / (omega_k * sqrt(3))

    The gradient w.r.t. δ_{k,c} is: sum_{i in c} (A_{i,k}/Ā_k) * (y_i - mu_i) / mu_i^{p-1}

    No Ā_k in the denominator — the reparametrization absorbs it.

    Args:
        data: SAPData with Phase 0+1 fields.
        omega: (5,) adaptive weights per cell type.
        a_bar: (5,) mean proportion per cell type.
        p: Tweedie power parameter.
        n_sites_sample: number of sites to sample for calibration.
        seed: random seed.

    Returns:
        lambda_max (float).
    """
    rng = np.random.RandomState(seed)
    n_sites = data.n_sites_filtered
    sample_idx = rng.choice(n_sites, min(n_sites_sample, n_sites), replace=False)

    y_all = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_all = data.x_base.values
    A = data.a_obs.values  # (24, 6)

    # Build factorial indicators
    conditions = data.sample_meta["condition"].values
    app_ind = np.array([config.SAP_FACTORIAL[c][0] for c in conditions], dtype=float)
    tau_ind = np.array([config.SAP_FACTORIAL[c][1] for c in conditions], dtype=float)
    int_ind = np.array([config.SAP_FACTORIAL[c][2] for c in conditions], dtype=float)

    lam_max_all = []

    for j in sample_idx:
        y = y_all[j]
        pos_mask = y > 0
        if not np.any(pos_mask):
            continue

        # At beta=0, mu = sum_k A_{i,k} * X^base_{k,j} (plus global covariates = 0)
        mu_null = A @ x_base_all[j]  # (24,)
        mu_null = np.maximum(mu_null, 1e-10)

        # Deviance gradient in δ-space (phi-free):
        # grad_{δ_{k,c}} D = sum_{i in c} (A_{i,k}/Ā_k) * (y_i - mu_i) / mu_i^{p-1}
        # Using A/Ā instead of A because δ_k = Ā_k · β_k.
        score = np.where(pos_mask, (y - mu_null) / np.power(mu_null, p - 1.0), 0.0)

        lam_max_j = 0.0
        for k in range(5):
            a_ratio = A[:, k] / max(a_bar[k], 1e-10)  # A_{i,k}/Ā_k
            grad_k = np.array([
                np.sum(a_ratio * score * app_ind),
                np.sum(a_ratio * score * tau_ind),
                np.sum(a_ratio * score * int_ind),
            ])
            grad_norm = np.linalg.norm(grad_k)
            # Sparsity threshold in δ-space: lambda * omega_k * sqrt(3)
            denom = omega[k] * np.sqrt(3.0)
            if denom > 1e-10:
                lam_max_jk = grad_norm / denom
                lam_max_j = max(lam_max_j, lam_max_jk)

        lam_max_all.append(lam_max_j)

    return float(np.max(lam_max_all)) if lam_max_all else 1.0


# ============================================================================
# IRLS inner loop (§3.5)
# ============================================================================

def _build_inner_design(
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    a_bar: np.ndarray,
) -> np.ndarray:
    """Build the 24 × 18 design matrix for the IRLS inner loop.

    In δ-space (delta = Ā_k · beta), the design columns for cell type k use
    A_{i,k}/Ā_k instead of A_{i,k}. This ratio is centered around 1.0 for
    each cell type, making the Hessian blocks comparable in scale.

    Columns:
    - 0: gender indicator (female=1)
    - 1-2: timepoint indicators (4mo, 6mo)
    - 3-17: (A_{i,k}/Ā_k) * factorial indicators (5 cell types × 3 terms)

    The RNA covariate enters as an offset (not in the design matrix).
    """
    conditions = sample_meta["condition"].values
    app_ind = np.array([config.SAP_FACTORIAL[c][0] for c in conditions], dtype=float)
    tau_ind = np.array([config.SAP_FACTORIAL[c][1] for c in conditions], dtype=float)
    int_ind = np.array([config.SAP_FACTORIAL[c][2] for c in conditions], dtype=float)

    cols = []

    # Global covariates (3 columns)
    cols.append((sample_meta["gender"].values == "fe").astype(float))
    cols.append((sample_meta["timepoint"].values == "4mo").astype(float))
    cols.append((sample_meta["timepoint"].values == "6mo").astype(float))

    # Cell-type × factorial (15 columns): use A_{i,k}/Ā_k for δ-space
    for idx_k, ct in enumerate(config.SAP_ESTIMATED_CELLTYPES):
        a_k = a_obs[ct].values  # (n_samples,)
        a_ratio = a_k / max(a_bar[idx_k], 1e-10)  # A_{i,k}/Ā_k ~ O(1)
        cols.append(a_ratio * app_ind)
        cols.append(a_ratio * tau_ind)
        cols.append(a_ratio * int_ind)

    return np.column_stack(cols)  # (24, 18)


def _theta_to_params(
    theta: np.ndarray,
    rho: float,
    gamma0: float,
    gamma1: float,
    phi: float,
    stratum: int,
    a_bar: Optional[np.ndarray] = None,
) -> SiteParams:
    """Convert flat theta vector (18,) to SiteParams.

    When a_bar is provided, theta[3:18] is in δ-space (bulk-contribution
    units) and is converted to β-space via β_k = δ_k / Ā_k.
    When a_bar is None, theta is β-space directly.
    """
    alpha_gen = theta[0]
    alpha_time = theta[1:3].copy()
    delta = theta[3:18].reshape(5, 3)
    if a_bar is not None:
        beta = delta / np.maximum(a_bar, 1e-10)[:, np.newaxis]
    else:
        beta = delta
    return SiteParams(
        beta=beta,
        alpha_gen=alpha_gen,
        alpha_time=alpha_time,
        rho=rho,
        gamma0=gamma0,
        gamma1=gamma1,
        phi=phi,
        stratum=stratum,
    )


def _params_to_theta(
    params: SiteParams,
    a_bar: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert SiteParams to flat theta vector (18,).

    When a_bar is provided, converts β → δ-space (δ_k = Ā_k · β_k).
    When a_bar is None, stores β directly.
    """
    theta = np.zeros(18)
    theta[0] = params.alpha_gen
    theta[1:3] = params.alpha_time
    if a_bar is not None:
        delta = params.beta * a_bar[:, np.newaxis]
        theta[3:18] = delta.ravel()
    else:
        theta[3:18] = params.beta.ravel()
    return theta


def irls_inner_loop(
    y: np.ndarray,
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    rho: float,
    phi: float,
    p: float,
    lambda_q: float,
    omega: np.ndarray,
    a_bar: np.ndarray,
    eta: float,
    stratum: int,
    theta_init: Optional[np.ndarray] = None,
    max_iter: int = None,
    tol: float = None,
    design_matrix: Optional[np.ndarray] = None,
    cached_arrays: Optional["SampleArrays"] = None,
) -> Tuple[SiteParams, bool, float]:
    """Penalized IRLS for the continuous (Tweedie) component at one site.

    Operates in δ-space where δ_{k,c} = Ā_k · β_{k,c}. The design matrix
    uses A_{i,k}/Ā_k columns so that the Hessian blocks are comparable
    across cell types. The Group Lasso proximal step thresholds δ-norms
    directly (no proportion factor).

    Args:
        y: (24,) observed bulk intensities (including zeros).
        a_obs, sample_meta: sample-level data.
        x_base_j: (6,) baseline per cell type.
        r_slice_j: (6, 24) RNA covariates.
        rho: fixed RNA coupling coefficient.
        phi, p: fixed Tweedie parameters.
        lambda_q: Group Lasso penalty for this stratum.
        omega: (5,) adaptive weights per cell type.
        a_bar: (5,) mean proportion per estimated cell type.
        eta: interaction penalty multiplier.
        stratum: intensity stratum index.
        theta_init: optional (18,) initial parameter vector (δ-space).
        max_iter: IRLS iterations (default from config).
        tol: convergence tolerance (default from config).

    Returns:
        (params, converged, final_deviance) — params has β in original space.
    """
    if max_iter is None:
        max_iter = config.IRLS_MAX_ITER
    if tol is None:
        tol = config.IRLS_TOL

    # Use only positive observations for the continuous component
    # (zeros are handled by the hurdle)
    pos_mask = y > 0
    y_pos = y.copy()  # keep all for mu computation; weight zeros to 0

    D = design_matrix if design_matrix is not None else _build_inner_design(a_obs, sample_meta, a_bar)  # (24, 18)

    # Initialize
    if theta_init is not None:
        theta = theta_init.copy()
    else:
        theta = np.zeros(18)

    converged = False
    prev_dev = np.inf
    dev = np.inf
    mu = None  # will be set at end of each iteration (or fresh at start)

    for iteration in range(max_iter):
        # Reconstruct params (δ→β conversion) to compute mu
        # Reuse mu from end of previous iteration if available
        if mu is None:
            params = _theta_to_params(theta, rho, 0.0, 0.0, phi, stratum, a_bar=a_bar)
            mu = compute_mu(params, a_obs, sample_meta, x_base_j, r_slice_j,
                           cached_arrays=cached_arrays)

        # Compute working weights and response (log link)
        log_mu = np.log(mu)
        w = np.power(mu, 2.0 - p) / phi  # (24,)

        # Zero out weights for zero observations (handled by hurdle)
        w[~pos_mask] = 0.0

        z = log_mu + (y - mu) / mu  # (24,)

        # Offset from RNA covariate and baseline
        # The working response needs to be adjusted for the offset
        # eta_offset = log(sum_k A_{k} * (X_base + rho*r)) computed from current params
        # But in IRLS with log link, we work in eta = log(mu) space.
        # The design matrix D maps theta → additive perturbation in S-space,
        # which is then mixed by A to get mu. This is NOT a standard GLM
        # because the link is on the mixed mu, not on S directly.

        # Due to the non-standard structure (mixing before link), we use a
        # gradient-based approach rather than classical IRLS.

        # Gradient of Tweedie deviance w.r.t. theta:
        # dD/dtheta = sum_i w_i * (y_i - mu_i) / mu_i * dmu_i/dtheta
        # where dmu_i/dtheta_p = sum_k A_{i,k} * dS_{k,i}/dtheta_p

        # Compute dmu/dtheta: (24, 18) Jacobian (δ-space)
        J = _compute_jacobian(theta, a_obs, sample_meta, x_base_j, r_slice_j,
                              rho, stratum, a_bar, cached_arrays=cached_arrays)

        # Gradient of negative log-likelihood (Tweedie quasi-score)
        # For Tweedie with log link: score_i = (y_i - mu_i) / (phi * mu_i^{p-1})
        score = np.where(pos_mask, (y - mu) / (phi * np.power(mu, p - 1.0)), 0.0)
        grad = -J.T @ score  # (18,): gradient of neg-loglik

        # Approximate Hessian (Fisher information)
        # H ≈ J^T W J where W = diag(w)
        JtWJ = J.T @ (w[:, np.newaxis] * J)  # (18, 18)

        # Regularize for stability
        JtWJ += 1e-8 * np.eye(18)

        # Newton step
        try:
            newton_step = np.linalg.solve(JtWJ, -grad)
        except np.linalg.LinAlgError:
            newton_step = -np.linalg.lstsq(JtWJ, grad, rcond=None)[0]

        # Line search: ensure deviance decreases
        step = 1.0
        theta_new = theta + step * newton_step
        for _ in range(10):
            params_new = _theta_to_params(theta_new, rho, 0.0, 0.0, phi, stratum, a_bar=a_bar)
            mu_new = compute_mu(params_new, a_obs, sample_meta, x_base_j, r_slice_j,
                               cached_arrays=cached_arrays)
            dev_new = tweedie_total_deviance(y[pos_mask], mu_new[pos_mask], p)
            if dev_new < prev_dev or step < 1e-4:
                break
            step *= 0.5
            theta_new = theta + step * newton_step

        theta = theta_new

        # Apply Group Lasso proximal step to δ groups (no Ā_k in threshold)
        for k in range(5):
            delta_k = theta[3 + k * 3: 3 + (k + 1) * 3]
            theta[3 + k * 3: 3 + (k + 1) * 3] = group_lasso_prox(
                delta_k, lambda_q, omega[k], eta, step_size=step,
            )

        # Compute current deviance (convert δ→β for mu computation)
        # This mu is reused at the top of the next iteration
        params = _theta_to_params(theta, rho, 0.0, 0.0, phi, stratum, a_bar=a_bar)
        mu = compute_mu(params, a_obs, sample_meta, x_base_j, r_slice_j,
                        cached_arrays=cached_arrays)
        dev = tweedie_total_deviance(y[pos_mask], mu[pos_mask], p)

        # Check convergence
        if prev_dev < np.inf:
            rel_change = abs(prev_dev - dev) / (abs(prev_dev) + 1e-10)
            if rel_change < tol:
                converged = True
                break
        prev_dev = dev

    # Build final params (convert δ→β)
    final_params = _theta_to_params(theta, rho, 0.0, 0.0, phi, stratum, a_bar=a_bar)
    return final_params, converged, float(dev)


def _compute_jacobian(
    theta: np.ndarray,
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    rho: float,
    stratum: int,
    a_bar: Optional[np.ndarray] = None,
    cached_arrays: Optional["SampleArrays"] = None,
) -> np.ndarray:
    """Compute dmu/dtheta Jacobian (24 × 18) analytically with active-set logic.

    In δ-space (when a_bar is provided), the Jacobian entries for the factorial
    columns use A_{i,k}/Ā_k instead of A_{i,k}, since dmu/dδ_k = A_{i,k}/Ā_k.

    The non-negativity projection S = max(S_unconstrained, 0) creates kinks
    in the objective. We use the active-set approach (Bertsekas, 1999):

    - If S_{k,i} > 0 (free): dS_{k,i}/dtheta = unconstrained derivative
    - If S_{k,i} = 0 (clamped at boundary): dS_{k,i}/dtheta = 0
    """
    # Convert θ (δ-space) to β-space for S-matrix computation
    params = _theta_to_params(theta, rho, 0.0, 0.0, 1.0, stratum, a_bar=a_bar)
    S_unconstrained = _compute_s_unconstrained(
        params, sample_meta, x_base_j, r_slice_j,
        cached_arrays=cached_arrays,
    )
    active = S_unconstrained > 0  # (6, 24): True = free, False = clamped

    sa = SampleArrays.ensure(cached_arrays, sample_meta, a_obs)
    n_samples = sa.A.shape[0]

    # A_active[i, k] = A[i, k] if S_{k,i} is free, else 0
    A_active = sa.A * active.T  # (n_samples, 6)

    # Global covariates: dmu_i/dalpha = sum_k A_active[i,k] * indicator[i]
    # sum over all 6 cell types
    a_sum = A_active.sum(axis=1)  # (n_samples,)

    J = np.zeros((n_samples, 18))
    J[:, 0] = a_sum * sa.female_ind
    J[:, 1] = a_sum * sa.time_4mo
    J[:, 2] = a_sum * sa.time_6mo

    # Factorial columns: dmu_i/ddelta_{k,f} = (A[i,k]/a_bar[k]) * fact[i,f] if active
    # Only for the 5 estimated cell types (k < 5)
    for k in range(5):
        if a_bar is not None:
            a_ratio_k = A_active[:, k] / max(a_bar[k], 1e-10)  # (n_samples,)
        else:
            a_ratio_k = A_active[:, k]
        # (n_samples, 3): outer product of a_ratio_k with each factorial indicator
        J[:, 3 + k * 3: 3 + (k + 1) * 3] = a_ratio_k[:, np.newaxis] * sa.fact_indicators

    return J


def _compute_s_unconstrained(
    params: SiteParams,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    cached_arrays: Optional["SampleArrays"] = None,
) -> np.ndarray:
    """Compute S_{k,c(i),j} WITHOUT non-negativity projection.

    Used by the active-set Jacobian to identify clamped components.
    """
    return _compute_s_core(params, sample_meta, x_base_j, r_slice_j,
                           project=False, cached_arrays=cached_arrays)


# ============================================================================
# Hurdle component fitting (§3.1)
# ============================================================================

def fit_hurdle_logistic(
    y: np.ndarray,
    mu: np.ndarray,
) -> Tuple[float, float]:
    """Fit dropout logistic regression: logit(pi) = gamma_0 + gamma_1 * log(mu).

    Simple Newton-Raphson for 2 parameters. Decoupled from the continuous
    component — uses current mu estimates as fixed predictors.

    Returns: (gamma0, gamma1)
    """
    z = (y <= 0).astype(float)  # binary: 1 = dropout
    x = np.log(np.maximum(mu, 1e-10))  # predictor

    # Newton-Raphson for logistic regression
    gamma = np.zeros(2)  # [intercept, slope]
    X = np.column_stack([np.ones_like(x), x])  # (24, 2)

    for _ in range(25):
        eta = X @ gamma
        eta = np.clip(eta, -20.0, 20.0)
        pi = 1.0 / (1.0 + np.exp(-eta))

        # Gradient and Hessian
        residual = z - pi
        W = pi * (1.0 - pi)
        grad = X.T @ residual
        H = X.T @ (W[:, np.newaxis] * X)
        H += 1e-8 * np.eye(2)

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        gamma += delta
        if np.linalg.norm(delta) < 1e-8:
            break

    return float(gamma[0]), float(gamma[1])


# ============================================================================
# Profile likelihood: rho update (§3.2.1)
# ============================================================================

def update_rho(
    y: np.ndarray,
    params: SiteParams,
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    p: float,
    lambda_rho: float,
    cached_arrays: Optional["SampleArrays"] = None,
) -> float:
    """One Newton step for rho_j under ridge penalty (§3.2.1).

    Maximizes: L(rho) - lambda_rho * rho^2
    via gradient: dL/drho - 2*lambda_rho*rho
    and Hessian: d2L/drho2 - 2*lambda_rho
    """
    pos_mask = y > 0
    if not np.any(pos_mask):
        return 0.0

    sa = SampleArrays.ensure(cached_arrays, sample_meta, a_obs)
    mu, S = _compute_mu_and_s(params, a_obs, sample_meta, x_base_j, r_slice_j,
                              cached_arrays=sa)

    # dmu/drho: for each sample i, dmu_i/drho = sum_k A_{i,k} * r_{k,i,j}
    # (only if S_{k,i,j} > 0 after projection; otherwise derivative is 0)
    active = S > 0  # (6, 24): which (k,i) are not at the boundary
    # Vectorized: sum_k A[i,k] * r[k,i] * active[k,i]
    dmu_drho = np.sum(sa.A * (active * r_slice_j).T, axis=1)  # (24,)

    # Tweedie score w.r.t. mu, then chain rule
    score = np.where(pos_mask,
                     (y - mu) / (params.phi * np.power(mu, p - 1.0)),
                     0.0)

    grad = float(np.sum(score * dmu_drho)) - 2.0 * lambda_rho * params.rho

    # Approximate Hessian (expected Fisher)
    w = np.where(pos_mask, np.power(mu, 2.0 - p) / params.phi, 0.0)
    hess = -float(np.sum(w * dmu_drho ** 2)) - 2.0 * lambda_rho

    if abs(hess) < 1e-15:
        return params.rho

    rho_new = params.rho - grad / hess
    return float(rho_new)


# ============================================================================
# Dispersion update (§3.5)
# ============================================================================

def update_phi(
    y: np.ndarray,
    mu: np.ndarray,
    p: float,
    n_eff_params: float,
) -> float:
    """Pearson estimator for site-specific dispersion phi_j.

    phi = sum_i (y_i - mu_i)^2 / V(mu_i)  /  (n_pos - p_eff)

    Only computed over positive observations (hurdle handles zeros).
    """
    pos_mask = y > 0
    n_pos = int(np.sum(pos_mask))
    dof = max(n_pos - n_eff_params, 1.0)

    if n_pos == 0:
        return 1.0

    y_pos = y[pos_mask]
    mu_pos = mu[pos_mask]
    v = tweedie_variance(mu_pos, p)

    pearson = np.sum((y_pos - mu_pos) ** 2 / v) / dof
    return max(float(pearson), 1e-6)  # floor for stability


# ============================================================================
# Profile likelihood: global Tweedie power p (§3.5)
# ============================================================================

def profile_tweedie_power(
    y_all: np.ndarray,
    mu_all: np.ndarray,
    phi_all: np.ndarray,
    p_grid: Optional[List[float]] = None,
) -> float:
    """Select global Tweedie power p by profile likelihood across all sites.

    Args:
        y_all: (n_sites, 24) observed intensities.
        mu_all: (n_sites, 24) predicted intensities.
        phi_all: (n_sites,) site-specific dispersions.
        p_grid: candidate p values (default from config).

    Returns:
        Optimal p value.
    """
    if p_grid is None:
        p_grid = config.TWEEDIE_P_GRID

    best_p = p_grid[0]
    best_ll = -np.inf

    # Pre-compute masks and filter out all-zero sites
    pos_mask = y_all > 0  # (n_sites, 24)
    has_positive = pos_mask.any(axis=1)  # (n_sites,)
    active_idx = np.where(has_positive)[0]

    mu_clamped = np.maximum(mu_all, 1e-10)

    for p in p_grid:
        # Batch saddlepoint log-likelihood across all sites and observations
        # For positive obs: -d/(2*phi) - 0.5*log(2*pi*phi*v_y)
        # For zero obs: -mu^{2-p} / (phi*(2-p))
        total_ll = 0.0
        for j in active_idx:
            pm = pos_mask[j]
            y_pos = y_all[j, pm]
            mu_pos = mu_clamped[j, pm]
            phi_j = phi_all[j]

            d = tweedie_deviance(y_pos, mu_pos, p)
            v_y = tweedie_variance(y_pos, p)
            total_ll += float(np.sum(
                -d / (2.0 * phi_j) - 0.5 * np.log(2.0 * np.pi * phi_j * v_y)
            ))

        if total_ll > best_ll:
            best_ll = total_ll
            best_p = p

    return best_p


# ============================================================================
# Single-site fitting: outer loop (§3.5)
# ============================================================================

def fit_site(
    j: int,
    y: np.ndarray,
    a_obs: pd.DataFrame,
    sample_meta: pd.DataFrame,
    x_base_j: np.ndarray,
    r_slice_j: np.ndarray,
    p: float,
    lambda_q: float,
    lambda_rho: float,
    omega: np.ndarray,
    a_bar: np.ndarray,
    eta: float,
    stratum: int,
    max_outer_iter: int = None,
    outer_tol: float = None,
    params_init: Optional[SiteParams] = None,
    gamma1_fixed: Optional[float] = None,
    design_matrix: Optional[np.ndarray] = None,
    cached_arrays: Optional["SampleArrays"] = None,
) -> Tuple[SiteParams, bool]:
    """Fit the Hurdle-Tweedie model for a single site j.

    Internally operates in δ-space (δ = Ā_k · β) for the IRLS inner loop,
    then converts back to β-space for the returned SiteParams.

    Alternates between:
    1. IRLS inner loop (alpha, delta) in δ-space
    2. rho update (Newton step with ridge penalty)
    3. phi update (Pearson estimator)
    4. Hurdle logistic (gamma_0 site-specific; gamma_1 pooled if provided)

    Args:
        a_bar: (5,) mean proportion per estimated cell type. Used for δ-space
            reparametrization.
        params_init: Optional warm-start parameters (in β-space). Converted
            to δ-space internally.
        gamma1_fixed: If provided, the hurdle slope is fixed to this value
            (pooled within intensity stratum). Only gamma_0 is re-estimated
            per site.

    Returns: (fitted_params, converged) — params in β-space.
    """
    if max_outer_iter is None:
        max_outer_iter = config.OUTER_MAX_ITER
    if outer_tol is None:
        outer_tol = config.OUTER_TOL

    # Initialize (warm-start if provided)
    if params_init is not None:
        params = SiteParams(
            beta=params_init.beta.copy(),
            alpha_gen=params_init.alpha_gen,
            alpha_time=params_init.alpha_time.copy(),
            rho=params_init.rho,
            gamma0=params_init.gamma0,
            gamma1=params_init.gamma1,
            phi=params_init.phi,
            stratum=stratum,
        )
    else:
        params = SiteParams.zeros(stratum=stratum)

    # Initial phi from method-of-moments (cold start only)
    if params_init is None:
        pos_mask = y > 0
        if np.any(pos_mask):
            y_pos = y[pos_mask]
            mu_init = np.mean(y_pos)
            if mu_init > 0:
                var_est = np.var(y_pos)
                # V(Y) = phi * mu^p => phi ≈ var / mu^p
                params.phi = max(var_est / (mu_init ** p), 1e-6)

    prev_ll = -np.inf
    # Convert params (β-space) to θ (δ-space) for IRLS
    theta = _params_to_theta(params, a_bar=a_bar)
    outer_converged = False

    for outer_iter in range(max_outer_iter):
        # 1. Inner loop: update alpha, delta (δ-space internally, returns β-space)
        params_inner, inner_conv, dev = irls_inner_loop(
            y, a_obs, sample_meta, x_base_j, r_slice_j,
            rho=params.rho, phi=params.phi, p=p,
            lambda_q=lambda_q, omega=omega, a_bar=a_bar, eta=eta,
            stratum=stratum, theta_init=theta,
            design_matrix=design_matrix,
            cached_arrays=cached_arrays,
        )
        params.beta = params_inner.beta  # already in β-space from IRLS
        params.alpha_gen = params_inner.alpha_gen
        params.alpha_time = params_inner.alpha_time
        theta = _params_to_theta(params, a_bar=a_bar)  # back to δ-space for next iter

        # 2. Update rho
        params.rho = update_rho(
            y, params, a_obs, sample_meta, x_base_j, r_slice_j, p, lambda_rho,
            cached_arrays=cached_arrays,
        )

        # 3. Update phi
        mu = compute_mu(params, a_obs, sample_meta, x_base_j, r_slice_j,
                        cached_arrays=cached_arrays)
        n_active = int(np.sum(np.abs(params.beta) > 1e-10))
        params.phi = update_phi(y, mu, p, n_eff_params=3.0 + n_active)

        # 4. Update hurdle (gamma_1 pooled by stratum if provided)
        if gamma1_fixed is not None:
            g0, _ = fit_hurdle_logistic(y, mu)
            params.gamma0 = g0
            params.gamma1 = gamma1_fixed
        else:
            params.gamma0, params.gamma1 = fit_hurdle_logistic(y, mu)

        # Check outer convergence (penalty in δ-space: no Ā_k factor)
        ll = hurdle_tweedie_loglik(y, mu, params.phi, p, params.gamma0, params.gamma1)
        penalty = lambda_rho * params.rho ** 2
        for k in range(5):
            # δ_k = Ā_k · β_k, so ||δ_k|| = Ā_k · ||β_k||
            delta_k_norm = a_bar[k] * np.linalg.norm(params.beta[k])
            penalty += lambda_q * omega[k] * np.sqrt(3.0) * delta_k_norm
        penalized_ll = ll - penalty

        if prev_ll > -np.inf:
            rel_change = abs(penalized_ll - prev_ll) / (abs(prev_ll) + 1e-10)
            if rel_change < outer_tol:
                outer_converged = True
                break
        prev_ll = penalized_ll

    return params, outer_converged


# ============================================================================
# Stratum-pooled hurdle slope (§3.1)
# ============================================================================

def _fit_stratum_pooled_gamma1(
    y_all: np.ndarray,
    A: np.ndarray,
    x_base_all: np.ndarray,
    strata: np.ndarray,
) -> np.ndarray:
    """Fit per-stratum pooled hurdle slope gamma_1.

    Pools all (y, mu_null) pairs within each intensity stratum and fits
    logistic regression: logit(pi) = gamma_0 + gamma_1 * log(mu).
    Returns the gamma_1 slope per stratum (gamma_0 is re-estimated per site).

    Args:
        y_all: (n_sites, 24) observed intensities (NaN already replaced with 0).
        A: (24, 6) cell-type composition matrix.
        x_base_all: (n_sites, 6) baseline intensities.
        strata: (n_sites,) intensity stratum indices.

    Returns:
        (N_INTENSITY_STRATA,) array of pooled gamma_1 slopes.
    """
    gamma1_by_stratum = np.zeros(config.N_INTENSITY_STRATA)
    for q in range(config.N_INTENSITY_STRATA):
        q_sites = np.where(strata == q)[0]
        if len(q_sites) == 0:
            continue
        # Pool all (y, mu_null) pairs across sites in this stratum
        z_pool = []
        log_mu_pool = []
        for jj in q_sites:
            mu_null = np.maximum(A @ x_base_all[jj], 1e-10)
            z_pool.append((y_all[jj] <= 0).astype(float))
            log_mu_pool.append(np.log(mu_null))
        z_pool = np.concatenate(z_pool)
        log_mu_pool = np.concatenate(log_mu_pool)
        # Fit logistic: logit(pi) = gamma_0 + gamma_1 * log(mu)
        X_pool = np.column_stack([np.ones_like(log_mu_pool), log_mu_pool])
        gamma_pool = np.zeros(2)
        for _ in range(25):
            eta_h = np.clip(X_pool @ gamma_pool, -20, 20)
            pi_h = 1.0 / (1.0 + np.exp(-eta_h))
            grad_h = X_pool.T @ (z_pool - pi_h)
            H_h = (X_pool.T @ ((pi_h * (1.0 - pi_h))[:, np.newaxis] * X_pool)
                   + 1e-8 * np.eye(2))
            try:
                delta_h = np.linalg.solve(H_h, grad_h)
            except np.linalg.LinAlgError:
                break
            gamma_pool += delta_h
            if np.linalg.norm(delta_h) < 1e-8:
                break
        gamma1_by_stratum[q] = gamma_pool[1]
    return gamma1_by_stratum


# ============================================================================
# Full model fitting (§3.5)
# ============================================================================

def _fit_site_worker(args: tuple) -> Tuple[int, SiteParams, bool]:
    """Module-level worker for ProcessPoolExecutor parallelism.

    Unpacks arguments and calls fit_site. Must be at module level for pickling.
    Returns (site_index, params, converged).
    """
    (j, y_j, a_obs, sample_meta, x_base_j, r_slice_j, p, lq,
     lambda_rho, omega, a_bar, eta, stratum_j, gamma1_fixed,
     design_matrix, cached_arrays) = args
    params_j, conv_j = fit_site(
        j, y_j, a_obs, sample_meta,
        x_base_j, r_slice_j, p, lq, lambda_rho,
        omega, a_bar, eta, stratum_j,
        gamma1_fixed=gamma1_fixed,
        design_matrix=design_matrix,
        cached_arrays=cached_arrays,
    )
    return j, params_j, conv_j


def _cv_site_worker(args: tuple) -> float:
    """Module-level worker for LOCO-CV site parallelism.

    Fits one site on training data, predicts on test data, returns deviance.
    """
    (j, y_train, y_test, train_a_obs, train_meta,
     test_a_obs, test_meta,
     x_base_j, r_train, r_test, p,
     lq, lambda_rho, omega, a_bar, eta, stratum_j,
     gamma1_fixed, design_matrix, sa_train, sa_test) = args

    params_j, _ = fit_site(
        j, y_train, train_a_obs, train_meta,
        x_base_j, r_train, p, lq, lambda_rho,
        omega, a_bar, eta, stratum_j,
        max_outer_iter=5,
        gamma1_fixed=gamma1_fixed,
        design_matrix=design_matrix,
        cached_arrays=sa_train,
    )

    mu_test = compute_mu(
        params_j, test_a_obs, test_meta,
        x_base_j, r_test, cached_arrays=sa_test,
    )
    pos_test = y_test > 0
    if np.any(pos_test):
        return float(tweedie_total_deviance(
            y_test[pos_test], mu_test[pos_test], p,
        ))
    return 0.0


def fit_hurdle_tweedie(
    data,  # SAPData (imported at call time to avoid circular imports)
    lambda_q: np.ndarray,
    lambda_rho: float,
    eta: float,
    gamma_cvs: float,
    n_workers: int = None,
    site_indices: Optional[np.ndarray] = None,
    p_init: float = 1.5,
) -> ModelFit:
    """Fit the full Hurdle-Tweedie model across all (or selected) sites.

    Args:
        data: SAPData with all Phase 0+1 fields populated.
        lambda_q: (Q,) per-stratum Group Lasso penalties.
        lambda_rho: ridge penalty on rho_j.
        eta: interaction penalty multiplier.
        gamma_cvs: CVS adaptive weight exponent.
        n_workers: parallelism (default from config).
        site_indices: subset of sites to fit (None = all).
        p_init: initial Tweedie power guess.

    Returns:
        ModelFit with estimated parameters for all sites.
    """
    if n_workers is None:
        n_workers = config.N_WORKERS

    n_sites = data.n_sites_filtered
    if site_indices is None:
        site_indices = np.arange(n_sites)

    # Compute adaptive weights and mean proportions
    omega = compute_adaptive_weights(data.cvs, gamma_cvs)
    a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values  # (5,)

    # Prepare bulk data as numpy
    y_all = data.bulk_phospho.values.copy()  # (n_sites, 24)
    # Replace NaN with 0 (treated as dropout by the hurdle)
    y_all = np.nan_to_num(y_all, nan=0.0)
    x_base_all = data.x_base.values  # (n_sites, 6)

    strata = data.intensity_strata
    if strata is None:
        from sap_data import compute_intensity_strata
        strata = compute_intensity_strata(data.x_base)

    # Pre-compute per-stratum hurdle slope gamma_1 (pooled across sites)
    gamma1_by_stratum = _fit_stratum_pooled_gamma1(
        y_all, data.a_obs.values, x_base_all, strata,
    )
    print(f"  Stratum-pooled gamma_1: {gamma1_by_stratum}")

    # Pre-compute design matrix and sample arrays (constant across all sites)
    D_cached = _build_inner_design(data.a_obs, data.sample_meta, a_bar)
    sa_cached = SampleArrays.from_data(data.sample_meta, data.a_obs)

    # Phase 1: fit each site independently
    all_params: List[Optional[SiteParams]] = [None] * n_sites
    converged = np.zeros(n_sites, dtype=bool)

    p = p_init

    # Build worker arguments for all sites
    worker_args = []
    for j in site_indices:
        lq = lambda_q[strata[j]]
        r_slice = data.r_tensor[:, :, j] if data.r_tensor is not None else np.zeros((6, 24))
        worker_args.append((
            j, y_all[j], data.a_obs, data.sample_meta,
            x_base_all[j], r_slice, p, lq, lambda_rho,
            omega, a_bar, eta, strata[j],
            float(gamma1_by_stratum[strata[j]]),
            D_cached, sa_cached,
        ))

    # Fit sites (parallel when n_workers > 1)
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for idx, (j, params_j, conv_j) in enumerate(
                pool.map(_fit_site_worker, worker_args),
            ):
                all_params[j] = params_j
                converged[j] = conv_j
                if (idx + 1) % 500 == 0:
                    n_conv = int(converged[site_indices[:idx + 1]].sum())
                    print(f"  Fitted {idx + 1}/{len(site_indices)} sites "
                          f"({n_conv} converged)")
    else:
        for idx, args_i in enumerate(worker_args):
            j, params_j, conv_j = _fit_site_worker(args_i)
            all_params[j] = params_j
            converged[j] = conv_j
            if (idx + 1) % 500 == 0:
                n_conv = int(converged[site_indices[:idx + 1]].sum())
                print(f"  Fitted {idx + 1}/{len(site_indices)} sites "
                      f"({n_conv} converged)")

    # Profile p using fitted mu values
    mu_all = np.zeros_like(y_all)
    phi_all = np.ones(n_sites)
    for j in site_indices:
        if all_params[j] is not None:
            r_slice = data.r_tensor[:, :, j] if data.r_tensor is not None else np.zeros((6, 24))
            mu_all[j] = compute_mu(
                all_params[j], data.a_obs, data.sample_meta,
                x_base_all[j], r_slice, cached_arrays=sa_cached,
            )
            phi_all[j] = all_params[j].phi

    p_new = profile_tweedie_power(
        y_all[site_indices], mu_all[site_indices], phi_all[site_indices],
    )
    print(f"  Profile p: {p_init} → {p_new}")

    # Collect binding stats
    binding_counts = np.zeros((6, 4))  # cell_types × conditions
    for j in site_indices:
        if all_params[j] is not None:
            r_slice = data.r_tensor[:, :, j] if data.r_tensor is not None else np.zeros((6, 24))
            S = compute_s_matrix(
                all_params[j], data.sample_meta, x_base_all[j], r_slice,
                cached_arrays=sa_cached,
            )
            # Count how many (k, condition) have S projected to 0
            for i, cond in enumerate(data.sample_meta["condition"].values):
                cond_idx = config.SAP_CONDITIONS.index(cond)
                for k in range(6):
                    if S[k, i] <= 0 and x_base_all[j, k] > 0:
                        binding_counts[k, cond_idx] += 1

    binding_df = pd.DataFrame(
        binding_counts,
        index=config.SAP_CELLTYPES,
        columns=config.SAP_CONDITIONS,
    )

    # Fill None params with zeros
    for j in range(n_sites):
        if all_params[j] is None:
            all_params[j] = SiteParams.zeros(stratum=strata[j])

    return ModelFit(
        site_params=all_params,
        p=p_new,
        converged=converged,
        binding_freq=binding_df,
    )


# ============================================================================
# LOCO-CV (§4.2)
# ============================================================================

def loco_cv(
    data,  # SAPData
    lambda_grid: np.ndarray,
    lambda_rho_grid: Optional[List[float]] = None,
    eta_grid: Optional[List[float]] = None,
    gamma_grid: Optional[List[float]] = None,
    p_init: float = 1.5,
    site_subsample: Optional[float] = None,
    n_workers: int = None,
) -> CVResult:
    """Leave-one-condition-out CV for hyperparameter selection (§4.2).

    Args:
        data: SAPData with Phase 1 fields.
        lambda_grid: (n_lambda,) candidate Group Lasso penalties (shared
                     across strata, then scaled by stratum-specific factor).
        lambda_rho_grid: ridge penalty candidates.
        eta_grid: interaction multiplier candidates.
        gamma_grid: CVS exponent candidates.
        p_init: initial Tweedie power.
        site_subsample: fraction of sites to use (None = all).
        n_workers: parallelism.

    Returns:
        CVResult with best hyperparameters.
    """
    if lambda_rho_grid is None:
        lambda_rho_grid = config.LAMBDA_RHO_GRID
    if eta_grid is None:
        eta_grid = config.ETA_GRID
    if gamma_grid is None:
        gamma_grid = config.GAMMA_GRID

    # Site subsampling for efficiency
    n_sites = data.n_sites_filtered
    if site_subsample is not None:
        n_sub = max(int(n_sites * site_subsample), 100)
        strata = data.intensity_strata
        # Stratified subsample
        site_indices = []
        for q in range(config.N_INTENSITY_STRATA):
            q_sites = np.where(strata == q)[0]
            n_q = max(int(len(q_sites) * site_subsample), 25)
            np.random.seed(42 + q)
            site_indices.extend(np.random.choice(q_sites, n_q, replace=False))
        site_indices = np.array(sorted(site_indices))
    else:
        site_indices = np.arange(n_sites)

    # Pre-compute per-stratum hurdle slope (pooled, full-data estimate)
    strata_cv = data.intensity_strata
    if strata_cv is None:
        from sap_data import compute_intensity_strata as _cis
        strata_cv = _cis(data.x_base)
    y_all_cv = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_cv = data.x_base.values
    gamma1_by_stratum = _fit_stratum_pooled_gamma1(
        y_all_cv, data.a_obs.values, x_base_cv, strata_cv,
    )

    conditions = config.SAP_CONDITIONS
    results = []

    for lam in lambda_grid:
        for lam_rho in lambda_rho_grid:
            for eta in eta_grid:
                for gamma in gamma_grid:
                    # Lambda vector: same penalty across strata for now
                    lam_q = np.full(config.N_INTENSITY_STRATA, lam)

                    fold_deviances = []
                    for held_out_cond in conditions:
                        # Split samples
                        train_mask = data.sample_meta["condition"].values != held_out_cond
                        test_mask = ~train_mask

                        # Create subset views
                        train_meta = data.sample_meta[train_mask]
                        train_a_obs = data.a_obs[train_mask]
                        test_a_obs = data.a_obs[test_mask]
                        test_meta = data.sample_meta[test_mask]

                        y_all = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
                        x_base_all = data.x_base.values

                        omega = compute_adaptive_weights(data.cvs, gamma)
                        a_bar = train_a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values

                        # Pre-compute design matrix and sample arrays for this fold
                        D_cv = _build_inner_design(train_a_obs, train_meta, a_bar)
                        sa_train = SampleArrays.from_data(train_meta, train_a_obs)
                        sa_test = SampleArrays.from_data(test_meta, test_a_obs)

                        # Build worker arguments for all sites in this fold
                        cv_worker_args = []
                        for j in site_indices:
                            r_slice = (data.r_tensor[:, :, j]
                                       if data.r_tensor is not None
                                       else np.zeros((6, 24)))
                            cv_worker_args.append((
                                j, y_all[j, train_mask], y_all[j, test_mask],
                                train_a_obs, train_meta,
                                test_a_obs, test_meta,
                                x_base_all[j], r_slice[:, train_mask],
                                r_slice[:, test_mask], p_init,
                                lam_q[data.intensity_strata[j]],
                                lam_rho, omega, a_bar, eta,
                                data.intensity_strata[j],
                                float(gamma1_by_stratum[strata_cv[j]]),
                                D_cv, sa_train, sa_test,
                            ))

                        # Fit sites (parallel when n_workers > 1)
                        if n_workers is not None and n_workers > 1:
                            from concurrent.futures import ProcessPoolExecutor
                            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                                fold_dev = sum(pool.map(_cv_site_worker, cv_worker_args))
                        else:
                            fold_dev = sum(
                                _cv_site_worker(a) for a in cv_worker_args
                            )

                        fold_deviances.append(fold_dev)

                    mean_dev = np.mean(fold_deviances)
                    results.append({
                        "lambda": lam, "lambda_rho": lam_rho,
                        "eta": eta, "gamma": gamma,
                        "mean_deviance": mean_dev,
                    })
                    print(f"  CV: λ={lam:.4f} λ_ρ={lam_rho:.3f} η={eta:.1f} "
                          f"γ={gamma:.1f} → dev={mean_dev:.2f}")

    cv_df = pd.DataFrame(results)
    best_row = cv_df.loc[cv_df["mean_deviance"].idxmin()]

    best_lam = float(best_row["lambda"])
    best_lam_q = np.full(config.N_INTENSITY_STRATA, best_lam)

    return CVResult(
        best_lambda=best_lam_q,
        best_lambda_rho=float(best_row["lambda_rho"]),
        best_eta=float(best_row["eta"]),
        best_gamma=float(best_row["gamma"]),
        selected_p=p_init,
        cv_scores=cv_df,
    )


def loco_cv_two_stage(
    data,  # SAPData
    lambda_grid: np.ndarray,
    lambda_rho_grid: Optional[List[float]] = None,
    eta_grid: Optional[List[float]] = None,
    gamma_grid: Optional[List[float]] = None,
    p_init: float = 1.5,
    site_subsample: Optional[float] = None,
    n_workers: int = None,
) -> CVResult:
    """Two-stage LOCO-CV for hyperparameter selection (§4.2).

    Stage 1: search (λ, λ_ρ) with fixed η and γ defaults.
    Stage 2: search (η, γ) with best (λ*, λ_ρ*) from stage 1.

    This reduces the grid from O(|λ|·|λ_ρ|·|η|·|γ|) to
    O(|λ|·|λ_ρ| + |η|·|γ|), typically 240 → 40 combos.
    """
    if lambda_rho_grid is None:
        lambda_rho_grid = config.LAMBDA_RHO_GRID
    if eta_grid is None:
        eta_grid = config.ETA_GRID
    if gamma_grid is None:
        gamma_grid = config.GAMMA_GRID

    n_stage1 = len(lambda_grid) * len(lambda_rho_grid)
    n_stage2 = len(eta_grid) * len(gamma_grid)
    print(f"\n  Two-stage CV: stage 1 = {n_stage1} combos (λ × λ_ρ), "
          f"stage 2 = {n_stage2} combos (η × γ)")

    # Stage 1: search (λ, λ_ρ) with fixed η, γ
    print(f"\n--- Stage 1: searching (λ, λ_ρ) with η={config.LOCO_STAGE1_ETA_DEFAULT}, "
          f"γ={config.LOCO_STAGE1_GAMMA_DEFAULT} ---")
    stage1_result = loco_cv(
        data, lambda_grid,
        lambda_rho_grid=lambda_rho_grid,
        eta_grid=[config.LOCO_STAGE1_ETA_DEFAULT],
        gamma_grid=[config.LOCO_STAGE1_GAMMA_DEFAULT],
        p_init=p_init,
        site_subsample=site_subsample,
        n_workers=n_workers,
    )
    print(f"  Stage 1 best: λ={stage1_result.best_lambda[0]:.4f}, "
          f"λ_ρ={stage1_result.best_lambda_rho:.4f}")

    # Stage 2: search (η, γ) with best (λ*, λ_ρ*)
    best_lam = float(stage1_result.best_lambda[0])
    best_lam_rho = stage1_result.best_lambda_rho
    print(f"\n--- Stage 2: searching (η, γ) with λ={best_lam:.4f}, "
          f"λ_ρ={best_lam_rho:.4f} ---")
    stage2_result = loco_cv(
        data, np.array([best_lam]),
        lambda_rho_grid=[best_lam_rho],
        eta_grid=eta_grid,
        gamma_grid=gamma_grid,
        p_init=p_init,
        site_subsample=site_subsample,
        n_workers=n_workers,
    )
    print(f"  Stage 2 best: η={stage2_result.best_eta:.1f}, "
          f"γ={stage2_result.best_gamma:.1f}")

    # Combine: best (λ, λ_ρ) from stage 1, best (η, γ) from stage 2
    combined_scores = pd.concat(
        [stage1_result.cv_scores, stage2_result.cv_scores],
        ignore_index=True,
    )

    return CVResult(
        best_lambda=stage1_result.best_lambda,
        best_lambda_rho=stage1_result.best_lambda_rho,
        best_eta=stage2_result.best_eta,
        best_gamma=stage2_result.best_gamma,
        selected_p=p_init,
        cv_scores=combined_scores,
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAP Hurdle-Tweedie model fitting",
    )
    parser.add_argument("--fit", action="store_true",
                        help="Full LOCO-CV + model fit")
    parser.add_argument("--fit-fast", action="store_true",
                        help="Coarse grid, site subsampling (dev/debug)")
    parser.add_argument("--fit-site", type=int, default=None,
                        help="Fit a single site (debugging)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Checkpoint 1: fit a few sites per stratum with timing")
    parser.add_argument("--self-test", action="store_true",
                        help="Run Tweedie utility self-tests")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Parallel workers (default: config.N_WORKERS)")
    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        return

    if args.smoke_test:
        _smoke_test_cli()
        return

    if args.fit_site is not None:
        _fit_single_site_cli(args.fit_site)
        return

    if args.fit or args.fit_fast:
        _fit_model_cli(fast=args.fit_fast, n_workers=args.n_workers)
        return

    parser.print_help()


def _smoke_test_cli() -> None:
    """Checkpoint 1: fit a small sample of sites with timing diagnostics.

    Picks 3 sites per intensity stratum (12 total), fits each, and reports
    per-site wall time, convergence, sparsity, and predicted mu range.
    Prints a timing projection for LOCO-CV and full fit.
    """
    from sap_data import load_all, compute_intensity_strata

    print("=" * 70)
    print("CHECKPOINT 1: Smoke Test")
    print("=" * 70)

    print("\nLoading data (Phase 1)...")
    t0_load = time.perf_counter()
    data, report = load_all(include_rna=True)
    load_time = time.perf_counter() - t0_load
    print(f"  Data loaded in {load_time:.1f}s")

    if not report.all_passed:
        print("\nWARNING: Not all diagnostics passed.")
        print(report.summary())

    strata = compute_intensity_strata(data.x_base)
    omega = compute_adaptive_weights(data.cvs, gamma=0.5)
    a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values  # (5,)

    # Pre-compute per-stratum hurdle slope
    y_all_pre = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_pre = data.x_base.values
    gamma1_by_stratum = _fit_stratum_pooled_gamma1(
        y_all_pre, data.a_obs.values, x_base_pre, strata,
    )
    print(f"  Stratum-pooled gamma_1: {gamma1_by_stratum}")

    n_sites = data.n_sites_filtered
    n_per_stratum = 3
    rng = np.random.RandomState(42)

    # Pick sites: 3 per stratum (stratified sample)
    test_sites = []
    for q in range(config.N_INTENSITY_STRATA):
        q_sites = np.where(strata == q)[0]
        chosen = rng.choice(q_sites, min(n_per_stratum, len(q_sites)), replace=False)
        test_sites.extend(chosen)
    test_sites = sorted(test_sites)

    # Compute data-scaled lambda via lambda_max calibration
    print(f"\nComputing lambda_max from score statistics at beta=0...")
    lam_max = compute_lambda_max(
        data, omega, a_bar, p=1.5, n_sites_sample=100, seed=42,
    )
    lam_test = lam_max * 0.01  # 1% of lambda_max: moderate sparsity
    print(f"  lambda_max = {lam_max:.2f}")
    print(f"  lambda_test = {lam_test:.2f} (1% of lambda_max)")

    print(f"\nFitting {len(test_sites)} sites ({n_per_stratum} per stratum)...")
    print(f"Total sites in dataset: {n_sites}")
    print()

    y_all = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_all = data.x_base.values
    sa_cached = SampleArrays.from_data(data.sample_meta, data.a_obs)

    timings = []
    results = []

    for idx, j in enumerate(test_sites):
        y = y_all[j]
        x_base_j = x_base_all[j]
        r_slice_j = (data.r_tensor[:, :, j]
                     if data.r_tensor is not None
                     else np.zeros((6, 24)))
        n_zeros = int(np.sum(y == 0))

        t_start = time.perf_counter()
        params, conv = fit_site(
            j, y, data.a_obs, data.sample_meta,
            x_base_j, r_slice_j, p=1.5,
            lambda_q=lam_test, lambda_rho=0.1,
            omega=omega, a_bar=a_bar, eta=2.0, stratum=strata[j],
            gamma1_fixed=float(gamma1_by_stratum[strata[j]]),
            cached_arrays=sa_cached,
        )
        elapsed = time.perf_counter() - t_start
        timings.append(elapsed)

        mu = compute_mu(params, data.a_obs, data.sample_meta, x_base_j, r_slice_j)

        n_active = sum(1 for k in range(5)
                       if np.linalg.norm(params.beta[k]) > 1e-6)
        beta_norms = [np.linalg.norm(params.beta[k]) for k in range(5)]
        delta_norms = [a_bar[k] * np.linalg.norm(params.beta[k]) for k in range(5)]

        results.append({
            "site": j, "stratum": strata[j], "converged": conv,
            "time_s": elapsed, "n_zeros": n_zeros,
            "n_active_ct": n_active, "rho": params.rho, "phi": params.phi,
            "mu_min": mu.min(), "mu_max": mu.max(),
            "beta_norms": beta_norms, "delta_norms": delta_norms,
        })

        conv_str = "CONV" if conv else "FAIL"
        print(f"  Site {j:5d}  stratum={strata[j]}  {elapsed:6.3f}s  "
              f"{conv_str}  zeros={n_zeros:2d}/24  "
              f"active_ct={n_active}/5  "
              f"rho={params.rho:+.4f}  phi={params.phi:.4f}  "
              f"mu=[{mu.min():.3f}, {mu.max():.3f}]")

    # Summary statistics
    times = np.array(timings)
    print(f"\n{'=' * 70}")
    print("TIMING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Per-site: mean={times.mean():.3f}s  "
          f"median={np.median(times):.3f}s  "
          f"min={times.min():.3f}s  max={times.max():.3f}s")
    print(f"  Std: {times.std():.3f}s")

    n_conv = sum(1 for r in results if r["converged"])
    print(f"\n  Convergence: {n_conv}/{len(results)}")
    n_active_list = [r["n_active_ct"] for r in results]
    print(f"  Active cell types: mean={np.mean(n_active_list):.1f}  "
          f"range=[{min(n_active_list)}, {max(n_active_list)}]")
    rho_vals = [r["rho"] for r in results]
    print(f"  rho: mean={np.mean(rho_vals):.4f}  "
          f"range=[{min(rho_vals):.4f}, {max(rho_vals):.4f}]")
    phi_vals = [r["phi"] for r in results]
    print(f"  phi: median={np.median(phi_vals):.4f}  "
          f"range=[{min(phi_vals):.4f}, {max(phi_vals):.4f}]")

    print(f"\n  lambda_max: {lam_max:.2f}")
    print(f"  lambda_test (1% of max): {lam_test:.2f}")
    lam_grid_rec = np.logspace(np.log10(lam_max / 100), np.log10(lam_max), 8)
    print(f"  Recommended LOCO-CV grid: {np.array2string(lam_grid_rec, precision=1)}")

    # Projections
    t_per_site = np.median(times)
    print(f"\n{'=' * 70}")
    print("COMPUTE PROJECTIONS (based on median per-site time)")
    print(f"{'=' * 70}")

    # LOCO-CV grids (two-stage: stage1 = λ×λ_ρ, stage2 = η×γ)
    n_lambda_full = config.LAMBDA_GRID_FULL_N
    n_lambda_fast = config.LAMBDA_GRID_FAST_N
    n_lam_rho = len(config.LAMBDA_RHO_GRID)
    n_eta = len(config.ETA_GRID)
    n_gamma = len(config.GAMMA_GRID)
    n_folds = 4  # LOCO-CV: 4 conditions

    combos_full = n_lambda_full * n_lam_rho + n_eta * n_gamma  # two-stage
    combos_fast = n_lambda_fast * n_lam_rho + n_eta * n_gamma  # two-stage

    n_sub = max(int(n_sites * 0.2), 100)
    n_workers = config.N_WORKERS

    # Fast mode: CV on subsample + full fit
    cv_fast_s = combos_fast * n_folds * n_sub * t_per_site / n_workers
    fit_fast_s = n_sites * t_per_site / n_workers
    total_fast_s = cv_fast_s + fit_fast_s

    # Full mode: CV on all sites + full fit
    cv_full_s = combos_full * n_folds * n_sites * t_per_site / n_workers
    fit_full_s = n_sites * t_per_site / n_workers
    total_full_s = cv_full_s + fit_full_s

    def _fmt_time(seconds: float) -> str:
        if seconds < 3600:
            return f"{seconds / 60:.0f} min"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f} hours"
        else:
            return f"{seconds / 86400:.1f} days"

    print(f"\n  Per-site time: {t_per_site:.3f}s")
    print(f"  Total sites: {n_sites}")
    print(f"  CV subsample (20%): {n_sub}")
    print(f"  Parallel workers: {n_workers}")
    print()
    print(f"  --fit-fast ({combos_fast} combos × {n_folds} folds × {n_sub} sites ÷ {n_workers} workers):")
    print(f"    CV phase:    {_fmt_time(cv_fast_s)}")
    print(f"    Final fit:   {_fmt_time(fit_fast_s)}")
    print(f"    Total:       {_fmt_time(total_fast_s)}")
    print()
    print(f"  --fit ({combos_full} combos × {n_folds} folds × {n_sites} sites ÷ {n_workers} workers):")
    print(f"    CV phase:    {_fmt_time(cv_full_s)}")
    print(f"    Final fit:   {_fmt_time(fit_full_s)}")
    print(f"    Total:       {_fmt_time(total_full_s)}")

    # Feasibility warning
    if total_fast_s > 48 * 3600:
        print(f"\n  ⚠ Even --fit-fast exceeds 48 hours. Consider reducing "
              f"the grid or increasing N_WORKERS.")
    elif total_full_s > 48 * 3600:
        print(f"\n  ⚠ --fit exceeds 48 hours. Consider fixing η/γ from fast fit "
              f"and searching only (λ, λ_ρ), or increasing N_WORKERS.")

    # Cell-type proportions
    print(f"\n  Mean proportions (a_bar): "
          + "  ".join(f"{ct[:8]}={a_bar[k]:.4f}"
                      for k, ct in enumerate(config.SAP_ESTIMATED_CELLTYPES)))

    # Per-cell-type detail: δ norms (bulk-space, optimization units)
    print(f"\n{'=' * 70}")
    print("PER-SITE DETAIL: Delta norms (bulk-space) by cell type")
    print(f"{'=' * 70}")
    ct_names = config.SAP_ESTIMATED_CELLTYPES
    header = f"  {'Site':>5s}  {'Strat':>5s}  "
    header += "  ".join(f"{ct[:8]:>8s}" for ct in ct_names)
    print(header)
    for r in results:
        row = f"  {r['site']:5d}  {r['stratum']:5d}  "
        row += "  ".join(f"{dn:8.4f}" for dn in r["delta_norms"])
        print(row)

    # Per-cell-type detail: β norms (cell-type-specific space)
    print(f"\n{'=' * 70}")
    print("PER-SITE DETAIL: Beta norms (cell-type space) by cell type")
    print(f"{'=' * 70}")
    header = f"  {'Site':>5s}  {'Strat':>5s}  "
    header += "  ".join(f"{ct[:8]:>8s}" for ct in ct_names)
    print(header)
    for r in results:
        row = f"  {r['site']:5d}  {r['stratum']:5d}  "
        row += "  ".join(f"{bn:8.4f}" for bn in r["beta_norms"])
        print(row)

    print(f"\n{'=' * 70}")
    print("Smoke test complete. Review results above before proceeding.")
    print(f"{'=' * 70}")


def _run_self_tests() -> None:
    """Self-tests for Tweedie utilities."""
    print("Tweedie utility self-tests")
    print("=" * 40)

    # Test 1: Deviance is non-negative, zero at y=mu
    y = np.array([1.0, 2.0, 5.0, 0.0, 0.5])
    mu = np.array([1.0, 2.0, 5.0, 0.1, 0.5])
    for p in [1.2, 1.5, 1.8]:
        d = tweedie_deviance(y, mu, p)
        assert np.all(d >= -1e-10), f"Deviance negative at p={p}: {d}"
        # At y=mu (nonzero), deviance should be ~0
        assert d[0] < 1e-10, f"Deviance at y=mu nonzero: {d[0]} at p={p}"
        assert d[4] < 1e-10, f"Deviance at y=mu nonzero: {d[4]} at p={p}"
    print("  [PASS] Deviance non-negative and zero at y=mu")

    # Test 2: Variance function
    mu_test = np.array([0.5, 1.0, 2.0, 10.0])
    for p in [1.2, 1.5, 1.8]:
        v = tweedie_variance(mu_test, p)
        expected = mu_test ** p
        assert np.allclose(v, expected), f"Variance mismatch at p={p}"
    print("  [PASS] Variance function V(mu) = mu^p")

    # Test 3: Log-likelihood is finite
    y = np.array([0.0, 0.0, 1.5, 3.2, 0.8])
    mu = np.array([0.5, 0.3, 1.2, 3.0, 1.0])
    for p in [1.2, 1.5, 1.8]:
        ll = tweedie_log_likelihood(y, mu, phi=1.0, p=p)
        assert np.isfinite(ll), f"Log-likelihood not finite at p={p}: {ll}"
    print("  [PASS] Log-likelihood finite for mixed zero/positive data")

    # Test 4: Deviance residuals
    dr = tweedie_deviance_residuals(y, mu, 1.5)
    assert len(dr) == len(y), "Deviance residuals wrong length"
    assert np.all(np.isfinite(dr)), "Deviance residuals contain non-finite values"
    print("  [PASS] Deviance residuals finite")

    # Test 5: Group Lasso proximal operator
    beta = np.array([1.0, 0.5, 0.3])
    # Large threshold → zero
    result = group_lasso_prox(beta, lambda_q=100.0, omega_k=1.0, eta=2.0)
    assert np.allclose(result, 0), f"Prox with large lambda should be zero: {result}"
    # Zero threshold → identity
    result = group_lasso_prox(beta, lambda_q=0.0, omega_k=1.0, eta=2.0)
    assert np.allclose(result, beta), f"Prox with zero lambda should be identity: {result}"
    print("  [PASS] Group Lasso proximal operator")

    # Test 6: Adaptive weights
    cvs_df = pd.DataFrame(
        {"AppP": [0.4, 0.5], "Ttau": [0.3, 0.4], "ApTt": [0.5, 0.6]},
        index=["ct1", "ct2"],
    )
    w0 = compute_adaptive_weights(cvs_df, gamma=0)
    assert np.allclose(w0, 1.0), "gamma=0 should give uniform weights"
    w1 = compute_adaptive_weights(cvs_df, gamma=1.0)
    assert w1[0] > w1[1], "Lower CVS should get higher weight"
    print("  [PASS] Adaptive weights")

    print("\nAll self-tests passed.")


def _fit_single_site_cli(site_idx: int) -> None:
    """Fit a single site for debugging."""
    from sap_data import load_all, compute_intensity_strata

    print(f"Loading data (Phase 1)...")
    data, report = load_all(include_rna=True)

    if site_idx >= data.n_sites_filtered:
        print(f"Error: site {site_idx} out of range (max {data.n_sites_filtered - 1})")
        sys.exit(1)

    strata = compute_intensity_strata(data.x_base)
    omega = compute_adaptive_weights(data.cvs, gamma=0.5)
    a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values  # (5,)

    y = np.nan_to_num(data.bulk_phospho.values[site_idx], nan=0.0)
    x_base_j = data.x_base.values[site_idx]
    r_slice_j = data.r_tensor[:, :, site_idx] if data.r_tensor is not None else np.zeros((6, 24))
    sa_cached = SampleArrays.from_data(data.sample_meta, data.a_obs)

    print(f"\nFitting site {site_idx} (stratum={strata[site_idx]})...")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}], zeros: {np.sum(y == 0)}/24")
    print(f"  x_base: [{x_base_j.min():.4f}, {x_base_j.max():.4f}]")

    params, conv = fit_site(
        site_idx, y, data.a_obs, data.sample_meta,
        x_base_j, r_slice_j, p=1.5,
        lambda_q=0.1, lambda_rho=0.1,
        omega=omega, a_bar=a_bar, eta=2.0, stratum=strata[site_idx],
        cached_arrays=sa_cached,
    )

    print(f"\n  Converged: {conv}")
    print(f"  rho: {params.rho:.6f}")
    print(f"  phi: {params.phi:.6f}")
    print(f"  gamma: ({params.gamma0:.4f}, {params.gamma1:.4f})")
    print(f"  alpha_gen: {params.alpha_gen:.6f}")
    print(f"  alpha_time: [{params.alpha_time[0]:.6f}, {params.alpha_time[1]:.6f}]")

    # Show which cell types have active beta
    for k, ct in enumerate(config.SAP_ESTIMATED_CELLTYPES):
        norm = np.linalg.norm(params.beta[k])
        status = "ACTIVE" if norm > 1e-6 else "zero"
        print(f"  beta[{ct[:15]:15s}]: ||β||={norm:.6f} ({status}) "
              f"  App={params.beta[k, 0]:.4f} Tau={params.beta[k, 1]:.4f} "
              f"Int={params.beta[k, 2]:.4f}")

    mu = compute_mu(params, data.a_obs, data.sample_meta, x_base_j, r_slice_j)
    dev = tweedie_total_deviance(y[y > 0], mu[y > 0], 1.5) if np.any(y > 0) else 0.0
    print(f"\n  Predicted mu range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  Total deviance (positive obs): {dev:.4f}")


def _fit_model_cli(fast: bool = False, n_workers: Optional[int] = None) -> None:
    """Full model fitting via CLI (Checkpoint 2/4)."""
    from sap_data import load_all

    print("Loading data (Phase 1)...")
    data, report = load_all(include_rna=True)

    if not report.all_passed:
        print("\nWARNING: Not all diagnostics passed. Proceeding anyway.")
        print(report.summary())

    # Calibrate λ grid from data
    omega_init = compute_adaptive_weights(data.cvs, config.LOCO_STAGE1_GAMMA_DEFAULT)
    a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values
    lam_max = compute_lambda_max(data, omega_init, a_bar, p=1.5)

    if fast:
        print(f"\n--- Fast mode: two-stage CV, 20% site subsampling ---")
        lambda_grid = np.logspace(
            np.log10(lam_max / 100), np.log10(lam_max),
            config.LAMBDA_GRID_FAST_N,
        )
        subsample = 0.2
    else:
        print(f"\n--- Full LOCO-CV ---")
        lambda_grid = np.logspace(
            np.log10(lam_max / 1000), np.log10(lam_max),
            config.LAMBDA_GRID_FULL_N,
        )
        subsample = None

    print(f"  λ_max = {lam_max:.2f}")
    print(f"  λ grid: [{lambda_grid[0]:.2f}, ..., {lambda_grid[-1]:.2f}] "
          f"({len(lambda_grid)} points)")

    print("\nRunning two-stage LOCO-CV...")
    cv_result = loco_cv_two_stage(
        data, lambda_grid,
        site_subsample=subsample,
        n_workers=n_workers,
    )

    print(f"\nBest hyperparameters:")
    print(f"  λ (per-stratum): {cv_result.best_lambda}")
    print(f"  λ_ρ: {cv_result.best_lambda_rho}")
    print(f"  η: {cv_result.best_eta}")
    print(f"  γ: {cv_result.best_gamma}")
    print(f"  p: {cv_result.selected_p}")

    # Check for boundary selections
    lam_best = float(cv_result.best_lambda[0])
    if np.isclose(lam_best, lambda_grid[0]) or np.isclose(lam_best, lambda_grid[-1]):
        print(f"  ⚠ WARNING: λ at grid boundary — consider extending the grid")
    if cv_result.best_lambda_rho in (config.LAMBDA_RHO_GRID[0], config.LAMBDA_RHO_GRID[-1]):
        print(f"  ⚠ WARNING: λ_ρ at grid boundary")
    if cv_result.best_eta in (config.ETA_GRID[0], config.ETA_GRID[-1]):
        print(f"  ⚠ WARNING: η at grid boundary")

    print("\nFitting full model with best hyperparameters...")
    model = fit_hurdle_tweedie(
        data,
        lambda_q=cv_result.best_lambda,
        lambda_rho=cv_result.best_lambda_rho,
        eta=cv_result.best_eta,
        gamma_cvs=cv_result.best_gamma,
        n_workers=n_workers,
    )
    model.cv_result = cv_result

    n_conv = int(model.converged.sum()) if model.converged is not None else 0
    print(f"\nConvergence: {n_conv}/{data.n_sites_filtered} sites")
    print(f"Global Tweedie power: {model.p}")

    if model.binding_freq is not None:
        print(f"\nNon-negativity binding frequency:")
        print(model.binding_freq.to_string())

    # Sparsity summary
    n_sites = len(model.site_params)
    active_counts = np.zeros(5)
    for sp in model.site_params:
        for k in range(5):
            if np.linalg.norm(sp.beta[k]) > 1e-6:
                active_counts[k] += 1
    print(f"\nSparsity pattern (fraction nonzero per cell type):")
    for k, ct in enumerate(config.SAP_ESTIMATED_CELLTYPES):
        print(f"  {ct:20s}: {active_counts[k]/n_sites:.1%} "
              f"({int(active_counts[k])}/{n_sites})")

    save_model(model)


if __name__ == "__main__":
    main()
