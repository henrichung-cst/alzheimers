"""
SAP Validation Suite: §5 (bMIND benchmark) and §6.1–§6.5 diagnostics.

Phase 3 of the SAP implementation. Validates the fitted Hurdle-Tweedie model
before downstream kinase enrichment.

Usage:
    python code/sap_validate.py --residual-orth          # §6.5 (fast, no refit)
    python code/sap_validate.py --cross-modality         # §6.2 (fast, no refit)
    python code/sap_validate.py --synthetic              # §6.1 all scenarios
    python code/sap_validate.py --synthetic --scenario sparse  # single scenario
    python code/sap_validate.py --pseudobulk             # §6.1.1
    python code/sap_validate.py --perturbation           # §6.3
    python code/sap_validate.py --permutation            # §6.4
    python code/sap_validate.py --bmind                  # §5
    python code/sap_validate.py --all                    # everything
    python code/sap_validate.py --summary                # print cached results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import config
from sap_model import (
    CVResult,
    ModelFit,
    SiteParams,
    compute_adaptive_weights,
    compute_mu,
    compute_s_matrix,
    fit_hurdle_tweedie,
    fit_site,
    load_model,
    tweedie_deviance,
    tweedie_deviance_residuals,
    tweedie_total_deviance,
    tweedie_variance,
)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str                          # e.g. "residual_orth", "cross_modality"
    passed: Optional[bool]             # None for informative-only (§6.2)
    metrics: Dict[str, float]          # per-cell-type and aggregate metrics
    detail: str                        # human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": self.metrics,
            "detail": self.detail,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ValidationResult":
        return ValidationResult(
            name=d["name"],
            passed=d.get("passed"),
            metrics=d.get("metrics", {}),
            detail=d.get("detail", ""),
        )


def _save_result(result: ValidationResult, filename: str) -> None:
    """Save a ValidationResult to the validation directory as JSON."""
    os.makedirs(config.SAP_VALIDATION_DIR, exist_ok=True)
    path = os.path.join(config.SAP_VALIDATION_DIR, filename)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  Saved: {path}")


def _load_result(filename: str) -> Optional[ValidationResult]:
    """Load a cached ValidationResult."""
    path = os.path.join(config.SAP_VALIDATION_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return ValidationResult.from_dict(json.load(f))


# ============================================================================
# §6.5 Residual Orthogonality Check
# ============================================================================

def validate_residual_orthogonality(
    data,  # SAPData
    model: ModelFit,
) -> ValidationResult:
    """§6.5: PCA on deviance residuals must not correlate with Condition.

    Extracts deviance residuals from the fitted model, runs PCA, and tests
    whether PC1/PC2 correlate with the condition variable via Kruskal-Wallis.
    """
    print("\n§6.5 Residual Orthogonality Check")
    print("=" * 40)

    n_sites = len(model.site_params)
    n_samples = data.a_obs.shape[0]
    y_all = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_all = data.x_base.values

    # Compute deviance residuals for all sites
    resid_matrix = np.zeros((n_samples, n_sites))
    for j in range(n_sites):
        sp = model.site_params[j]
        r_slice = (data.r_tensor[:, :, j]
                   if data.r_tensor is not None
                   else np.zeros((6, n_samples)))
        mu_j = compute_mu(sp, data.a_obs, data.sample_meta, x_base_all[j], r_slice)
        mu_j = np.maximum(mu_j, 1e-10)
        resid_matrix[:, j] = tweedie_deviance_residuals(y_all[j], mu_j, model.p)

    # PCA via SVD (center columns first)
    resid_centered = resid_matrix - resid_matrix.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(resid_centered, full_matrices=False)
    pc_scores = U * S  # (n_samples, n_components)

    # Kruskal-Wallis test: PC1 and PC2 vs condition
    conditions = data.sample_meta["condition"].values
    unique_conds = sorted(set(conditions))
    groups_pc1 = [pc_scores[conditions == c, 0] for c in unique_conds]
    groups_pc2 = [pc_scores[conditions == c, 1] for c in unique_conds]

    kw_stat1, kw_p1 = stats.kruskal(*groups_pc1)
    kw_stat2, kw_p2 = stats.kruskal(*groups_pc2)

    passed = (kw_p1 > config.RESIDUAL_ORTH_ALPHA and
              kw_p2 > config.RESIDUAL_ORTH_ALPHA)

    var_explained = S ** 2 / np.sum(S ** 2)

    metrics = {
        "pc1_kw_stat": float(kw_stat1),
        "pc1_kw_pval": float(kw_p1),
        "pc2_kw_stat": float(kw_stat2),
        "pc2_kw_pval": float(kw_p2),
        "pc1_var_explained": float(var_explained[0]),
        "pc2_var_explained": float(var_explained[1]),
    }

    status = "PASS" if passed else "FAIL"
    detail = (
        f"PC1 (var={var_explained[0]:.1%}): KW p={kw_p1:.4f}; "
        f"PC2 (var={var_explained[1]:.1%}): KW p={kw_p2:.4f} → {status}"
    )
    print(f"  {detail}")

    result = ValidationResult(
        name="residual_orthogonality",
        passed=passed,
        metrics=metrics,
        detail=detail,
    )
    _save_result(result, "residual_orth.json")
    return result


# ============================================================================
# §6.2 Cross-Modality Concordance Check
# ============================================================================

def validate_cross_modality_concordance(
    data,  # SAPData
    model: ModelFit,
) -> ValidationResult:
    """§6.2: Spearman correlation between RNA LFC and phospho β per (cell type, condition).

    Informative only — not a blocking gate.
    """
    print("\n§6.2 Cross-Modality Concordance Check")
    print("=" * 40)

    if data.gkp is None or data.kinase_genes is None:
        print("  SKIP: Phase 1 RNA data not loaded (run with --phase1)")
        return ValidationResult(
            name="cross_modality_concordance",
            passed=None,
            metrics={},
            detail="Skipped: Phase 1 RNA data not available",
        )

    from sap_data import build_kinase_substrate_map

    # Build kinase-substrate map: site_index → list of kinase genes
    ks_map = build_kinase_substrate_map(data.site_meta, data.kinase_genes)

    # Invert: kinase_gene → list of site indices
    kinase_to_sites: Dict[str, List[int]] = {}
    for site_idx, kinases in ks_map.items():
        for kin in kinases:
            kinase_to_sites.setdefault(kin, []).append(site_idx)

    # Compute RNA LFC per (cell_type, condition, gene) from gkp
    # gkp has MultiIndex (cell_type, sample_id)
    est_types = config.SAP_ESTIMATED_CELLTYPES
    conditions_to_test = ["AppP", "Ttau", "ApTt"]
    sample_conds = data.sample_meta["condition"].values
    sample_ids = data.sample_meta.index.tolist()

    # Map sample_id → condition for gkp lookup
    wtyp_samples = [sid for sid, c in zip(sample_ids, sample_conds) if c == "WTyp"]

    metrics: Dict[str, float] = {}
    results_detail = []

    for k_idx, ct in enumerate(est_types):
        for cond in conditions_to_test:
            cond_samples = [sid for sid, c in zip(sample_ids, sample_conds) if c == cond]

            rna_lfc_list = []
            phospho_beta_list = []

            for kin_gene in kinase_to_sites:
                # RNA LFC for this kinase gene
                if kin_gene not in data.gkp.columns:
                    continue
                try:
                    wtyp_expr = data.gkp.loc[(ct, wtyp_samples), kin_gene].values
                    cond_expr = data.gkp.loc[(ct, cond_samples), kin_gene].values
                except KeyError:
                    continue

                mean_wtyp = np.mean(wtyp_expr)
                mean_cond = np.mean(cond_expr)
                if abs(mean_wtyp) < 1e-10 and abs(mean_cond) < 1e-10:
                    continue
                # Use difference (data is centered/scaled) rather than log ratio
                rna_lfc = mean_cond - mean_wtyp

                # Phospho beta for substrate sites of this kinase
                site_indices = kinase_to_sites[kin_gene]
                # Extract the appropriate factorial beta component
                cond_idx = {"AppP": 0, "Ttau": 1, "ApTt": 2}[cond]
                betas = [model.site_params[j].beta[k_idx, cond_idx]
                         for j in site_indices if j < len(model.site_params)]
                if not betas:
                    continue

                rna_lfc_list.append(rna_lfc)
                phospho_beta_list.append(float(np.mean(betas)))

            if len(rna_lfc_list) < 10:
                key = f"{ct}_{cond}"
                metrics[f"spearman_{key}"] = float("nan")
                results_detail.append(f"  {ct} × {cond}: too few pairs ({len(rna_lfc_list)})")
                continue

            rho_s, p_val = stats.spearmanr(rna_lfc_list, phospho_beta_list)
            key = f"{ct}_{cond}"
            metrics[f"spearman_{key}"] = float(rho_s)
            metrics[f"pval_{key}"] = float(p_val)
            metrics[f"n_pairs_{key}"] = len(rna_lfc_list)

            interp = "concordant" if rho_s > 0.15 else ("anti" if rho_s < -0.15 else "decoupled")
            results_detail.append(
                f"  {ct:20s} × {cond}: ρ_S={rho_s:+.3f} (p={p_val:.3f}, n={len(rna_lfc_list)}) [{interp}]"
            )

    detail = "\n".join(results_detail)
    print(detail)

    result = ValidationResult(
        name="cross_modality_concordance",
        passed=None,  # informative only
        metrics=metrics,
        detail=detail,
    )
    _save_result(result, "cross_modality.json")
    return result


# ============================================================================
# Helpers for recovery assessment
# ============================================================================

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 if either vector is constant."""
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_slope(hat: np.ndarray, true: np.ndarray) -> float:
    """OLS slope of hat on true (no intercept)."""
    denom = np.sum(true ** 2)
    if denom < 1e-10:
        return 0.0
    return float(np.sum(hat * true) / denom)


# ============================================================================
# §6.1 Synthetic Phospho-Validation (Primary Gate)
# ============================================================================

def _generate_delta_true(
    scenario: str,
    data,  # SAPData
    model: ModelFit,
    seed: int = 42,
) -> np.ndarray:
    """Generate injected condition effects Δ^true for synthetic validation.

    Returns: (5, J, 3) array where axes are (cell_type, site, factorial_term).
    """
    rng = np.random.default_rng(seed)
    n_types = 5
    n_sites = len(model.site_params)
    delta = np.zeros((n_types, n_sites, 3))

    if scenario == "mdes":
        # Inject at composition-adjusted MDES per cell type.
        # Δ enters bulk diluted by A_{i,k}, so the effective MDES must
        # be scaled by 1/A_bar_k to produce a detectable change in bulk.
        # Δ_k = ±2 * sqrt(phi * mu^p / n_eff) / A_bar_k
        n_per_cond = 6  # 24 samples / 4 conditions
        x_base = data.x_base.values  # (J, 6)
        a_obs_vals = data.a_obs.values  # (24, 6)
        a_bar = np.maximum(a_obs_vals.mean(axis=0), 1e-4)  # (6,) mean proportion
        for k in range(n_types):
            phis = np.array([sp.phi for sp in model.site_params])
            mus = np.maximum(x_base[:, k], 1e-6)
            se = np.sqrt(phis * np.power(mus, model.p) / n_per_cond)
            mdes = 2.0 * se / a_bar[k]  # composition-adjusted MDES
            # Inject App effect at all sites, random sign
            signs = rng.choice([-1, 1], size=n_sites)
            delta[k, :, 0] = signs * mdes

    elif scenario == "sparse":
        # 5% of sites in 2 randomly chosen cell types
        n_affected = max(int(config.SYNTH_SPARSE_FRAC * n_sites), 10)
        chosen_types = rng.choice(n_types, size=config.SYNTH_SPARSE_NTYPES, replace=False)
        affected_sites = rng.choice(n_sites, size=n_affected, replace=False)
        for k in chosen_types:
            effect_size = rng.normal(0, 0.5, size=n_affected)
            delta[k, affected_sites, 0] = effect_size  # App effect
            delta[k, affected_sites, 1] = rng.normal(0, 0.3, size=n_affected)  # Tau

    elif scenario == "dense":
        # 25% of sites across all 5 cell types
        n_affected = max(int(config.SYNTH_DENSE_FRAC * n_sites), 50)
        affected_sites = rng.choice(n_sites, size=n_affected, replace=False)
        for k in range(n_types):
            delta[k, affected_sites, 0] = rng.normal(0, 0.5, size=n_affected)
            delta[k, affected_sites, 1] = rng.normal(0, 0.3, size=n_affected)
            delta[k, affected_sites, 2] = rng.normal(0, 0.2, size=n_affected)

    elif scenario == "de_novo":
        # Inject positive delta at sites where x_base ≈ 0
        x_base = data.x_base.values
        for k in range(n_types):
            zero_mask = x_base[:, k] < 1e-4
            zero_sites = np.where(zero_mask)[0]
            if len(zero_sites) < 5:
                continue
            n_inject = min(len(zero_sites), max(int(0.1 * len(zero_sites)), 10))
            chosen = rng.choice(zero_sites, size=n_inject, replace=False)
            delta[k, chosen, 0] = rng.exponential(0.3, size=n_inject)

    elif scenario == "rna_discordant":
        # Inject delta opposite in sign to r_tensor signal
        if data.r_tensor is None:
            # Fall back to random if no RNA data
            return _generate_delta_true("dense", data, model, seed)
        n_affected = max(int(0.15 * n_sites), 50)
        affected_sites = rng.choice(n_sites, size=n_affected, replace=False)
        for k in range(n_types):
            for j in affected_sites:
                # Mean r_tensor signal for this (cell_type, site) across samples
                r_mean = np.mean(data.r_tensor[k, :, j])
                # Inject opposite sign, scaled
                delta[k, j, 0] = -np.sign(r_mean) * abs(rng.normal(0.5, 0.2))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return delta


def _generate_synthetic_bulk(
    data,  # SAPData
    delta_true: np.ndarray,
    model: ModelFit,
    seed: int = 42,
) -> np.ndarray:
    """Construct synthetic bulk observations Y_syn from X_base + Δ^true + ε.

    Y_syn[i,j] = Σ_k A_{i,k}(X^base_{k,j} + Δ^true_{k,c(i),j}) + ε_{i,j}

    where ε follows compound Poisson-Gamma noise matching fitted (phi, p).

    Returns: (J, 24) synthetic bulk matrix.
    """
    rng = np.random.default_rng(seed)
    n_sites = len(model.site_params)
    n_samples = data.a_obs.shape[0]
    a_obs = data.a_obs.values  # (24, 6)
    x_base = data.x_base.values  # (J, 6)
    sample_conds = data.sample_meta["condition"].values

    # Map condition → factorial index for delta
    cond_to_idx = {"WTyp": -1, "AppP": 0, "Ttau": 1, "ApTt": 2}

    y_syn = np.zeros((n_sites, n_samples))

    for j in range(n_sites):
        sp = model.site_params[j]
        for i in range(n_samples):
            cond = sample_conds[i]
            # Perturbed cell-type signals
            mu_j_i = 0.0
            for k in range(5):
                s_kj = x_base[j, k]
                cidx = cond_to_idx[cond]
                if cidx >= 0:
                    s_kj += delta_true[k, j, cidx]
                s_kj = max(s_kj, 0.0)
                mu_j_i += a_obs[i, k] * s_kj
            # "Other" cell type (index 5) — baseline only
            mu_j_i += a_obs[i, 5] * max(x_base[j, 5], 0.0)
            mu_j_i = max(mu_j_i, 1e-10)

            # Hurdle: dropout
            dropout_prob = 1.0 / (1.0 + np.exp(-(sp.gamma0 + sp.gamma1 * np.log(mu_j_i))))
            if rng.random() < dropout_prob:
                y_syn[j, i] = 0.0
            else:
                # Compound Poisson-Gamma noise for Tweedie p in (1, 2)
                y_syn[j, i] = _sample_tweedie(mu_j_i, sp.phi, model.p, rng)

    return y_syn


def _sample_tweedie(mu: float, phi: float, p: float, rng) -> float:
    """Sample from Tweedie distribution using compound Poisson-Gamma representation.

    For p in (1, 2): Y = sum of N Gamma variates, where:
      N ~ Poisson(lambda), lambda = mu^(2-p) / (phi * (2-p))
      Each variate ~ Gamma(shape=-p/(2-p), scale=phi*(p-1)*mu^(p-1))
    """
    lam = mu ** (2 - p) / (phi * (2 - p))
    n_claims = rng.poisson(lam)
    if n_claims == 0:
        return 0.0
    alpha = -p / (2 - p)  # note: this is positive since p < 2
    # Actually alpha = (2-p)/(p-1) for the shape parameter
    # Standard parameterization: shape = -p/(2-p) is negative; use |shape|
    shape = (2 - p) / (p - 1)
    scale = phi * (p - 1) * mu ** (p - 1)
    return float(np.sum(rng.gamma(shape, scale, size=n_claims)))


def validate_synthetic_phospho(
    data,  # SAPData
    model: ModelFit,
    scenario: str = "all",
    n_workers: int = None,
    site_subsample: float = None,
    seed: int = 42,
) -> List[ValidationResult]:
    """§6.1: Synthetic phospho-validation across one or all scenarios.

    Injects known Δ^true, reconstructs synthetic bulk, refits model,
    assesses recovery per cell type.
    """
    print("\n§6.1 Synthetic Phospho-Validation")
    print("=" * 40)

    if n_workers is None:
        n_workers = config.N_WORKERS

    scenarios = config.SYNTH_SCENARIOS if scenario == "all" else [scenario]
    results = []

    # Build kinase-substrate map for split mapped/unmapped reporting
    from sap_data import build_kinase_substrate_map
    ks_map = build_kinase_substrate_map(data.site_meta, data.kinase_genes or [])
    mapped_set = set(ks_map.keys())

    for scen in scenarios:
        print(f"\n  Scenario: {scen}")
        delta_true = _generate_delta_true(scen, data, model, seed)

        # Generate synthetic bulk
        y_syn = _generate_synthetic_bulk(data, delta_true, model, seed)
        print(f"    Y_syn shape: {y_syn.shape}, zeros: {np.sum(y_syn == 0)}/{y_syn.size}")

        # Site subsampling
        n_sites = len(model.site_params)
        if site_subsample is not None:
            n_sub = max(int(n_sites * site_subsample), 100)
            rng = np.random.default_rng(seed + 1)
            site_indices = np.sort(rng.choice(n_sites, n_sub, replace=False))
        else:
            site_indices = np.arange(n_sites)

        # Refit model on synthetic data
        print(f"    Refitting model on {len(site_indices)} sites...")
        cv = model.cv_result
        if cv is None:
            lam_q = np.full(config.N_INTENSITY_STRATA, 0.1)
            lam_rho, eta, gamma = 0.1, 2.0, 0.5
        else:
            lam_q = cv.best_lambda
            lam_rho, eta, gamma = cv.best_lambda_rho, cv.best_eta, cv.best_gamma

        data_syn = _make_data_copy_with_y(data, y_syn)
        model_syn = fit_hurdle_tweedie(
            data_syn, lam_q, lam_rho, eta, gamma,
            n_workers=n_workers, site_indices=site_indices,
        )

        # Assess recovery — overall and split by mapped/unmapped
        metrics: Dict[str, float] = {}
        all_passed = True
        est_types = config.SAP_ESTIMATED_CELLTYPES

        # Partition assessed sites into mapped and unmapped
        mapped_mask = np.array([j in mapped_set for j in site_indices])
        unmapped_mask = ~mapped_mask
        metrics["n_mapped"] = float(mapped_mask.sum())
        metrics["n_unmapped"] = float(unmapped_mask.sum())

        for k, ct in enumerate(est_types):
            d_hat = np.array([model_syn.site_params[i].beta[k]
                              for i in range(len(model_syn.site_params))])  # (n_assessed, 3)
            d_true_k = delta_true[k, site_indices, :]  # (n_assessed, 3)

            # Overall correlation
            hat_flat = d_hat.flatten()
            true_flat = d_true_k.flatten()
            r_val = _safe_pearson(hat_flat, true_flat)
            slope = _safe_slope(hat_flat, true_flat)

            # Split: mapped sites
            r_mapped = _safe_pearson(d_hat[mapped_mask].flatten(),
                                     d_true_k[mapped_mask].flatten())
            # Split: unmapped sites
            r_unmapped = _safe_pearson(d_hat[unmapped_mask].flatten(),
                                       d_true_k[unmapped_mask].flatten())

            thresh = config.SYNTH_PEARSON_PER_CELLTYPE.get(ct, config.SYNTH_PEARSON_OVERALL)
            ct_pass = r_val >= thresh
            if not ct_pass:
                all_passed = False

            metrics[f"pearson_{ct}"] = r_val
            metrics[f"pearson_mapped_{ct}"] = r_mapped
            metrics[f"pearson_unmapped_{ct}"] = r_unmapped
            metrics[f"slope_{ct}"] = slope
            metrics[f"pass_{ct}"] = float(ct_pass)

            lo, hi = config.SYNTH_SLOPE_RANGE
            slope_ok = lo <= slope <= hi
            print(f"    {ct:20s}: r={r_val:.3f} (mapped={r_mapped:.3f}, unmapped={r_unmapped:.3f}), "
                  f"slope={slope:.3f} ({'OK' if slope_ok else 'BIAS'}) "
                  f"→ {'PASS' if ct_pass else 'FAIL'}")

        # Scenario 5 supplementary check
        if scen == "rna_discordant" and data.r_tensor is not None:
            for k, ct in enumerate(est_types):
                d_hat = np.array([model_syn.site_params[i].beta[k, 0]
                                  for i in range(len(model_syn.site_params))])
                d_true_k = delta_true[k, site_indices, 0]
                r_mean_k = np.array([np.mean(data.r_tensor[k, :, j]) for j in site_indices])

                r_vs_true = float(np.corrcoef(d_hat, d_true_k)[0, 1]) if np.std(d_true_k) > 1e-10 else 0
                r_vs_rna = float(np.corrcoef(d_hat, r_mean_k)[0, 1]) if np.std(r_mean_k) > 1e-10 else 0
                metrics[f"rna_discord_r_true_{ct}"] = r_vs_true
                metrics[f"rna_discord_r_rna_{ct}"] = r_vs_rna
                overrides = r_vs_true > r_vs_rna
                print(f"    {ct:20s}: r(Δ̂,Δ)={r_vs_true:.3f} vs r(Δ̂,r)={r_vs_rna:.3f} "
                      f"→ {'phospho overrides RNA' if overrides else 'WARNING: RNA dominates'}")

        result = ValidationResult(
            name=f"synthetic_{scen}",
            passed=all_passed,
            metrics=metrics,
            detail=f"Scenario {scen}: {'PASS' if all_passed else 'FAIL'}",
        )
        _save_result(result, f"synthetic_{scen}.json")
        results.append(result)

    return results


def _make_data_copy_with_y(data, y_override: np.ndarray):
    """Create a shallow copy of SAPData with overridden bulk_phospho."""
    from sap_data import SAPData
    import copy
    data_copy = copy.copy(data)
    data_copy.bulk_phospho = pd.DataFrame(
        y_override,
        index=data.bulk_phospho.index,
        columns=data.bulk_phospho.columns,
    )
    return data_copy


# ============================================================================
# §6.1.1 Pseudobulk Stress Test
# ============================================================================

def validate_pseudobulk_stress(
    data,  # SAPData
    model: ModelFit,
    n_workers: int = None,
    site_subsample: float = None,
    seed: int = 42,
) -> ValidationResult:
    """§6.1.1: Aggregate scRNA-seq into synthetic bulk, fit, compare to ground truth."""
    print("\n§6.1.1 Pseudobulk Stress Test")
    print("=" * 40)

    if data.gkp is None:
        print("  SKIP: Phase 1 RNA data not loaded")
        return ValidationResult(
            name="pseudobulk_stress", passed=None, metrics={},
            detail="Skipped: Phase 1 RNA data not available",
        )

    if n_workers is None:
        n_workers = config.N_WORKERS

    from sap_data import build_kinase_substrate_map

    ks_map = build_kinase_substrate_map(data.site_meta, data.kinase_genes)

    # Build ground-truth per-cell-type profiles and synthetic bulk
    est_types = config.SAP_ESTIMATED_CELLTYPES
    n_sites = len(model.site_params)
    n_samples = data.a_obs.shape[0]
    a_obs = data.a_obs.values
    sample_ids = data.sample_meta.index.tolist()

    # Ground truth: per-cell-type expression at mapped sites
    gt_profiles = np.zeros((5, n_samples, n_sites))  # (K, 24, J)
    y_pseudo = np.zeros((n_sites, n_samples))

    for j in range(n_sites):
        kinases = ks_map.get(j, [])
        if not kinases:
            # No mapped kinase — use site average (weak prior per SAP §5.2)
            for k_idx, ct in enumerate(est_types):
                for s_idx, sid in enumerate(sample_ids):
                    try:
                        gt_profiles[k_idx, s_idx, j] = float(
                            data.gkp.loc[(ct, sid)].mean()
                        )
                    except KeyError:
                        pass
        else:
            available = [g for g in kinases if g in data.gkp.columns]
            if not available:
                continue
            for k_idx, ct in enumerate(est_types):
                for s_idx, sid in enumerate(sample_ids):
                    try:
                        gt_profiles[k_idx, s_idx, j] = float(
                            data.gkp.loc[(ct, sid), available].mean()
                        )
                    except KeyError:
                        pass

        # Aggregate to pseudobulk
        for i in range(n_samples):
            for k in range(5):
                y_pseudo[j, i] += a_obs[i, k] * gt_profiles[k, i, j]
            y_pseudo[j, i] += a_obs[i, 5] * gt_profiles[0, i, j] * 0.1  # weak Other

    # Refit on pseudobulk
    print(f"  Pseudobulk Y shape: {y_pseudo.shape}, "
          f"nonzero: {np.sum(y_pseudo > 0)}/{y_pseudo.size}")

    n_assess = n_sites
    if site_subsample is not None:
        n_assess = max(int(n_sites * site_subsample), 100)
    site_indices = np.sort(
        np.random.default_rng(seed).choice(n_sites, min(n_assess, n_sites), replace=False)
    )

    cv = model.cv_result
    if cv is None:
        lam_q = np.full(config.N_INTENSITY_STRATA, 0.1)
        lam_rho, eta, gamma = 0.1, 2.0, 0.5
    else:
        lam_q = cv.best_lambda
        lam_rho, eta, gamma = cv.best_lambda_rho, cv.best_eta, cv.best_gamma

    data_pseudo = _make_data_copy_with_y(data, y_pseudo)
    model_pseudo = fit_hurdle_tweedie(
        data_pseudo, lam_q, lam_rho, eta, gamma,
        n_workers=n_workers, site_indices=site_indices,
    )

    # Compare recovered vs ground truth condition effects
    sample_conds = data.sample_meta["condition"].values
    wtyp_mask = sample_conds == "WTyp"
    metrics: Dict[str, float] = {}
    all_passed = True

    # Split assessed sites into mapped / unmapped
    mapped_mask = np.array([j in set(ks_map.keys()) for j in site_indices])
    unmapped_mask = ~mapped_mask
    metrics["n_mapped"] = float(mapped_mask.sum())
    metrics["n_unmapped"] = float(unmapped_mask.sum())

    for k, ct in enumerate(est_types):
        # Ground truth condition effect: mean(cond) - mean(WTyp) per site
        gt_wtyp = gt_profiles[k, wtyp_mask, :].mean(axis=0)  # (J,)
        recovered = np.array([model_pseudo.site_params[i].beta[k, 0]
                              for i in range(len(model_pseudo.site_params))])

        # Use AppP as the primary comparison condition
        app_mask = sample_conds == "AppP"
        gt_app = gt_profiles[k, app_mask, :].mean(axis=0)
        gt_effect = (gt_app - gt_wtyp)[site_indices]

        r_val = _safe_pearson(recovered, gt_effect)
        r_mapped = _safe_pearson(recovered[mapped_mask], gt_effect[mapped_mask])
        r_unmapped = _safe_pearson(recovered[unmapped_mask], gt_effect[unmapped_mask])

        thresh = config.SYNTH_PEARSON_PER_CELLTYPE.get(ct, config.SYNTH_PEARSON_OVERALL)
        ct_pass = r_val >= thresh
        if not ct_pass:
            all_passed = False

        metrics[f"pearson_{ct}"] = r_val
        metrics[f"pearson_mapped_{ct}"] = r_mapped
        metrics[f"pearson_unmapped_{ct}"] = r_unmapped
        metrics[f"pass_{ct}"] = float(ct_pass)
        print(f"  {ct:20s}: r={r_val:.3f} (mapped={r_mapped:.3f}, unmapped={r_unmapped:.3f}) "
              f"→ {'PASS' if ct_pass else 'FAIL'}")

    result = ValidationResult(
        name="pseudobulk_stress",
        passed=all_passed,
        metrics=metrics,
        detail=f"Pseudobulk stress test: {'PASS' if all_passed else 'FAIL'}",
    )
    _save_result(result, "pseudobulk_stress.json")
    return result


# ============================================================================
# §6.3 Perturbation Audit
# ============================================================================

def _perturb_worker(args: Tuple) -> np.ndarray:
    """Worker for perturbation audit. Returns nonzero mask (n_sites, 5)."""
    (y_all, a_obs_vals, sample_meta_dict, x_base_all, r_tensor,
     strata, omega, a_bar, p, lam_q, lam_rho, eta,
     site_indices, sigma, iteration, seed,
     warm_beta, warm_alpha_gen, warm_alpha_time, warm_rho,
     warm_gamma0, warm_gamma1, warm_phi) = args

    rng = np.random.default_rng(seed + iteration)
    n_samples, n_types = a_obs_vals.shape

    # Perturb A_obs
    eps = rng.normal(0, sigma, size=(n_samples, n_types))
    a_star = np.clip(a_obs_vals + eps, 0, None)
    row_sums = a_star.sum(axis=1, keepdims=True)
    a_star = a_star / np.maximum(row_sums, 1e-10)
    a_star_df = pd.DataFrame(a_star, columns=[f"ct_{i}" for i in range(n_types)])

    # Reconstruct sample_meta DataFrame
    sample_meta = pd.DataFrame(sample_meta_dict)

    nonzero_mask = np.zeros((len(site_indices), 5), dtype=bool)

    for idx, j in enumerate(site_indices):
        r_slice = r_tensor[:, :, j] if r_tensor is not None else np.zeros((n_types, n_samples))

        # Warm start
        params_init = SiteParams(
            beta=warm_beta[idx].copy(),
            alpha_gen=float(warm_alpha_gen[idx]),
            alpha_time=warm_alpha_time[idx].copy(),
            rho=float(warm_rho[idx]),
            gamma0=float(warm_gamma0[idx]),
            gamma1=float(warm_gamma1[idx]),
            phi=float(warm_phi[idx]),
            stratum=int(strata[j]),
        )

        params_j, _ = fit_site(
            j, y_all[j], a_star_df, sample_meta,
            x_base_all[j], r_slice, p,
            float(lam_q[strata[j]]), lam_rho, omega, a_bar, eta,
            int(strata[j]), max_outer_iter=5, params_init=params_init,
        )

        for k in range(5):
            nonzero_mask[idx, k] = np.linalg.norm(params_j.beta[k]) > 1e-6

    return nonzero_mask


def validate_perturbation_audit(
    data,  # SAPData
    model: ModelFit,
    sigma_grid: List[float] = None,
    n_iter: int = None,
    n_workers: int = None,
    site_subsample: float = 0.2,
    seed: int = 42,
) -> List[ValidationResult]:
    """§6.3: Perturbation audit — inject noise into A_obs, assess stability."""
    print("\n§6.3 Perturbation Audit")
    print("=" * 40)

    if sigma_grid is None:
        sigma_grid = config.PERTURB_SIGMA_GRID
    if n_iter is None:
        n_iter = config.PERTURB_N_ITER
    if n_workers is None:
        n_workers = config.N_WORKERS

    cv = model.cv_result
    if cv is None:
        lam_q = np.full(config.N_INTENSITY_STRATA, 0.1)
        lam_rho, eta, gamma_cvs = 0.1, 2.0, 0.5
    else:
        lam_q = cv.best_lambda
        lam_rho, eta, gamma_cvs = cv.best_lambda_rho, cv.best_eta, cv.best_gamma

    omega = compute_adaptive_weights(data.cvs, gamma_cvs)
    a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values

    # Site subsampling
    n_sites = len(model.site_params)
    n_sub = max(int(n_sites * site_subsample), 100)
    rng = np.random.default_rng(seed)
    site_indices = np.sort(rng.choice(n_sites, min(n_sub, n_sites), replace=False))

    # Prepare warm-start arrays for selected sites
    warm_beta = np.array([model.site_params[j].beta for j in site_indices])
    warm_alpha_gen = np.array([model.site_params[j].alpha_gen for j in site_indices])
    warm_alpha_time = np.array([model.site_params[j].alpha_time for j in site_indices])
    warm_rho = np.array([model.site_params[j].rho for j in site_indices])
    warm_gamma0 = np.array([model.site_params[j].gamma0 for j in site_indices])
    warm_gamma1 = np.array([model.site_params[j].gamma1 for j in site_indices])
    warm_phi = np.array([model.site_params[j].phi for j in site_indices])

    # Original nonzero status
    orig_nonzero = np.zeros((len(site_indices), 5), dtype=bool)
    for idx, j in enumerate(site_indices):
        for k in range(5):
            orig_nonzero[idx, k] = np.linalg.norm(model.site_params[j].beta[k]) > 1e-6

    y_all = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_all = data.x_base.values
    strata = data.intensity_strata
    sample_meta_dict = {
        col: data.sample_meta[col].tolist() for col in data.sample_meta.columns
    }
    r_tensor = data.r_tensor

    results = []

    for sigma in sigma_grid:
        print(f"\n  σ_A = {sigma}")

        # Build worker args
        worker_args = [
            (y_all, data.a_obs.values, sample_meta_dict, x_base_all, r_tensor,
             strata, omega, a_bar, model.p, lam_q, lam_rho, eta,
             site_indices, sigma, it, seed,
             warm_beta, warm_alpha_gen, warm_alpha_time, warm_rho,
             warm_gamma0, warm_gamma1, warm_phi)
            for it in range(n_iter)
        ]

        # Run iterations (parallel or sequential)
        from concurrent.futures import ProcessPoolExecutor
        collapse_counts = np.zeros((len(site_indices), 5), dtype=int)

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for i, nonzero_mask in enumerate(pool.map(_perturb_worker, worker_args)):
                    # Count collapses: was nonzero in original, now zero
                    collapsed = orig_nonzero & ~nonzero_mask
                    collapse_counts += collapsed.astype(int)
                    if (i + 1) % 50 == 0:
                        print(f"    Iteration {i + 1}/{n_iter}")
        else:
            for i, args_i in enumerate(worker_args):
                nonzero_mask = _perturb_worker(args_i)
                collapsed = orig_nonzero & ~nonzero_mask
                collapse_counts += collapsed.astype(int)
                if (i + 1) % 50 == 0:
                    print(f"    Iteration {i + 1}/{n_iter}")

        # Compute collapse rates
        n_orig_nonzero = orig_nonzero.sum()
        collapse_rate = collapse_counts / max(n_iter, 1)
        n_sensitive = int(np.sum(
            (collapse_rate > config.PERTURB_COLLAPSE_THRESH) & orig_nonzero
        ))

        metrics = {
            "sigma": sigma,
            "n_iter": n_iter,
            "n_sites_assessed": len(site_indices),
            "n_orig_nonzero_groups": int(n_orig_nonzero),
            "n_composition_sensitive": n_sensitive,
            "frac_sensitive": n_sensitive / max(n_orig_nonzero, 1),
            "mean_collapse_rate": float(collapse_rate[orig_nonzero].mean())
            if n_orig_nonzero > 0 else 0.0,
        }

        print(f"    Original nonzero groups: {n_orig_nonzero}")
        print(f"    Composition-sensitive (>10% collapse): {n_sensitive} "
              f"({metrics['frac_sensitive']:.1%})")
        print(f"    Mean collapse rate: {metrics['mean_collapse_rate']:.3f}")

        result = ValidationResult(
            name=f"perturbation_{sigma}",
            passed=None,  # perturbation is flagging, not pass/fail
            metrics=metrics,
            detail=f"σ={sigma}: {n_sensitive} composition-sensitive groups",
        )
        _save_result(result, f"perturbation_{sigma}.json")
        results.append(result)

    return results


# ============================================================================
# §6.4 Permutation Null (False Positive Calibration)
# ============================================================================

def _generate_restricted_permutation(
    sample_meta: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """Generate condition labels permuted within (gender × timepoint) blocks.

    Returns permuted condition array of length n_samples.
    """
    rng = np.random.default_rng(seed)
    conditions = sample_meta["condition"].values.copy()
    genders = sample_meta["gender"].values
    timepoints = sample_meta["timepoint"].values

    # Identify blocks
    for g in np.unique(genders):
        for t in np.unique(timepoints):
            block_mask = (genders == g) & (timepoints == t)
            block_indices = np.where(block_mask)[0]
            # Permute condition labels within this block
            block_conds = conditions[block_indices]
            rng.shuffle(block_conds)
            conditions[block_indices] = block_conds

    return conditions


def _permute_worker(args: Tuple) -> Tuple[np.ndarray, int]:
    """Worker for permutation null. Returns (nonzero_counts (n_sites, 5), perm_idx)."""
    (y_all, a_obs_vals, sample_meta_dict, x_base_all, r_tensor,
     strata, omega, a_bar, p, lam_q, lam_rho, eta,
     site_indices, perm_idx, seed,
     warm_beta, warm_alpha_gen, warm_alpha_time, warm_rho,
     warm_gamma0, warm_gamma1, warm_phi) = args

    # Reconstruct sample_meta and apply permuted conditions
    sample_meta = pd.DataFrame(sample_meta_dict)
    perm_conditions = _generate_restricted_permutation(sample_meta, seed + perm_idx)
    sample_meta_perm = sample_meta.copy()
    sample_meta_perm["condition"] = perm_conditions

    a_obs_df = pd.DataFrame(a_obs_vals, columns=[f"ct_{i}" for i in range(a_obs_vals.shape[1])])

    nonzero_mask = np.zeros((len(site_indices), 5), dtype=bool)

    for idx, j in enumerate(site_indices):
        r_slice = r_tensor[:, :, j] if r_tensor is not None else np.zeros((a_obs_vals.shape[1], a_obs_vals.shape[0]))

        params_init = SiteParams(
            beta=warm_beta[idx].copy(),
            alpha_gen=float(warm_alpha_gen[idx]),
            alpha_time=warm_alpha_time[idx].copy(),
            rho=float(warm_rho[idx]),
            gamma0=float(warm_gamma0[idx]),
            gamma1=float(warm_gamma1[idx]),
            phi=float(warm_phi[idx]),
            stratum=int(strata[j]),
        )

        params_j, _ = fit_site(
            j, y_all[j], a_obs_df, sample_meta_perm,
            x_base_all[j], r_slice, p,
            float(lam_q[strata[j]]), lam_rho, omega, a_bar, eta,
            int(strata[j]), max_outer_iter=5, params_init=params_init,
        )

        for k in range(5):
            nonzero_mask[idx, k] = np.linalg.norm(params_j.beta[k]) > 1e-6

    return nonzero_mask, perm_idx


def validate_permutation_null(
    data,  # SAPData
    model: ModelFit,
    n_perm: int = None,
    n_workers: int = None,
    site_subsample: float = 0.2,
    seed: int = 42,
) -> ValidationResult:
    """§6.4: Permutation null — restricted permutation of condition labels."""
    print("\n§6.4 Permutation Null (False Positive Calibration)")
    print("=" * 40)

    if n_perm is None:
        n_perm = config.PERM_NULL_N
    if n_workers is None:
        n_workers = config.N_WORKERS

    cv = model.cv_result
    if cv is None:
        lam_q = np.full(config.N_INTENSITY_STRATA, 0.1)
        lam_rho, eta, gamma_cvs = 0.1, 2.0, 0.5
    else:
        lam_q = cv.best_lambda
        lam_rho, eta, gamma_cvs = cv.best_lambda_rho, cv.best_eta, cv.best_gamma

    omega = compute_adaptive_weights(data.cvs, gamma_cvs)
    a_bar = data.a_obs[config.SAP_ESTIMATED_CELLTYPES].mean(axis=0).values

    # Site subsampling
    n_sites = len(model.site_params)
    n_sub = max(int(n_sites * site_subsample), 100)
    rng = np.random.default_rng(seed)
    site_indices = np.sort(rng.choice(n_sites, min(n_sub, n_sites), replace=False))

    # Original nonzero rate
    orig_nonzero = np.zeros((len(site_indices), 5), dtype=bool)
    for idx, j in enumerate(site_indices):
        for k in range(5):
            orig_nonzero[idx, k] = np.linalg.norm(model.site_params[j].beta[k]) > 1e-6
    orig_rate = orig_nonzero.mean()

    # Warm-start arrays
    warm_beta = np.array([model.site_params[j].beta for j in site_indices])
    warm_alpha_gen = np.array([model.site_params[j].alpha_gen for j in site_indices])
    warm_alpha_time = np.array([model.site_params[j].alpha_time for j in site_indices])
    warm_rho = np.array([model.site_params[j].rho for j in site_indices])
    warm_gamma0 = np.array([model.site_params[j].gamma0 for j in site_indices])
    warm_gamma1 = np.array([model.site_params[j].gamma1 for j in site_indices])
    warm_phi = np.array([model.site_params[j].phi for j in site_indices])

    y_all = np.nan_to_num(data.bulk_phospho.values, nan=0.0)
    x_base_all = data.x_base.values
    strata = data.intensity_strata
    sample_meta_dict = {
        col: data.sample_meta[col].tolist() for col in data.sample_meta.columns
    }

    # Build worker args — use zero warm start for null (original params are biased)
    zero_beta = np.zeros_like(warm_beta)
    zero_alpha_gen = np.zeros_like(warm_alpha_gen)
    zero_alpha_time = np.zeros_like(warm_alpha_time)
    zero_rho = np.zeros_like(warm_rho)
    zero_gamma0 = np.zeros_like(warm_gamma0)
    zero_gamma1 = np.zeros_like(warm_gamma1)

    worker_args = [
        (y_all, data.a_obs.values, sample_meta_dict, x_base_all, data.r_tensor,
         strata, omega, a_bar, model.p, lam_q, lam_rho, eta,
         site_indices, perm_i, seed,
         zero_beta, zero_alpha_gen, zero_alpha_time, zero_rho,
         zero_gamma0, zero_gamma1, warm_phi)
        for perm_i in range(n_perm)
    ]

    # Run permutations
    null_rates = []
    from concurrent.futures import ProcessPoolExecutor

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for nonzero_mask, perm_i in pool.map(_permute_worker, worker_args):
                null_rates.append(nonzero_mask.mean())
                if (perm_i + 1) % 100 == 0:
                    print(f"    Permutation {perm_i + 1}/{n_perm}")
    else:
        for args_i in worker_args:
            nonzero_mask, perm_i = _permute_worker(args_i)
            null_rates.append(nonzero_mask.mean())
            if (perm_i + 1) % 100 == 0:
                print(f"    Permutation {perm_i + 1}/{n_perm}")

    null_rates = np.array(null_rates)
    mean_null_rate = float(null_rates.mean())
    tolerance = config.PERM_NULL_SPARSITY_TOLERANCE

    # Check: null rate should not exceed original by more than tolerance (relative)
    inflation = (mean_null_rate - orig_rate) / max(orig_rate, 1e-6)
    passed = inflation <= tolerance

    metrics = {
        "orig_nonzero_rate": float(orig_rate),
        "mean_null_nonzero_rate": mean_null_rate,
        "inflation": float(inflation),
        "null_rate_std": float(null_rates.std()),
        "null_rate_95pct": float(np.percentile(null_rates, 95)),
        "n_perm": n_perm,
        "n_sites_assessed": len(site_indices),
    }

    # Save null distribution for Phase 4
    null_path = os.path.join(config.SAP_VALIDATION_DIR, "permutation_null_dist.npz")
    os.makedirs(config.SAP_VALIDATION_DIR, exist_ok=True)
    np.savez_compressed(null_path, null_rates=null_rates)

    status = "PASS" if passed else "FAIL"
    detail = (
        f"Original nonzero rate: {orig_rate:.3f}, "
        f"Null mean: {mean_null_rate:.3f} (inflation: {inflation:+.1%}) → {status}"
    )
    print(f"  {detail}")

    result = ValidationResult(
        name="permutation_null",
        passed=passed,
        metrics=metrics,
        detail=detail,
    )
    _save_result(result, "permutation_null.json")
    return result


# ============================================================================
# §5 bMIND Benchmark
# ============================================================================

def validate_bmind_benchmark(
    data,  # SAPData
    model: ModelFit,
) -> ValidationResult:
    """§5: bMIND benchmark comparator via R subprocess."""
    import subprocess
    import tempfile

    print("\n§5 bMIND Benchmark")
    print("=" * 40)

    if data.gkp is None:
        print("  SKIP: Phase 1 RNA data not loaded")
        return ValidationResult(
            name="bmind_concordance", passed=None, metrics={},
            detail="Skipped: Phase 1 RNA data not available",
        )

    bmind_dir = os.path.join(config.SAP_VALIDATION_DIR, "bmind")
    os.makedirs(bmind_dir, exist_ok=True)

    # Prepare input files
    print("  Writing bMIND inputs...")

    # Bulk matrix (J × 24)
    bulk_path = os.path.join(bmind_dir, "bulk_matrix.tsv")
    data.bulk_phospho.to_csv(bulk_path, sep="\t")

    # Composition matrix (24 × 6)
    aobs_path = os.path.join(bmind_dir, "a_obs_matrix.tsv")
    data.a_obs.to_csv(aobs_path, sep="\t")

    # Reference profiles: collapse G^KP to (6 cell types × genes) means
    # Also compute per-cell-type covariance for bMIND prior (per SAP author feedback)
    est_types = config.SAP_ESTIMATED_CELLTYPES
    ref_data = {}
    ref_covs = {}
    for ct in est_types:
        try:
            ct_data = data.gkp.loc[ct]  # (24 samples × genes)
            ref_data[ct] = ct_data.mean(axis=0)
            ref_covs[ct] = ct_data.cov()  # (genes × genes) empirical covariance
        except KeyError:
            ref_data[ct] = pd.Series(0.0, index=data.gkp.columns)
    # Add "Other" as average of all
    ref_data["Other"] = pd.DataFrame(ref_data).mean(axis=1) * 0.1
    ref_df = pd.DataFrame(ref_data)
    ref_path = os.path.join(bmind_dir, "reference_profiles.tsv")
    ref_df.to_csv(ref_path, sep="\t")

    # Write per-cell-type covariance matrices (used by bMIND if its API supports it)
    for ct, cov_df in ref_covs.items():
        cov_path = os.path.join(bmind_dir, f"ref_cov_{ct}.tsv")
        cov_df.to_csv(cov_path, sep="\t")

    # Call R script
    r_script = os.path.join(config.REPO_ROOT, "code", "r", "run_bmind.R")
    output_path = os.path.join(bmind_dir, "X_bmind.tsv")

    print("  Running bMIND via R...")
    try:
        result = subprocess.run(
            ["Rscript", r_script,
             "--input-dir", bmind_dir,
             "--output-dir", bmind_dir],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(f"  R script failed:\n{result.stderr[:500]}")
            return ValidationResult(
                name="bmind_concordance", passed=None, metrics={},
                detail=f"bMIND R script failed: {result.stderr[:200]}",
            )
    except FileNotFoundError:
        print("  Rscript not found. Install R and bMIND package.")
        return ValidationResult(
            name="bmind_concordance", passed=None, metrics={},
            detail="Rscript not found",
        )
    except subprocess.TimeoutExpired:
        print("  bMIND timed out (1h limit)")
        return ValidationResult(
            name="bmind_concordance", passed=None, metrics={},
            detail="bMIND timed out",
        )

    # Read bMIND output
    if not os.path.exists(output_path):
        print("  bMIND output not found")
        return ValidationResult(
            name="bmind_concordance", passed=None, metrics={},
            detail="bMIND output file not found",
        )

    print("  Reading bMIND output...")
    x_bmind_df = pd.read_csv(output_path, sep="\t", index_col=0)

    # Parse: columns are "celltype_sampleidx" format
    n_types = 6
    n_samples = data.a_obs.shape[0]
    n_sites = len(model.site_params)

    # Reshape to (6, 24, J) — depends on R output format
    # Assume columns ordered: ct0_s0, ct0_s1, ..., ct0_s23, ct1_s0, ...
    x_bmind = np.zeros((n_types, n_samples, min(n_sites, x_bmind_df.shape[0])))
    n_bmind_sites = min(n_sites, x_bmind_df.shape[0])
    if x_bmind_df.shape[1] == n_types * n_samples:
        vals = x_bmind_df.values[:n_bmind_sites]
        x_bmind = vals.reshape(n_bmind_sites, n_types, n_samples).transpose(1, 2, 0)
    else:
        print(f"  WARNING: Unexpected bMIND output shape: {x_bmind_df.shape}")

    # Compute LFC_bMIND per (cell_type, condition)
    sample_conds = data.sample_meta["condition"].values
    wtyp_mask = sample_conds == "WTyp"

    metrics: Dict[str, float] = {}
    all_concordant = True

    for k, ct in enumerate(est_types):
        wtyp_mean = x_bmind[k, wtyp_mask, :].mean(axis=0)  # (J,)

        for cond in ["AppP", "Ttau", "ApTt"]:
            cond_mask = sample_conds == cond
            cond_mean = x_bmind[k, cond_mask, :].mean(axis=0)
            # Avoid log(0)
            safe_wtyp = np.maximum(wtyp_mean, 1e-10)
            safe_cond = np.maximum(cond_mean, 1e-10)
            lfc_bmind = np.log2(safe_cond / safe_wtyp)

            # Compare to Tweedie beta
            cond_idx = {"AppP": 0, "Ttau": 1, "ApTt": 2}[cond]
            beta_tweedie = np.array([
                model.site_params[j].beta[k, cond_idx]
                for j in range(min(n_bmind_sites, n_sites))
            ])

            if np.std(lfc_bmind) < 1e-10 or np.std(beta_tweedie) < 1e-10:
                r_val = 0.0
            else:
                r_val = float(np.corrcoef(
                    beta_tweedie[:n_bmind_sites], lfc_bmind[:n_bmind_sites]
                )[0, 1])

            key = f"{ct}_{cond}"
            metrics[f"pearson_{key}"] = r_val

            thresh = config.BMIND_CONCORDANCE_R_THRESH
            concordant = r_val >= thresh
            if not concordant:
                all_concordant = False
            print(f"  {ct:20s} × {cond}: r={r_val:.3f} "
                  f"(thresh={thresh}) → {'OK' if concordant else 'LOW'}")

    result = ValidationResult(
        name="bmind_concordance",
        passed=all_concordant,
        metrics=metrics,
        detail=f"bMIND concordance: {'all axes above threshold' if all_concordant else 'some axes below threshold'}",
    )
    _save_result(result, "bmind_concordance.json")
    return result


# ============================================================================
# Summary
# ============================================================================

def print_validation_summary() -> None:
    """Print a summary of all cached validation results."""
    print("\nSAP Validation Summary")
    print("=" * 60)

    files = [
        ("§6.5 Residual Orthogonality", "residual_orth.json"),
        ("§6.2 Cross-Modality Concordance", "cross_modality.json"),
        ("§6.1 Synthetic: mdes", "synthetic_mdes.json"),
        ("§6.1 Synthetic: sparse", "synthetic_sparse.json"),
        ("§6.1 Synthetic: dense", "synthetic_dense.json"),
        ("§6.1 Synthetic: de_novo", "synthetic_de_novo.json"),
        ("§6.1 Synthetic: rna_discordant", "synthetic_rna_discordant.json"),
        ("§6.1.1 Pseudobulk Stress", "pseudobulk_stress.json"),
        ("§6.3 Perturbation (σ=0.03)", "perturbation_0.03.json"),
        ("§6.3 Perturbation (σ=0.05)", "perturbation_0.05.json"),
        ("§6.3 Perturbation (σ=0.07)", "perturbation_0.07.json"),
        ("§6.4 Permutation Null", "permutation_null.json"),
        ("§5 bMIND Concordance", "bmind_concordance.json"),
    ]

    for label, filename in files:
        result = _load_result(filename)
        if result is None:
            status = "NOT RUN"
        elif result.passed is None:
            status = "INFO"
        elif result.passed:
            status = "PASS"
        else:
            status = "FAIL"

        detail = result.detail[:60] if result else ""
        print(f"  {label:40s} [{status:7s}]  {detail}")


def _print_model_summary(model: ModelFit) -> None:
    """Print fitted model diagnostics before running validations (§8.4 transparency)."""
    n = len(model.site_params)
    print(f"\nFitted model summary ({n} sites)")
    print("-" * 50)
    print(f"  Tweedie power (p): {model.p}")

    if model.p <= 1.05 or model.p >= 1.95:
        print(f"  WARNING: p={model.p} is near the boundary of the Tweedie family")

    # phi distribution
    phis = np.array([sp.phi for sp in model.site_params])
    print(f"  phi: median={np.median(phis):.4f}, "
          f"IQR=[{np.percentile(phis, 25):.4f}, {np.percentile(phis, 75):.4f}], "
          f"range=[{phis.min():.4f}, {phis.max():.4f}]")

    # rho distribution
    rhos = np.array([sp.rho for sp in model.site_params])
    rho_nz = np.sum(np.abs(rhos) > 1e-6)
    print(f"  rho: median={np.median(rhos):.4f}, "
          f"IQR=[{np.percentile(rhos, 25):.4f}, {np.percentile(rhos, 75):.4f}], "
          f"nonzero={rho_nz}/{n} ({100*rho_nz/n:.1f}%)")

    if rho_nz < n * 0.05:
        print(f"  WARNING: <5% of sites have nonzero rho — RNA covariate may be uninformative")

    # Convergence
    if model.converged is not None:
        n_conv = int(model.converged.sum())
        print(f"  Convergence: {n_conv}/{n} ({100*n_conv/n:.1f}%)")

    # Sparsity per cell type
    est_types = config.SAP_ESTIMATED_CELLTYPES
    print(f"  Nonzero beta groups per cell type:")
    for k, ct in enumerate(est_types):
        nz = sum(1 for sp in model.site_params if np.linalg.norm(sp.beta[k]) > 1e-6)
        print(f"    {ct:20s}: {nz}/{n} ({100*nz/n:.1f}%)")

    # CV result
    if model.cv_result is not None:
        cv = model.cv_result
        print(f"  LOCO-CV hyperparameters:")
        print(f"    lambda (per-stratum): {cv.best_lambda}")
        print(f"    lambda_rho: {cv.best_lambda_rho}")
        print(f"    eta: {cv.best_eta}")
        print(f"    gamma: {cv.best_gamma}")
    print()


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAP Validation Suite (§5, §6.1–§6.5)",
    )
    parser.add_argument("--all", action="store_true", help="Run all validations")
    parser.add_argument("--residual-orth", action="store_true", help="§6.5: Residual orthogonality")
    parser.add_argument("--cross-modality", action="store_true", help="§6.2: Cross-modality concordance")
    parser.add_argument("--synthetic", action="store_true", help="§6.1: Synthetic phospho-validation")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "mdes", "sparse", "dense", "de_novo", "rna_discordant"],
                        help="§6.1 scenario (default: all)")
    parser.add_argument("--pseudobulk", action="store_true", help="§6.1.1: Pseudobulk stress test")
    parser.add_argument("--perturbation", action="store_true", help="§6.3: Perturbation audit")
    parser.add_argument("--permutation", action="store_true", help="§6.4: Permutation null")
    parser.add_argument("--bmind", action="store_true", help="§5: bMIND benchmark")
    parser.add_argument("--model-file", type=str, default=None, help="Path to fitted model")
    parser.add_argument("--n-workers", type=int, default=None, help="Parallelism")
    parser.add_argument("--site-subsample", type=float, default=None,
                        help="Site subsampling fraction for expensive validations")
    parser.add_argument("--summary", action="store_true", help="Print cached results summary")
    args = parser.parse_args()

    if args.summary:
        print_validation_summary()
        return

    # Determine which validations to run
    run_any = (args.all or args.residual_orth or args.cross_modality or
               args.synthetic or args.pseudobulk or args.perturbation or
               args.permutation or args.bmind)

    if not run_any:
        parser.print_help()
        return

    # Load data and model
    from sap_data import load_all
    needs_rna = (args.all or args.cross_modality or args.pseudobulk or args.bmind or
                 args.synthetic or args.perturbation or args.permutation)
    print("Loading data...")
    data, report = load_all(include_rna=needs_rna)

    print("Loading fitted model...")
    model = load_model(args.model_file)
    _print_model_summary(model)

    n_workers = args.n_workers or config.N_WORKERS
    site_sub = args.site_subsample

    # Run validations
    if args.all or args.residual_orth:
        validate_residual_orthogonality(data, model)

    if args.all or args.cross_modality:
        validate_cross_modality_concordance(data, model)

    if args.all or args.synthetic:
        validate_synthetic_phospho(data, model, scenario=args.scenario,
                                   n_workers=n_workers, site_subsample=site_sub)

    if args.all or args.pseudobulk:
        validate_pseudobulk_stress(data, model, n_workers=n_workers,
                                   site_subsample=site_sub)

    if args.all or args.perturbation:
        validate_perturbation_audit(data, model, n_workers=n_workers,
                                    site_subsample=site_sub or 0.2)

    if args.all or args.permutation:
        validate_permutation_null(data, model, n_workers=n_workers,
                                  site_subsample=site_sub or 0.2)

    if args.all or args.bmind:
        validate_bmind_benchmark(data, model)

    # Print summary
    print_validation_summary()


if __name__ == "__main__":
    main()
