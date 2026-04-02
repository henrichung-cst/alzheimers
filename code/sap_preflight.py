"""
Pre-flight diagnostics for the Joint Kinase-Activity Factor Model.

Implements sap_extension.md §4: four diagnostics that must pass before
building the joint kinase-activity factor model.

§4.1 — Kinase-substrate matrix (W) conditioning
§4.2 — Joint design matrix (D_joint) conditioning (critical go/no-go)
§4.3 — Cell-type restriction matrix (R) impact
§4.4 — Per-kinase identifiability (leverage profile)

Usage:
    python code/sap_preflight.py              # run all diagnostics
    python code/sap_preflight.py --w-only     # §4.1 only (fast)
    python code/sap_preflight.py --summary    # print cached results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import sap_data
from analysis_utils import get_expression_cache

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
PREFLIGHT_DIR = os.path.join("outputs", "reports", "preflight")

# ---------------------------------------------------------------------------
# §4.1 thresholds (from sap_extension.md)
# ---------------------------------------------------------------------------
W_EFF_RANK_MIN = 50
W_COND_MAX = 1e4
W_COVERAGE_MIN = 0.80

# §4.2 thresholds
DSITE_KAPPA = 302  # per-site κ from production model
DJOINT_KAPPA_FAIL = 1e5
DJOINT_KAPPA_STRONG = 1e3
N_SUBSAMPLE_SITES = 500

# Percentile thresholds for W sparsification
W_PERCENTILES = [10, 25, 50, 75]


# ---------------------------------------------------------------------------
# W matrix construction
# ---------------------------------------------------------------------------

def build_W_matrix(
    site_meta: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build the kinase-substrate scoring matrix W from kinase-library PSSMs.

    Scores each site's motif against every ser/thr kinase PSSM. The motif
    format in site_meta is 13 characters (6 + phospho-site + 6). The PSSMs
    cover 9 positions: -5 to -1 and +1 to +4 relative to the phospho-site.

    Returns:
        W: (M_kinases × J_scored_sites) scoring matrix.
        kinase_names: list of kinase abbreviations (length M).
        site_mask: boolean array (J_total,) — True for scoreable sites.
    """
    import kinase_library.modules.data as kld

    print("Loading kinase-library PSSMs (all ser/thr kinases)...")
    # Returns (M × 207) DataFrame: rows=kinases, cols=position-AA pairs like '-5P'
    all_pssm = kld.get_all_matrices(kin_type="ser_thr")
    kinase_names = list(all_pssm.index)
    M = len(kinase_names)
    print(f"  {M} kinases, PSSM shape: {all_pssm.shape}")

    # PSSM positions: -5,-4,-3,-2,-1,+1,+2,+3,+4
    # In 13-char motif (0-indexed), phospho-site is at index 6:
    #   pos -5 → motif[1], -4 → [2], -3 → [3], -2 → [4], -1 → [5]
    #   pos +1 → motif[7], +2 → [8], +3 → [9], +4 → [10]
    pssm_positions = [-5, -4, -3, -2, -1, 1, 2, 3, 4]
    motif_char_idx = {-5: 1, -4: 2, -3: 3, -2: 4, -1: 5,
                      1: 7, 2: 8, 3: 9, 4: 10}

    # Pre-build PSSM lookup: pssm_arr[kinase_idx, pos_idx, aa_ord] = score
    # Use amino acid ordinal (A=0, ...) for fast indexing
    pssm_cols = list(all_pssm.columns)
    pssm_vals = all_pssm.values  # (M, 207)

    # Build column-name → column-index lookup
    col_lookup = {c: i for i, c in enumerate(pssm_cols)}

    motifs = site_meta["motif"].values
    J = len(motifs)

    # Identify scoreable sites: must be 13 chars, no NaN
    site_mask = np.zeros(J, dtype=bool)
    for j in range(J):
        m = motifs[j]
        if isinstance(m, str) and len(m) >= 13:
            site_mask[j] = True

    n_matched = int(site_mask.sum())
    print(f"  Scoreable sites: {n_matched}/{J} ({100 * n_matched / J:.1f}%)")

    matched_indices = np.where(site_mask)[0]
    W = np.zeros((M, n_matched), dtype=np.float64)

    # Score each site against all kinases
    for j_out, j_in in enumerate(matched_indices):
        motif = motifs[j_in]
        for pos_i, pos in enumerate(pssm_positions):
            char_idx = motif_char_idx[pos]
            if char_idx >= len(motif):
                continue
            aa = motif[char_idx]
            if aa == "_" or aa == "x":
                continue
            col_name = f"{pos}{aa}"
            ci = col_lookup.get(col_name)
            if ci is not None:
                W[:, j_out] += pssm_vals[:, ci]

    print(f"  W shape: {W.shape}")
    print(f"  Score range: [{W.min():.2f}, {W.max():.2f}]")
    print(f"  Score mean: {W.mean():.2f}, std: {W.std():.2f}")

    return W, kinase_names, site_mask


def threshold_W(W: np.ndarray, percentile: float) -> np.ndarray:
    """Zero out entries below the given percentile of each site's scores.

    For each site (column), compute the percentile of its score distribution
    and zero out entries below that threshold. This sparsifies the dense
    PSSM scoring matrix.
    """
    W_thresh = W.copy()
    for j in range(W.shape[1]):
        col = W[:, j]
        thresh_val = np.percentile(col, percentile)
        W_thresh[col < thresh_val, j] = 0.0
    return W_thresh


# ---------------------------------------------------------------------------
# R matrix construction (cell-type restriction)
# ---------------------------------------------------------------------------

def build_R_matrix(
    kinase_names: List[str],
    data: sap_data.SAPData,
    allen_cache: Dict,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Construct the cell-type restriction matrix R from snRNA-seq + Allen.

    For each kinase m and estimated cell type k (K=5):
      R[m,k] = 1 if kinase gene is expressed in cell type k per snRNA-seq
               AND is brain-expressed per Allen Atlas (or has no Allen entry).

    Uses the pre-CPM-normalized aggexp to check raw expression, not the
    centered/scaled gkp (which loses magnitude information).

    Returns:
        R: (M × K) binary matrix.
        annotation: DataFrame with per-kinase expression details.
    """
    cell_types = config.SAP_ESTIMATED_CELLTYPES  # 5 types
    K = len(cell_types)
    M = len(kinase_names)

    # Reload aggexp at CPM level (before centering) to check expression
    print("Loading aggexp for R matrix construction...")
    aggexp_pooled = sap_data.load_aggexp_pooled()
    sample_map = data.sample_id_map
    if sample_map is None:
        sample_map = sap_data.build_aggexp_sample_map(aggexp_pooled)
    df = sap_data._remap_aggexp_samples(aggexp_pooled, sample_map)
    df = df.loc[df.index.get_level_values("cell_type").isin(cell_types)]
    df = sap_data._cpm_normalize(df)

    # Map kinase names to mouse case for matching aggexp columns
    kinase_mouse_map = {}
    for g in kinase_names:
        mouse = g[0].upper() + g[1:].lower() if len(g) > 1 else g.upper()
        kinase_mouse_map[g] = mouse

    # Also check the mapping cache for alternative symbols
    mapping_overrides = {}
    if os.path.exists(config.MAPPING_CACHE_FILE):
        mc = pd.read_csv(config.MAPPING_CACHE_FILE, index_col=0)
        if "gene_symbol" in mc.columns:
            for kinase_abbrev, row in mc.iterrows():
                gene = row["gene_symbol"]
                if pd.notna(gene):
                    mapping_overrides[str(kinase_abbrev).upper()] = gene

    aggexp_genes = set(df.columns)

    R = np.zeros((M, K), dtype=np.int32)
    annotations = []

    for m_idx, kinase in enumerate(kinase_names):
        # Resolve gene name
        mouse_name = kinase_mouse_map[kinase]
        # Check mapping cache override
        if kinase in mapping_overrides:
            alt_name = mapping_overrides[kinase]
            gene_name = alt_name if alt_name in aggexp_genes else mouse_name
        else:
            gene_name = mouse_name

        in_aggexp = gene_name in aggexp_genes

        # Allen brain-wide check
        allen_key = gene_name  # Allen cache keys use the gene symbol
        # Try both mouse case and original
        allen_info = allen_cache.get(allen_key) or allen_cache.get(kinase) or {}
        allen_expressed = allen_info.get("expressed", 1)  # default permissive
        if isinstance(allen_expressed, str):
            allen_expressed = allen_expressed.lower() in ("1", "true", "yes")

        ct_expressed = {}
        for k_idx, ct in enumerate(cell_types):
            if not in_aggexp:
                ct_expressed[ct] = False
                continue
            if not allen_expressed:
                ct_expressed[ct] = False
                continue
            # Check: CPM > 1 in >= 50% of samples for this cell type
            ct_vals = df.loc[ct, gene_name]
            frac_detected = (ct_vals > 1.0).mean()
            expressed = frac_detected >= 0.5
            ct_expressed[ct] = expressed
            if expressed:
                R[m_idx, k_idx] = 1

        annotations.append({
            "kinase": kinase,
            "gene_name": gene_name,
            "in_aggexp": in_aggexp,
            "allen_expressed": bool(allen_expressed),
            **{f"R_{ct}": ct_expressed.get(ct, False) for ct in cell_types},
            "R_sum": int(R[m_idx].sum()),
        })

    ann_df = pd.DataFrame(annotations)
    n_ones = R.sum()
    n_total = M * K
    print(f"  R matrix: {M} kinases × {K} cell types")
    print(f"  Entries = 1: {n_ones}/{n_total} ({100 * n_ones / n_total:.1f}%)")
    print(f"  Kinases in aggexp: {ann_df['in_aggexp'].sum()}/{M}")
    print(f"  Kinases with R_sum=0 (no expression): "
          f"{(ann_df['R_sum'] == 0).sum()}/{M}")

    return R, ann_df


# ---------------------------------------------------------------------------
# Joint design matrix construction
# ---------------------------------------------------------------------------

def _stratified_site_subsample(
    x_base: pd.DataFrame,
    site_mask: np.ndarray,
    n_sites: int = N_SUBSAMPLE_SITES,
    seed: int = 42,
) -> np.ndarray:
    """Select a stratified subsample of sites by intensity quartile.

    Returns indices into the matched-site array (not the full site array).
    """
    rng = np.random.default_rng(seed)
    # Mean intensity across cell types for matched sites
    matched_indices = np.where(site_mask)[0]
    mean_intensity = x_base.iloc[matched_indices].mean(axis=1).values
    quartiles = np.digitize(
        mean_intensity,
        np.percentile(mean_intensity, [25, 50, 75]),
    )
    n_per_q = n_sites // 4
    selected = []
    for q in range(4):
        q_indices = np.where(quartiles == q)[0]
        n_take = min(n_per_q, len(q_indices))
        selected.extend(rng.choice(q_indices, size=n_take, replace=False))
    # Fill remainder from full pool if needed
    if len(selected) < n_sites:
        remaining = list(set(range(len(matched_indices))) - set(selected))
        n_extra = n_sites - len(selected)
        selected.extend(rng.choice(remaining, size=min(n_extra, len(remaining)),
                                   replace=False))
    return np.array(sorted(selected[:n_sites]))


def build_joint_design(
    a_obs: pd.DataFrame,
    W: np.ndarray,
    R: np.ndarray,
    sample_meta: pd.DataFrame,
    sub_site_idx: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[str, str, str]]]:
    """Build the joint design matrix D_joint for the factor model.

    For the site subsample, the observation vector is vec(Y_{i,j}) across
    (samples × sites). For each active (kinase, cell_type, factorial_component)
    triple, the design column is vec(A_{i,k} * W_{m,j} * 1[c(i)]).

    Only main effects (App, Tau) are included — interaction dropped per §3.1.

    Returns:
        D_joint: (N_obs × P_params) design matrix, column-standardized.
        param_labels: list of (kinase, cell_type, factorial) tuples.
    """
    cell_types = config.SAP_ESTIMATED_CELLTYPES
    conditions = sample_meta["condition"].values
    # Main effects only (no interaction)
    factorial_labels = ["App", "Tau"]
    factorial_indicators = {
        "App": np.array([config.SAP_FACTORIAL[c][0] for c in conditions], dtype=float),
        "Tau": np.array([config.SAP_FACTORIAL[c][1] for c in conditions], dtype=float),
    }

    N = len(sample_meta)  # 24
    J_sub = len(sub_site_idx)
    N_obs = N * J_sub

    # Identify active (kinase, cell_type) pairs where R=1
    M = W.shape[0]
    param_labels = []
    for m in range(M):
        for k_idx, ct in enumerate(cell_types):
            if R[m, k_idx] == 1:
                for fc in factorial_labels:
                    param_labels.append((m, ct, fc))

    P = len(param_labels)
    print(f"  D_joint dimensions: {N_obs} obs × {P} params "
          f"({M} kinases, {len(cell_types)} cell types, "
          f"{len(factorial_labels)} factorial components)")

    # Build D_joint column by column
    D = np.zeros((N_obs, P), dtype=np.float64)
    A_vals = a_obs[cell_types].values  # (24, 5)
    W_sub = W[:, sub_site_idx]  # (M, J_sub)

    for p_idx, (m, ct, fc) in enumerate(param_labels):
        k_idx = cell_types.index(ct)
        a_k = A_vals[:, k_idx]  # (24,)
        w_m = W_sub[m, :]       # (J_sub,)
        fc_ind = factorial_indicators[fc]  # (24,)

        # Column = vec(A_{i,k} * W_{m,j} * 1[c(i)]) for i=0..N-1, j=0..J_sub-1
        col = np.outer(a_k * fc_ind, w_m).ravel()  # (N * J_sub,)
        D[:, p_idx] = col

    # Column-standardize
    norms = np.linalg.norm(D, axis=0)
    norms[norms == 0] = 1.0
    D /= norms

    return D, param_labels


# ---------------------------------------------------------------------------
# §4.1: Kinase-substrate matrix conditioning
# ---------------------------------------------------------------------------

def diag_W_conditioning(W: np.ndarray, kinase_names: List[str]) -> Dict:
    """§4.1: SVD analysis of W at multiple thresholds."""
    print("\n" + "=" * 60)
    print("§4.1: Kinase-Substrate Matrix (W) Conditioning")
    print("=" * 60)

    results = {"thresholds": {}}

    for pct in W_PERCENTILES:
        W_t = threshold_W(W, pct)
        # SVD (compute on transposed if M < J for efficiency)
        sv = np.linalg.svd(W_t, compute_uv=False)
        sv = sv[sv > 1e-15]

        eff_rank = int(np.sum(sv > 0.01 * sv[0])) if len(sv) > 0 else 0
        kappa = float(sv[0] / sv[-1]) if len(sv) > 1 else np.inf

        # Coverage: fraction of sites with at least one nonzero kinase
        site_coverage = float((np.abs(W_t).sum(axis=0) > 0).mean())
        mean_kinases_per_site = float((np.abs(W_t) > 0).sum(axis=0).mean())
        sparsity = float((W_t == 0).mean())

        entry = {
            "effective_rank": eff_rank,
            "condition_number": kappa,
            "site_coverage": site_coverage,
            "mean_kinases_per_site": mean_kinases_per_site,
            "sparsity": sparsity,
            "top_10_sv": sv[:10].tolist() if len(sv) >= 10 else sv.tolist(),
            "n_sv_total": len(sv),
        }
        results["thresholds"][pct] = entry

        passed = eff_rank >= W_EFF_RANK_MIN and kappa < W_COND_MAX and site_coverage > W_COVERAGE_MIN
        status = "PASS" if passed else "FAIL"
        print(f"\n  Percentile {pct}th threshold:")
        print(f"    [{status}] Effective rank: {eff_rank} (threshold: ≥{W_EFF_RANK_MIN})")
        print(f"    [{status}] κ(W): {kappa:.1f} (threshold: <{W_COND_MAX:.0f})")
        print(f"    Site coverage: {site_coverage:.3f} ({100 * site_coverage:.1f}%)")
        print(f"    Mean kinases/site: {mean_kinases_per_site:.1f}")
        print(f"    Sparsity: {100 * sparsity:.1f}%")

    # Overall pass: at least one threshold passes all three criteria
    any_pass = any(
        t["effective_rank"] >= W_EFF_RANK_MIN
        and t["condition_number"] < W_COND_MAX
        and t["site_coverage"] > W_COVERAGE_MIN
        for t in results["thresholds"].values()
    )
    results["passed"] = any_pass
    print(f"\n  §4.1 Overall: {'PASS' if any_pass else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
# §4.2: Joint design matrix conditioning (critical diagnostic)
# ---------------------------------------------------------------------------

def diag_joint_conditioning(
    a_obs: pd.DataFrame,
    W: np.ndarray,
    R: np.ndarray,
    sample_meta: pd.DataFrame,
    x_base: pd.DataFrame,
    site_mask: np.ndarray,
    best_W_pct: int = 25,
) -> Dict:
    """§4.2: SVD of the joint design matrix D_joint."""
    print("\n" + "=" * 60)
    print("§4.2: Joint Design Matrix Conditioning (Critical)")
    print("=" * 60)

    # Subsample sites
    sub_idx = _stratified_site_subsample(x_base, site_mask, N_SUBSAMPLE_SITES)
    print(f"  Using {len(sub_idx)} stratified-subsample sites")

    results = {"thresholds": {}}

    for pct in W_PERCENTILES:
        print(f"\n  W threshold: {pct}th percentile")
        W_t = threshold_W(W, pct)

        D_joint, param_labels = build_joint_design(
            a_obs, W_t, R, sample_meta, sub_idx,
        )

        M_eff = len(param_labels) // 2  # 2 factorial components per active pair
        print(f"  M_eff (active kinase×cell_type pairs): {M_eff}")

        # SVD
        sv = np.linalg.svd(D_joint, compute_uv=False)
        sv = sv[sv > 1e-15]

        eff_rank = int(np.sum(sv > 0.01 * sv[0])) if len(sv) > 0 else 0
        kappa = float(sv[0] / sv[-1]) if len(sv) > 1 else np.inf

        entry = {
            "M_eff": M_eff,
            "n_params": len(param_labels),
            "n_obs": D_joint.shape[0],
            "effective_rank": eff_rank,
            "condition_number": kappa,
            "kappa_ratio_vs_site": kappa / DSITE_KAPPA,
            "eff_rank_fraction": eff_rank / max(len(param_labels), 1),
            "top_20_sv": sv[:20].tolist() if len(sv) >= 20 else sv.tolist(),
            "tail_20_sv": sv[-20:].tolist() if len(sv) >= 20 else sv.tolist(),
            "n_sv_total": len(sv),
        }
        results["thresholds"][pct] = entry

        # Pass criteria
        pass_basic = kappa < DSITE_KAPPA * 10 and eff_rank >= len(param_labels) / 2
        strong_pass = kappa < DJOINT_KAPPA_STRONG and eff_rank >= 0.8 * len(param_labels)
        fail = kappa > DJOINT_KAPPA_FAIL or eff_rank < len(param_labels) / 4

        if strong_pass:
            verdict = "STRONG PASS"
        elif pass_basic:
            verdict = "PASS"
        elif fail:
            verdict = "FAIL"
        else:
            verdict = "MARGINAL"

        entry["verdict"] = verdict

        print(f"    κ(D_joint): {kappa:.1f} (per-site κ: {DSITE_KAPPA})")
        print(f"    Effective rank: {eff_rank}/{len(param_labels)} "
              f"({100 * eff_rank / max(len(param_labels), 1):.1f}%)")
        print(f"    [{verdict}]")

    # Overall verdict: use the best threshold
    best_verdict = "FAIL"
    for entry in results["thresholds"].values():
        v = entry["verdict"]
        if v == "STRONG PASS":
            best_verdict = "STRONG PASS"
            break
        elif v == "PASS" and best_verdict != "STRONG PASS":
            best_verdict = "PASS"
        elif v == "MARGINAL" and best_verdict == "FAIL":
            best_verdict = "MARGINAL"

    results["overall_verdict"] = best_verdict
    print(f"\n  §4.2 Overall: {best_verdict}")

    return results


# ---------------------------------------------------------------------------
# §4.3: Cell-type restriction impact
# ---------------------------------------------------------------------------

def diag_restriction_impact(
    a_obs: pd.DataFrame,
    W: np.ndarray,
    R: np.ndarray,
    sample_meta: pd.DataFrame,
    x_base: pd.DataFrame,
    site_mask: np.ndarray,
    kinase_names: List[str],
    best_pct: int = 25,
) -> Dict:
    """§4.3: Compare D_joint conditioning with R vs. all-ones R."""
    print("\n" + "=" * 60)
    print("§4.3: Cell-Type Restriction Impact")
    print("=" * 60)

    sub_idx = _stratified_site_subsample(x_base, site_mask, N_SUBSAMPLE_SITES)
    W_t = threshold_W(W, best_pct)

    # With restriction
    D_r, labels_r = build_joint_design(a_obs, W_t, R, sample_meta, sub_idx)
    sv_r = np.linalg.svd(D_r, compute_uv=False)
    sv_r = sv_r[sv_r > 1e-15]
    kappa_r = float(sv_r[0] / sv_r[-1]) if len(sv_r) > 1 else np.inf
    eff_rank_r = int(np.sum(sv_r > 0.01 * sv_r[0])) if len(sv_r) > 0 else 0

    # Without restriction (all-ones R)
    M = W.shape[0]
    K = len(config.SAP_ESTIMATED_CELLTYPES)
    R_ones = np.ones((M, K), dtype=np.int32)
    print("  Running with R = all-ones (no restriction)...")
    D_all, labels_all = build_joint_design(a_obs, W_t, R_ones, sample_meta, sub_idx)
    sv_all = np.linalg.svd(D_all, compute_uv=False)
    sv_all = sv_all[sv_all > 1e-15]
    kappa_all = float(sv_all[0] / sv_all[-1]) if len(sv_all) > 1 else np.inf
    eff_rank_all = int(np.sum(sv_all > 0.01 * sv_all[0])) if len(sv_all) > 0 else 0

    M_eff_r = len(labels_r) // 2
    M_eff_all = len(labels_all) // 2

    results = {
        "with_R": {
            "M_eff": M_eff_r,
            "n_params": len(labels_r),
            "kappa": kappa_r,
            "effective_rank": eff_rank_r,
        },
        "all_ones": {
            "M_eff": M_eff_all,
            "n_params": len(labels_all),
            "kappa": kappa_all,
            "effective_rank": eff_rank_all,
        },
        "M_eff_reduction_pct": 100 * (1 - M_eff_r / max(M_eff_all, 1)),
        "kappa_improvement": kappa_all / max(kappa_r, 1e-10),
    }

    print(f"\n  {'Metric':<25} {'With R':>12} {'All-ones':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'M_eff':<25} {M_eff_r:>12} {M_eff_all:>12}")
    print(f"  {'N params':<25} {len(labels_r):>12} {len(labels_all):>12}")
    print(f"  {'κ(D_joint)':<25} {kappa_r:>12.1f} {kappa_all:>12.1f}")
    print(f"  {'Effective rank':<25} {eff_rank_r:>12} {eff_rank_all:>12}")
    print(f"\n  M_eff reduction: {results['M_eff_reduction_pct']:.1f}%")
    print(f"  κ improvement: {results['kappa_improvement']:.2f}×")

    meaningful = results["M_eff_reduction_pct"] >= 30 and results["kappa_improvement"] >= 2
    results["meaningful_improvement"] = meaningful
    print(f"  Tangible benefit: {'YES' if meaningful else 'NO'} "
          f"(threshold: ≥30% M_eff reduction AND ≥2× κ improvement)")

    return results


# ---------------------------------------------------------------------------
# §4.4: Per-kinase identifiability (leverage profile)
# ---------------------------------------------------------------------------

def diag_kinase_leverage(
    a_obs: pd.DataFrame,
    W: np.ndarray,
    R: np.ndarray,
    sample_meta: pd.DataFrame,
    x_base: pd.DataFrame,
    site_mask: np.ndarray,
    kinase_names: List[str],
    R_annotation: pd.DataFrame,
    best_pct: int = 25,
    ridge_lambda: float = 1.0,
) -> pd.DataFrame:
    """§4.4: Per-kinase leverage from the hat matrix of D_joint.

    Computes the Ridge-regularized hat matrix diagonal:
        h_p = D_p' (D'D + λI)^{-1} D_p
    for each parameter p, then sums across factorial components per (kinase, cell_type).
    """
    print("\n" + "=" * 60)
    print("§4.4: Per-Kinase Identifiability Profile")
    print("=" * 60)

    sub_idx = _stratified_site_subsample(x_base, site_mask, N_SUBSAMPLE_SITES)
    W_t = threshold_W(W, best_pct)
    D, param_labels = build_joint_design(a_obs, W_t, R, sample_meta, sub_idx)

    # Ridge-regularized hat matrix diagonal
    # H = D (D'D + λI)^{-1} D'
    # Diagonal: h_p = sum_i D_{i,p}^2 * [(D'D + λI)^{-1}]_{p,p}
    # More efficient: compute via SVD
    print("  Computing Ridge-regularized leverage...")
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    # hat diagonal for each param: h_p = sum_r (V_{p,r} * sv_r / (sv_r^2 + λ))^2 * N_obs
    # Actually: H = V diag(sv^2/(sv^2+λ)) V'
    # diagonal of V diag(d) V' = sum_r V_{p,r}^2 * d_r
    d = sv ** 2 / (sv ** 2 + ridge_lambda)
    V = Vt.T  # (P, min(N_obs, P))
    leverage = (V ** 2) @ d  # (P,)

    # Aggregate per (kinase, cell_type): sum leverage across factorial components
    cell_types = config.SAP_ESTIMATED_CELLTYPES
    kinase_ct_leverage = {}
    for p_idx, (m, ct, fc) in enumerate(param_labels):
        key = (m, ct)
        kinase_ct_leverage[key] = kinase_ct_leverage.get(key, 0.0) + leverage[p_idx]

    # Build results table
    rows = []
    for (m_idx, ct), lev in kinase_ct_leverage.items():
        kinase = kinase_names[m_idx]
        # Count substrates for this kinase in the subsample
        w_col = np.abs(W_t[m_idx, sub_idx])
        n_substrates_sub = int((w_col > 0).sum())
        n_substrates_all = int((np.abs(W_t[m_idx, :]) > 0).sum())
        r_row_sum = int(R[m_idx].sum())

        rows.append({
            "kinase": kinase,
            "cell_type": ct,
            "leverage": float(lev),
            "n_substrates_subsample": n_substrates_sub,
            "n_substrates_total": n_substrates_all,
            "R_row_sum": r_row_sum,
        })

    df = pd.DataFrame(rows).sort_values("leverage", ascending=False)

    # Per-kinase aggregate (sum across cell types)
    kinase_agg = df.groupby("kinase").agg(
        total_leverage=("leverage", "sum"),
        n_active_cell_types=("cell_type", "count"),
        n_substrates_total=("n_substrates_total", "first"),
    ).sort_values("total_leverage", ascending=False)

    # Per-kinase × per-cell-type pivot for the breakdown analysis
    ct_pivot = df.pivot_table(
        index="kinase", columns="cell_type", values="leverage", fill_value=0.0,
    )
    # Compute concentration: max cell-type leverage / total leverage
    kinase_agg["max_ct_leverage"] = ct_pivot.max(axis=1)
    kinase_agg["max_ct_name"] = ct_pivot.idxmax(axis=1)
    kinase_agg["concentration"] = (
        kinase_agg["max_ct_leverage"] / kinase_agg["total_leverage"].clip(lower=1e-10)
    )
    kinase_agg = kinase_agg.sort_values("total_leverage", ascending=False)

    # Print top 20 with per-cell-type breakdown
    ct_short = {
        "Excitatory_neurons": "Exc",
        "Oligodendrocytes": "Oligo",
        "GABAergic_neurons": "GABA",
        "Astrocytes": "Astro",
        "Microglia": "Micro",
    }
    ct_order = list(ct_short.keys())

    header_cts = "  ".join(f"{ct_short[c]:>5}" for c in ct_order)
    print(f"\n  Top 20 kinases by total leverage (per-cell-type breakdown):")
    print(f"  {'Kinase':<10} {'Total':>7} {header_cts}  {'Conc':>5} {'Dominant':<8} {'Subs':>6}")
    print(f"  {'-'*10} {'-'*7} {'  '.join('-'*5 for _ in ct_order)}  {'-'*5} {'-'*8} {'-'*6}")
    for _, row in kinase_agg.head(20).iterrows():
        ct_vals = "  ".join(
            f"{ct_pivot.loc[row.name, c]:>5.3f}" if c in ct_pivot.columns
            and row.name in ct_pivot.index else f"{'---':>5}"
            for c in ct_order
        )
        dom = ct_short.get(row["max_ct_name"], row["max_ct_name"][:5])
        print(f"  {row.name:<10} {row['total_leverage']:>7.4f} {ct_vals}  "
              f"{row['concentration']:>5.0%} {dom:<8} {row['n_substrates_total']:>6.0f}")

    print(f"\n  Bottom 20 kinases by total leverage:")
    print(f"  {'Kinase':<10} {'Total':>7} {header_cts}  {'Conc':>5} {'Dominant':<8} {'Subs':>6}")
    print(f"  {'-'*10} {'-'*7} {'  '.join('-'*5 for _ in ct_order)}  {'-'*5} {'-'*8} {'-'*6}")
    for _, row in kinase_agg.tail(20).iterrows():
        ct_vals = "  ".join(
            f"{ct_pivot.loc[row.name, c]:>5.3f}" if c in ct_pivot.columns
            and row.name in ct_pivot.index else f"{'---':>5}"
            for c in ct_order
        )
        dom = ct_short.get(row["max_ct_name"], row["max_ct_name"][:5])
        print(f"  {row.name:<10} {row['total_leverage']:>7.4f} {ct_vals}  "
              f"{row['concentration']:>5.0%} {dom:<8} {row['n_substrates_total']:>6.0f}")

    # Summary statistics on concentration
    conc = kinase_agg["concentration"]
    n_balanced = int((conc < 0.40).sum())
    n_dominated = int((conc > 0.60).sum())
    n_total_k = len(kinase_agg)
    print(f"\n  Leverage concentration summary ({n_total_k} kinases with R > 0):")
    print(f"    Balanced (max CT < 40% of total):   {n_balanced} kinases")
    print(f"    Moderate (40-60%):                   {n_total_k - n_balanced - n_dominated} kinases")
    print(f"    Dominated (max CT > 60% of total):   {n_dominated} kinases")

    # Dominant cell-type distribution
    dom_counts = kinase_agg["max_ct_name"].value_counts()
    print(f"\n  Dominant cell type distribution:")
    for ct_name in ct_order:
        n = dom_counts.get(ct_name, 0)
        print(f"    {ct_short[ct_name]:<8} {n:>4} kinases")

    # Save the full pivot for downstream use
    leverage_ct_path = os.path.join(PREFLIGHT_DIR, "kinase_leverage_by_celltype.csv")
    export_df = kinase_agg.join(ct_pivot, how="left")
    export_df.to_csv(leverage_ct_path)
    print(f"\n  Saved: {leverage_ct_path}")

    return df


# ---------------------------------------------------------------------------
# SVD spectrum plots
# ---------------------------------------------------------------------------

def _plot_svd_spectra(results_41: Dict, output_dir: str):
    """Plot SVD spectra for W at different thresholds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full spectrum
    ax = axes[0]
    for pct in W_PERCENTILES:
        entry = results_41["thresholds"][pct]
        svs = entry["top_10_sv"]
        ax.semilogy(range(len(svs)), svs, "o-", label=f"{pct}th pct", markersize=3)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("W matrix: top 10 singular values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: summary metrics
    ax = axes[1]
    pcts = sorted(results_41["thresholds"].keys())
    eff_ranks = [results_41["thresholds"][p]["effective_rank"] for p in pcts]
    coverages = [results_41["thresholds"][p]["site_coverage"] for p in pcts]
    ax.bar([str(p) for p in pcts], eff_ranks, alpha=0.7, label="Effective rank")
    ax.axhline(W_EFF_RANK_MIN, color="red", linestyle="--", label=f"Threshold ({W_EFF_RANK_MIN})")
    ax.set_xlabel("Percentile threshold")
    ax.set_ylabel("Effective rank")
    ax.set_title("W effective rank vs threshold")
    ax.legend()

    # Coverage on twin axis
    ax2 = ax.twinx()
    ax2.plot([str(p) for p in pcts], coverages, "ro-", label="Coverage")
    ax2.axhline(W_COVERAGE_MIN, color="orange", linestyle=":", label=f"Min coverage ({W_COVERAGE_MIN})")
    ax2.set_ylabel("Site coverage")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "w_svd_spectrum.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_djoint_spectrum(results_42: Dict, output_dir: str):
    """Plot SVD spectrum of D_joint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: singular value spectrum
    ax = axes[0]
    for pct in W_PERCENTILES:
        entry = results_42["thresholds"].get(pct)
        if entry is None:
            continue
        svs = entry["top_20_sv"]
        ax.semilogy(range(len(svs)), svs, "o-", label=f"W {pct}th pct", markersize=3)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("D_joint: top 20 singular values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: condition number comparison
    ax = axes[1]
    pcts = sorted(results_42["thresholds"].keys())
    kappas = [results_42["thresholds"][p]["condition_number"] for p in pcts]
    bars = ax.bar([str(p) for p in pcts], kappas, alpha=0.7)
    ax.axhline(DSITE_KAPPA, color="blue", linestyle="--",
               label=f"Per-site κ ({DSITE_KAPPA})")
    ax.axhline(DSITE_KAPPA * 10, color="red", linestyle="--",
               label=f"Pass threshold ({DSITE_KAPPA * 10})")
    ax.axhline(DJOINT_KAPPA_STRONG, color="green", linestyle="--",
               label=f"Strong pass ({DJOINT_KAPPA_STRONG})")
    ax.set_xlabel("W percentile threshold")
    ax.set_ylabel("κ(D_joint)")
    ax.set_title("Joint design condition number")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "djoint_svd_spectrum.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all_diagnostics(w_only: bool = False) -> Dict:
    """Run all pre-flight diagnostics and save results."""
    os.makedirs(PREFLIGHT_DIR, exist_ok=True)

    # Load data
    print("=" * 60)
    print("Pre-Flight Diagnostics: Joint Kinase-Activity Factor Model")
    print("sap_extension.md §4")
    print("=" * 60)

    print("\n--- Loading data ---")
    if w_only:
        a_obs = sap_data.load_a_obs()
        sample_meta = sap_data.build_sample_meta(list(a_obs.index))
        _, site_meta = sap_data.load_x_base()
        data = None
    else:
        data, diag_report = sap_data.load_all(include_rna=True)
        a_obs = data.a_obs
        sample_meta = data.sample_meta
        site_meta = data.site_meta

    # Build W matrix
    print("\n--- Building W matrix ---")
    W, kinase_names, site_mask = build_W_matrix(site_meta)

    all_results = {}

    # §4.1: W conditioning
    all_results["4.1_W_conditioning"] = diag_W_conditioning(W, kinase_names)

    # Save W SVD plots
    _plot_svd_spectra(all_results["4.1_W_conditioning"], PREFLIGHT_DIR)

    if w_only:
        print("\n--- Skipping §4.2-4.4 (--w-only mode) ---")
    else:
        # Build R matrix
        print("\n--- Building R matrix ---")
        allen_cache = get_expression_cache(config.ALLEN_EXPRESSION_CACHE_FILE)
        R, R_ann = build_R_matrix(kinase_names, data, allen_cache)
        R_ann.to_csv(os.path.join(PREFLIGHT_DIR, "R_annotation.csv"), index=False)
        print(f"  Saved: {os.path.join(PREFLIGHT_DIR, 'R_annotation.csv')}")

        # §4.2: Joint conditioning
        all_results["4.2_joint_conditioning"] = diag_joint_conditioning(
            a_obs, W, R, sample_meta, data.x_base, site_mask,
        )
        _plot_djoint_spectrum(all_results["4.2_joint_conditioning"], PREFLIGHT_DIR)

        # §4.3: Restriction impact
        all_results["4.3_restriction_impact"] = diag_restriction_impact(
            a_obs, W, R, sample_meta, data.x_base, site_mask, kinase_names,
        )

        # §4.4: Leverage profile
        leverage_df = diag_kinase_leverage(
            a_obs, W, R, sample_meta, data.x_base, site_mask,
            kinase_names, R_ann,
        )
        leverage_path = os.path.join(PREFLIGHT_DIR, "kinase_leverage.csv")
        leverage_df.to_csv(leverage_path, index=False)
        print(f"  Saved: {leverage_path}")

    # Save summary
    summary = _build_summary(all_results)
    summary_path = os.path.join(PREFLIGHT_DIR, "preflight_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary: {summary_path}")

    # Print final verdict
    _print_verdict(all_results)

    return all_results


def _build_summary(results: Dict) -> Dict:
    """Build a compact summary dict for JSON serialization."""
    summary = {}
    for section, data in results.items():
        if isinstance(data, dict):
            # Strip large SV arrays for the summary
            compact = {}
            for k, v in data.items():
                if k == "thresholds" and isinstance(v, dict):
                    compact[k] = {}
                    for pct, entry in v.items():
                        compact[k][pct] = {
                            ek: ev for ek, ev in entry.items()
                            if ek not in ("top_10_sv", "top_20_sv", "tail_20_sv")
                        }
                else:
                    compact[k] = v
            summary[section] = compact
        else:
            summary[section] = data
    return summary


def _print_verdict(results: Dict):
    """Print the final go/no-go recommendation."""
    print("\n" + "=" * 60)
    print("PRE-FLIGHT SUMMARY")
    print("=" * 60)

    w_pass = results.get("4.1_W_conditioning", {}).get("passed", False)
    print(f"  §4.1 W conditioning:      {'PASS' if w_pass else 'FAIL'}")

    joint = results.get("4.2_joint_conditioning", {})
    joint_verdict = joint.get("overall_verdict", "NOT RUN")
    print(f"  §4.2 Joint conditioning:  {joint_verdict}")

    restriction = results.get("4.3_restriction_impact", {})
    r_benefit = restriction.get("meaningful_improvement", None)
    if r_benefit is not None:
        print(f"  §4.3 Restriction impact:  {'MEANINGFUL' if r_benefit else 'MARGINAL'}")
    else:
        print(f"  §4.3 Restriction impact:  NOT RUN")

    print(f"  §4.4 Leverage profile:    See kinase_leverage.csv")

    print()
    if joint_verdict in ("STRONG PASS", "PASS"):
        print("  RECOMMENDATION: PROCEED with joint factor model implementation.")
        if joint_verdict == "PASS":
            print("  (Marginal conditioning — expect only a subset of kinases")
            print("   with high leverage to produce validated estimates.)")
    elif joint_verdict == "MARGINAL":
        print("  RECOMMENDATION: PROCEED CAUTIOUSLY.")
        print("  Only kinases with high leverage (§4.4) are likely identifiable.")
    else:
        print("  RECOMMENDATION: DO NOT PROCEED with joint factor model.")
        print("  Fall back to Tier 2-3 annotated inference (sap_extension.md §2.3).")
    print("=" * 60)


def print_cached_summary():
    """Print previously computed results from disk."""
    path = os.path.join(PREFLIGHT_DIR, "preflight_summary.json")
    if not os.path.exists(path):
        print(f"No cached results found at {path}")
        print("Run: python code/sap_preflight.py")
        return
    with open(path) as f:
        summary = json.load(f)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight diagnostics for the joint kinase-activity factor model "
                    "(sap_extension.md §4).",
    )
    parser.add_argument("--w-only", action="store_true",
                        help="Run §4.1 only (W matrix conditioning, no RNA data needed)")
    parser.add_argument("--summary", action="store_true",
                        help="Print cached results from previous run")
    args = parser.parse_args()

    if args.summary:
        print_cached_summary()
        return

    run_all_diagnostics(w_only=args.w_only)


if __name__ == "__main__":
    main()
