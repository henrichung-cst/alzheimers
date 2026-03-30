"""Permutation-based p-value correction for kinase enrichment.

Replaces BH correction with empirical p-values derived from permuting
site labels. Also provides m_eff (effective number of independent tests)
via substrate overlap analysis and Higher Criticism for diffuse signal
detection.
"""

import numpy as np
import pandas as pd
import kinase_library as kl
from concurrent.futures import ProcessPoolExecutor, as_completed
import config


# ---------------------------------------------------------------------------
# Permutation-based empirical p-values
# ---------------------------------------------------------------------------

def _precompute_percentiles(work_df, seq_col, kin_type):
    """Score ALL sites against all kinases once, before permutation loop.

    Returns a DataFrame (index=motif sequences, columns=kinases) of
    percentile values that can be injected via submit_percentiles().
    """
    from kinase_library.objects.phosphoproteomics import PhosphoProteomics
    pps = PhosphoProteomics(work_df, seq_col=seq_col, pp=False,
                            drop_invalid_subs=True,
                            new_seq_phos_res_cols=False,
                            suppress_warnings=True)
    pps.percentile(kin_type=kin_type, values_only=True)
    return getattr(pps, f"{kin_type}_percentiles")


def _run_single_permutation(work_df, seq_col, lfc_col, percent_rank,
                            percent_thresh, kin_type, kl_method, kl_thresh,
                            precomputed_pctiles, seed):
    """Run one Fisher enrichment on permuted data.

    Shuffles LFC values across sites (motifs stay fixed), constructs a
    DiffPhosData, injects pre-computed percentiles to skip re-scoring,
    and returns raw Fisher p-values per kinase.
    """
    rng = np.random.default_rng(seed)
    perm_df = work_df.copy()
    perm_df[lfc_col] = rng.permutation(perm_df[lfc_col].values)

    import warnings, io, contextlib
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        dpd = kl.DiffPhosData(
            perm_df, seq_col=seq_col, lfc_col=lfc_col,
            percent_rank=percent_rank, percent_thresh=percent_thresh,
            suppress_warnings=True,
        )

        # Inject pre-computed percentiles so kinase_enrichment() skips scoring
        if precomputed_pctiles is not None:
            dpd.submit_percentiles(kin_type=kin_type,
                                   percentiles=precomputed_pctiles,
                                   suppress_messages=True)

        enrich = dpd.kinase_enrichment(
            kin_type=kin_type, kl_method=kl_method, kl_thresh=kl_thresh,
        )
    return enrich.combined_enrichment_results["most_sig_fisher_pval"]


def _permutation_worker(args):
    """Picklable wrapper for ProcessPoolExecutor."""
    return _run_single_permutation(*args)


def run_permutation_correction(work_df, observed_pvals, kin_type, pct,
                               n_perm, seed, n_workers,
                               kl_method=None, kl_thresh=None):
    """Compute empirical p-values via permutation of site labels.

    Parameters
    ----------
    work_df : DataFrame
        Must contain 'motif' and 'log2_fold_change' columns.
    observed_pvals : Series
        Raw Fisher p-values per kinase from the real (unpermuted) data.
    kin_type : str
    pct : float
        Percentile threshold (e.g. 5).
    n_perm : int
        Number of permutations (e.g. 1000).
    seed : int
        Random seed for reproducibility.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    empirical_p : Series
        Empirical p-values indexed by kinase name.
    """
    if kl_method is None:
        kl_method = config.KL_METHOD
    if kl_thresh is None:
        kl_thresh = config.KL_THRESH

    seq_col = "motif"
    lfc_col = "log2_fold_change"

    # Pre-compute percentiles for all sites (the expensive step, done once)
    print(f"    Pre-computing percentiles for {len(work_df)} sites...")
    precomputed = _precompute_percentiles(work_df, seq_col, kin_type)

    # Generate deterministic per-permutation seeds
    master_rng = np.random.default_rng(seed)
    perm_seeds = master_rng.integers(0, 2**31, size=n_perm)

    # Build argument tuples for workers
    common_args = (seq_col, lfc_col, config.PERCENT_RANK, pct,
                   kin_type, kl_method, kl_thresh, precomputed)
    tasks = [(work_df, *common_args, int(s)) for s in perm_seeds]

    print(f"    Running {n_perm} permutations ({n_workers} workers)...")
    null_pvals = []

    if n_workers <= 1:
        # Sequential for debugging
        for i, task in enumerate(tasks):
            null_pvals.append(_permutation_worker(task))
            if (i + 1) % 100 == 0:
                print(f"      {i + 1}/{n_perm} permutations complete")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_permutation_worker, t): i
                       for i, t in enumerate(tasks)}
            done = 0
            for future in as_completed(futures):
                null_pvals.append(future.result())
                done += 1
                if done % 100 == 0:
                    print(f"      {done}/{n_perm} permutations complete")

    null_matrix = pd.DataFrame(null_pvals)  # (n_perm, n_kinases)

    # Empirical p-value: (count of null p <= observed p, +1) / (N + 1)
    kinases = observed_pvals.index
    empirical_p = pd.Series(index=kinases, dtype=float)
    for k in kinases:
        if k in null_matrix.columns:
            count = (null_matrix[k].values <= observed_pvals[k]).sum()
            empirical_p[k] = (count + 1) / (n_perm + 1)
        else:
            empirical_p[k] = 1.0  # kinase not in null (shouldn't happen)

    print(f"    Permutation complete. Empirical p range: "
          f"[{empirical_p.min():.4f}, {empirical_p.max():.4f}]")
    return empirical_p


# ---------------------------------------------------------------------------
# m_eff: effective number of independent tests (Galwey method)
# ---------------------------------------------------------------------------

def compute_meff(kin_type, kl_method=None, kl_thresh=None, cache_path="data/meff_cache.csv"):
    """Calculate effective number of independent kinase tests.

    Builds a Jaccard similarity matrix from kinase substrate overlap,
    performs eigenvalue decomposition, and applies the Galwey method.
    Result is cached since it depends only on the kinase-library scoring
    matrix, not on any particular comparison.
    """
    import os

    if kl_method is None:
        kl_method = config.KL_METHOD
    if kl_thresh is None:
        kl_thresh = config.KL_THRESH

    # Check cache
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        row = cached[
            (cached["kin_type"] == kin_type) &
            (cached["kl_thresh"] == kl_thresh)
        ]
        if len(row) > 0:
            m = int(row.iloc[0]["m_eff"])
            print(f"  m_eff = {m} (cached)")
            return m

    print(f"  Computing m_eff for {kin_type} at kl_thresh={kl_thresh}...")

    # Get all kinases and a reference phosphoproteome to score against
    from kinase_library.objects.phosphoproteomics import PhosphoProteomics
    from kinase_library.modules import data as kl_data

    kinases = kl_data.get_kinase_list(kin_type=kin_type)
    phosprot = kl_data.get_phosphoproteome()
    phosprot_pps = PhosphoProteomics(phosprot, pp=False,
                                     drop_invalid_subs=True,
                                     new_seq_phos_res_cols=False,
                                     suppress_warnings=True)
    phosprot_pps.percentile(kin_type=kin_type, kinases=kinases, values_only=True)
    pctiles = getattr(phosprot_pps, f"{kin_type}_percentiles")

    # For each kinase, define "substrates" as sites with percentile >= kl_thresh
    substrate_sets = {}
    for kin in kinases:
        if kin in pctiles.columns:
            substrate_sets[kin] = set(pctiles.index[pctiles[kin] >= kl_thresh])
        else:
            substrate_sets[kin] = set()

    kin_list = [k for k in kinases if k in substrate_sets and len(substrate_sets[k]) > 0]
    n = len(kin_list)

    # Build Jaccard similarity matrix
    jaccard = np.zeros((n, n))
    for i in range(n):
        si = substrate_sets[kin_list[i]]
        for j in range(i, n):
            sj = substrate_sets[kin_list[j]]
            union = len(si | sj)
            if union > 0:
                jac = len(si & sj) / union
            else:
                jac = 0.0
            jaccard[i, j] = jaccard[j, i] = jac

    # Eigenvalue decomposition → Galwey method
    eigenvalues = np.linalg.eigvalsh(jaccard)
    eigenvalues = eigenvalues[eigenvalues > 0]
    m_eff = int(np.ceil(
        (np.sum(np.sqrt(eigenvalues))) ** 2 / np.sum(eigenvalues)
    ))

    print(f"  m_eff = {m_eff} (from {n} kinases with substrates)")

    # Cache result
    row = pd.DataFrame([{
        "kin_type": kin_type, "kl_thresh": kl_thresh,
        "n_kinases": n, "m_eff": m_eff,
    }])
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(cache_path, index=False)

    return m_eff


def apply_meff_bh(raw_pvals, m_eff):
    """BH correction using m_eff instead of the full number of tests.

    Standard BH: adj_p_i = p_i * m / rank_i (with monotonicity enforcement).
    m_eff BH:    adj_p_i = p_i * m_eff / rank_i.
    """
    pvals = raw_pvals.values.copy()
    n = len(pvals)
    sort_idx = np.argsort(pvals)
    sorted_p = pvals[sort_idx]

    # BH with m_eff
    adj = np.empty(n)
    for i in range(n):
        rank = i + 1
        adj[i] = sorted_p[i] * m_eff / rank

    # Enforce monotonicity (step-up)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    # Unsort
    result = np.empty(n)
    result[sort_idx] = adj

    return pd.Series(result, index=raw_pvals.index)


# ---------------------------------------------------------------------------
# Higher Criticism (Donoho & Jin, 2004)
# ---------------------------------------------------------------------------

def compute_higher_criticism(raw_pvals):
    """Compute the Higher Criticism (HC*) statistic.

    Tests H0: all p-values ~ Uniform(0,1)
    vs   H1: a sparse subset are stochastically smaller.

    Parameters
    ----------
    raw_pvals : array-like
        Raw (unadjusted) p-values.

    Returns
    -------
    hc_star : float
        Maximum HC value (restricted to first half of sorted p-values).
    hc_index : int
        1-based index at which HC_star is achieved.
    """
    p_sorted = np.sort(np.asarray(raw_pvals, dtype=float))
    n = len(p_sorted)
    max_idx = max(1, n // 2)

    hc_values = np.full(max_idx, -np.inf)
    for i in range(1, max_idx + 1):
        p_i = p_sorted[i - 1]
        expected = i / n
        denom = np.sqrt(p_i * (1 - p_i))
        if denom > 0 and 0 < p_i < 1:
            hc_values[i - 1] = np.sqrt(n) * (expected - p_i) / denom

    best_idx = int(np.argmax(hc_values))
    hc_star = float(hc_values[best_idx])
    return hc_star, best_idx + 1  # 1-based index
