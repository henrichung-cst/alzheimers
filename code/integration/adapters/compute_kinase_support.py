"""Adapter 5.5: Compute substrate-based kinase support scores for Incytr pathways.

External reranking layer that connects kinase activity evidence (from MEA)
to pathways through kinase-substrate relationships (from kldata), bypassing
Incytr's expression threshold gate.

Dual-channel architecture:
  - Internal (Incytr native): kinases that are pathway nodes scored by SiK/activity
  - External (this adapter): kinases that phosphorylate pathway node genes,
    regardless of whether those kinases are expressed enough to be nodes

Deduplication: kinases already present as EM or Target nodes in a pathway are
excluded from the external score for that pathway (their contribution is
captured by Incytr's internal channel).

Score formula per kinase-substrate edge:
  edge_weight = |NES| x IDF x attribution_weight
  IDF = 1/log(N)  where N = #significant attributed kinases targeting substrate
  attribution_weight = combined_score x cell_type_relevance

Per-pathway score = median(edge_weights).  Median aggregation is robust to
hub-substrate inflation (many weak edges from promiscuous substrates) and
single-outlier dominance (one strong edge in low-degree pathways).

Outputs:
  intermediates/kinase_support_scores.csv
  intermediates/adjusted_rankings.csv
  intermediates/reranking_summary.json
  intermediates/permutation_pvalues.csv  (with --permutations)
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import (load_mouse_gene_to_kinase_mapping,
                    build_substrate_kinase_map, ensure_intermediates_dir)
import config_integration as icfg


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_mea_kinases(contrast, fdr_threshold):
    """Load MEA kinase data for the contrast.

    Returns:
      sig_kinases: dict kinase_abbrev -> NES (FDR < threshold)
      all_kinases: dict kinase_abbrev -> |NES| (all tested, for null model)
    """
    mea = pd.read_csv(icfg.MEA_STOICHIOMETRY_CSV)
    mea_c = mea[mea["contrast"] == contrast]
    all_kinases = dict(zip(mea_c["kinase"], mea_c["NES"].abs()))
    sig = mea_c[mea_c["FDR"] < fdr_threshold]
    sig_kinases = dict(zip(sig["kinase"], sig["NES"]))
    return sig_kinases, all_kinases


def _load_attribution_weights(contrast, sender, receiver, sender_discount):
    """Compute per-kinase attribution weights from unified attribution.

    For each kinase, takes the best weight across relevant cell types:
      receiver attribution: combined_score x 1.0
      sender attribution:   combined_score x sender_discount

    Returns dict: kinase_abbrev -> attribution_weight
    """
    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV)
    attr_c = attr[attr["contrast"] == contrast]

    weights = {}
    for _, row in attr_c.iterrows():
        kin = row["kinase"]
        ct = row["cell_type"]
        score = row["combined_score"]

        if ct == receiver:
            w = score * 1.0
        elif ct == sender:
            w = score * sender_discount
        else:
            continue

        if kin not in weights or w > weights[kin]:
            weights[kin] = w

    return weights


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _compute_idf_map(sub_to_kins, mouse_to_abbrevs, sig_kinases, attr_weights):
    """Pre-compute per-substrate IDF values.

    IDF(substrate) = 1/log(N) where N = number of significant attributed
    kinases targeting that substrate.  Returns 1.0 when N <= 1.
    """
    idf_map = {}
    for sub_gene, kin_genes in sub_to_kins.items():
        n_sig = 0
        for kin_gene in kin_genes:
            for abbrev in mouse_to_abbrevs.get(kin_gene, set()):
                if abbrev in sig_kinases and abbrev in attr_weights:
                    n_sig += 1
        idf_map[sub_gene] = 1.0 / math.log(n_sig) if n_sig > 1 else 1.0
    return idf_map


def compute_scores(pathways, sub_to_kins, idf_map, sig_kinases, attr_weights,
                   mouse_to_abbrevs):
    """Compute kinase support scores for all pathways.

    Uses median edge weight as the per-pathway score.  The median is robust
    to both hub-substrate inflation (many weak edges) and single-outlier
    dominance (one strong edge in a low-degree pathway).  The sum is
    retained as ``kinase_support_score_sum`` for reference.

    Returns DataFrame with per-pathway kinase_support_score and metadata.
    """

    rows = []
    for _, pw in pathways.iterrows():
        path_id = pw["Path"]
        em_gene = pw["EM"]
        tg_gene = pw["Target"]
        tpds = pw["TPDS"]
        pds = pw["PDS"]

        # All node genes for deduplication
        node_genes = {em_gene, tg_gene, pw["Receptor"], pw["Ligand"]}

        edge_weights = []
        edge_weights_no_idf = []
        seen_edges = set()          # (abbrev, sub_gene) to avoid double-count
        unique_kinases = set()
        n_excluded = 0
        nes_signs = []

        for sub_gene in (em_gene, tg_gene):
            sub_idf = idf_map.get(sub_gene, 1.0)

            for kin_gene in sub_to_kins.get(sub_gene, set()):
                # Deduplication: skip kinases that are pathway nodes
                if kin_gene in node_genes:
                    n_excluded += 1
                    continue

                for abbrev in mouse_to_abbrevs.get(kin_gene, set()):
                    if abbrev not in sig_kinases or abbrev not in attr_weights:
                        continue

                    edge_key = (abbrev, sub_gene)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    nes = sig_kinases[abbrev]
                    aw = attr_weights[abbrev]

                    edge_weights.append(abs(nes) * sub_idf * aw)
                    edge_weights_no_idf.append(abs(nes) * aw)
                    unique_kinases.add(abbrev)
                    nes_signs.append(np.sign(nes))

        # Concordance flag: sign(NES consensus) vs sign(TPDS)
        if not nes_signs:
            conc_flag = "none"
        else:
            mean_sign = np.mean(nes_signs)
            tpds_sign = np.sign(tpds)
            if tpds_sign == 0:
                conc_flag = "mixed"
            elif abs(mean_sign) >= 0.5:
                conc_flag = ("concordant"
                             if np.sign(mean_sign) == tpds_sign
                             else "discordant")
            else:
                conc_flag = "mixed"

        rows.append({
            "Path": path_id,
            "kinase_support_score": float(np.median(edge_weights))
                if edge_weights else 0.0,
            "kinase_support_score_sum": sum(edge_weights),
            "kinase_support_score_no_idf": float(np.median(edge_weights_no_idf))
                if edge_weights_no_idf else 0.0,
            "n_distinct_kinases": len(unique_kinases),
            "n_node_kinases_excluded": n_excluded,
            "top_kinases": ";".join(sorted(unique_kinases)[:10]),
            "concordance_flag": conc_flag,
            "TPDS": tpds,
            "PDS": pds,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Adjusted rankings
# ---------------------------------------------------------------------------

def compute_adjusted_rankings(scores_df, lambda_values):
    """Compute adjusted_score = TPDS + lambda * kinase_support_score."""
    out = scores_df[["Path", "TPDS", "kinase_support_score"]].copy()
    out["tpds_rank"] = out["TPDS"].rank(ascending=False, method="min")

    for lam in lambda_values:
        col_s = f"adjusted_score_lam{lam}"
        col_r = f"adjusted_rank_lam{lam}"
        out[col_s] = out["TPDS"] + lam * out["kinase_support_score"]
        out[col_r] = out[col_s].rank(ascending=False, method="min")

    return out


# ---------------------------------------------------------------------------
# Sensitivity analyses
# ---------------------------------------------------------------------------

def run_sensitivity_analyses(scores_df, results_full, lambda_values):
    """PhPDS_ps redundancy, IDF sensitivity, lambda sensitivity."""
    summary = {}

    # 1. PhPDS_ps redundancy
    if "PhPDS_ps" in results_full.columns:
        merged = scores_df.merge(
            results_full[["Path", "PhPDS_ps"]], on="Path", how="left"
        )
        phPDS = pd.to_numeric(merged["PhPDS_ps"], errors="coerce")
        ks = merged["kinase_support_score"]
        mask = phPDS.notna() & ks.notna() & (ks > 0)
        if mask.sum() > 10:
            rho, pval = stats.spearmanr(phPDS[mask], ks[mask])
            summary["spearman_rho_phPDS_vs_kscore"] = round(rho, 4)
            summary["spearman_pval_phPDS_vs_kscore"] = float(f"{pval:.2e}")
            print(f"  PhPDS_ps vs kinase_support_score: rho={rho:.4f}, p={pval:.2e}")

    # 2. IDF sensitivity: top-20 overlap with vs without IDF
    rank_idf = scores_df["kinase_support_score"].rank(
        ascending=False, method="min")
    rank_no_idf = scores_df["kinase_support_score_no_idf"].rank(
        ascending=False, method="min")
    top20_idf = set(scores_df.loc[rank_idf <= 20, "Path"])
    top20_no_idf = set(scores_df.loc[rank_no_idf <= 20, "Path"])
    if top20_idf:
        overlap = len(top20_idf & top20_no_idf) / len(top20_idf)
        summary["idf_top20_overlap"] = round(overlap, 4)
        print(f"  IDF top-20 overlap: {overlap:.1%}")

    # 3. Lambda sensitivity: Kendall tau-b across adjacent values
    adj = compute_adjusted_rankings(scores_df, lambda_values)
    tau_pairs = {}
    for i in range(len(lambda_values) - 1):
        l1, l2 = lambda_values[i], lambda_values[i + 1]
        r1 = adj[f"adjusted_rank_lam{l1}"]
        r2 = adj[f"adjusted_rank_lam{l2}"]
        tau, _ = stats.kendalltau(r1, r2)
        tau_pairs[f"tau_lam{l1}_vs_lam{l2}"] = round(tau, 4)
    summary["lambda_kendall_tau"] = tau_pairs
    if tau_pairs:
        print(f"  Lambda Kendall tau: {tau_pairs}")

    # General statistics
    summary["n_pathways"] = len(scores_df)
    summary["n_nonzero_score"] = int((scores_df["kinase_support_score"] > 0).sum())
    summary["n_zero_score"] = int((scores_df["kinase_support_score"] == 0).sum())
    summary["n_with_excluded_nodes"] = int(
        (scores_df["n_node_kinases_excluded"] > 0).sum())

    return summary


# ---------------------------------------------------------------------------
# Permutation null models
# ---------------------------------------------------------------------------

def run_permutation_tests(pathways, sub_to_kins, idf_map, sig_kinases,
                          attr_weights, mouse_to_abbrevs, all_mea_nes,
                          n_permutations):
    """Dual permutation null models for median-aggregated scores.

    Designed for median aggregation where the question is: "does this
    pathway's kinase evidence reflect enrichment for disease-relevant,
    cell-type-attributed kinases?"

    Null 1 (enrichment null): For each pathway with N edges, sample N
      kinases from the full MEA universe (311 kinases, not just the 134
      significant+attributed).  Sampled kinases use their actual |NES|
      but a uniform attribution weight (median of observed weights).
      Tests whether the pathway's score reflects concentration of
      disease-significant, properly attributed kinases.

    Null 2 (wiring null): Reassign each pathway's edges to random
      kinases drawn from the full MEA universe, keeping edge count and
      IDF coefficients fixed.  Tests whether the specific kinase-
      substrate wiring matters or random connections give similar medians.
    """
    print(f"\nRunning permutation tests ({n_permutations} iterations)...")

    # Full MEA kinase universe (background)
    bg_kinases = sorted(all_mea_nes.keys())
    n_bg = len(bg_kinases)
    bg_nes_vec = np.array([all_mea_nes[k] for k in bg_kinases])
    print(f"  Background kinase pool: {n_bg} (full MEA universe)")

    # Uniform attribution weight for null draws: use median of observed
    median_attr_weight = float(np.median(list(attr_weights.values())))
    print(f"  Null attribution weight (median observed): {median_attr_weight:.4f}")

    # Significant+attributed kinases
    relevant_kinases = sorted(k for k in sig_kinases if k in attr_weights)
    kin_idx = {k: i for i, k in enumerate(relevant_kinases)}
    nes_vec = np.array([abs(sig_kinases[k]) for k in relevant_kinases])

    # Build per-pathway edge lists: (kinase_index, idf_coeff, attr_weight)
    n_pw = len(pathways)
    pw_edges = []

    for _, pw in pathways.iterrows():
        node_genes = {pw["EM"], pw["Target"], pw["Receptor"], pw["Ligand"]}
        seen = set()
        edges = []

        for sub_gene in (pw["EM"], pw["Target"]):
            sub_idf = idf_map.get(sub_gene, 1.0)

            for kin_gene in sub_to_kins.get(sub_gene, set()):
                if kin_gene in node_genes:
                    continue
                for abbrev in mouse_to_abbrevs.get(kin_gene, set()):
                    if abbrev not in kin_idx:
                        continue
                    edge_key = (abbrev, sub_gene)
                    if edge_key in seen:
                        continue
                    seen.add(edge_key)

                    edges.append((kin_idx[abbrev], sub_idf))

        pw_edges.append(edges)

    observed = np.zeros(n_pw)
    for i, edges in enumerate(pw_edges):
        if edges:
            ews = np.array([idf * attr_weights[relevant_kinases[j]] * nes_vec[j]
                            for j, idf in edges])
            observed[i] = np.median(ews)

    # Identify active pathways
    has_edges = np.array([len(e) > 0 for e in pw_edges])
    active_idx = np.where(has_edges)[0]
    print(f"  {len(active_idx)} pathways with edges (of {n_pw} total)")

    # Pre-extract per-pathway data for speed
    pw_idf_coeffs = [np.array([idf for _, idf in pw_edges[i]])
                     for i in active_idx]
    pw_degrees = np.array([len(pw_edges[i]) for i in active_idx])

    obs_active = observed[active_idx]
    rng = np.random.default_rng(42)

    # Group pathways by degree for vectorized permutation.
    # Precompute arrays once — these are invariant across all iterations.
    attr_weight_arr = np.fromiter(attr_weights.values(), dtype=float)

    degree_groups = defaultdict(list)
    for k in range(len(active_idx)):
        degree_groups[pw_degrees[k]].append((k, pw_idf_coeffs[k]))

    # Precompute per-group arrays (avoids rebuilding each batch)
    degree_group_data = {}
    for d, members in degree_groups.items():
        local_indices = np.array([m[0] for m in members], dtype=np.intp)
        idf_arrays = np.array([m[1] for m in members])   # (n_members, d)
        obs_slice = obs_active[local_indices]             # (n_members,)
        degree_group_data[d] = (local_indices, idf_arrays, obs_slice)

    batch_size = min(500, n_permutations)

    def _run_null(label, make_aw):
        """Run one null model.  ``make_aw`` returns the attribution-weight
        tensor of shape ``(b, n_members, d)`` for each degree group."""
        print(f"  {label}...")
        ge = np.zeros(len(active_idx))
        n_done = 0
        while n_done < n_permutations:
            b = min(batch_size, n_permutations - n_done)
            for d, (idx, idf_arr, obs_sl) in degree_group_data.items():
                n_members = len(idx)
                drawn = rng.integers(0, n_bg, size=(b, n_members, d))
                nes_drawn = bg_nes_vec[drawn]
                aw = make_aw(b, n_members, d)
                ews = idf_arr[np.newaxis, :, :] * nes_drawn * aw
                perm_medians = np.median(ews, axis=2)
                ge[idx] += (perm_medians >= obs_sl[np.newaxis, :]).sum(axis=0)
            n_done += b
            if n_done % 2000 == 0 or n_done == n_permutations:
                print(f"    {n_done}/{n_permutations}")
        return ge

    null1_ge = _run_null(
        "Null 1 (enrichment null)",
        lambda b, nm, d: median_attr_weight,  # scalar broadcasts
    )
    null2_ge = _run_null(
        "Null 2 (wiring null)",
        lambda b, nm, d: attr_weight_arr[
            rng.integers(0, len(attr_weight_arr), size=(b, nm, d))],
    )

    # Assemble p-values (non-active pathways get p=1)
    pval_null1 = np.ones(n_pw)
    pval_null2 = np.ones(n_pw)
    pval_null1[active_idx] = (null1_ge + 1) / (n_permutations + 1)
    pval_null2[active_idx] = (null2_ge + 1) / (n_permutations + 1)

    _, fdr_null1, _, _ = multipletests(pval_null1, method="fdr_bh")
    _, fdr_null2, _, _ = multipletests(pval_null2, method="fdr_bh")

    fdr_gate = icfg.PHOSPHO_FDR_GATE
    results = pd.DataFrame({
        "Path": pathways["Path"].values,
        "pval_null1": pval_null1,
        "pval_null2": pval_null2,
        "fdr_null1": fdr_null1,
        "fdr_null2": fdr_null2,
        "significant_both": (fdr_null1 < fdr_gate) & (fdr_null2 < fdr_gate),
    })

    n_sig1 = (fdr_null1 < fdr_gate).sum()
    n_sig2 = (fdr_null2 < fdr_gate).sum()
    n_both = results["significant_both"].sum()
    print(f"  Null 1 significant (FDR<{fdr_gate}): {n_sig1}")
    print(f"  Null 2 significant (FDR<{fdr_gate}): {n_sig2}")
    print(f"  Significant under both: {n_both}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute substrate-based kinase support scores")
    parser.add_argument("--permutations", action="store_true",
                        help="Run permutation null models")
    args = parser.parse_args()

    ensure_intermediates_dir()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    results_full = pd.read_csv(
        os.path.join(icfg.INTERMEDIATES_DIR, "results_full.csv"))
    pathways = results_full[
        ["Path", "EM", "Target", "Receptor", "Ligand", "TPDS", "PDS"]
    ].copy()
    print(f"  {len(pathways)} pathways")

    kldata = pd.read_csv(os.path.join(icfg.INTERMEDIATES_DIR, "kldata.csv"))
    print(f"  {len(kldata)} kinase-substrate edges in kldata")

    # ------------------------------------------------------------------
    # 2. Build naming bridge (mouse gene -> kinase abbreviations)
    # ------------------------------------------------------------------
    mouse_to_abbrevs = load_mouse_gene_to_kinase_mapping()
    print(f"  {sum(len(v) for v in mouse_to_abbrevs.values())} "
          f"mouse gene -> abbreviation mappings")

    # ------------------------------------------------------------------
    # 3. Load MEA kinases and attribution weights
    # ------------------------------------------------------------------
    sig_kinases, all_mea_nes = _load_mea_kinases(
        icfg.CONTRAST, icfg.PHOSPHO_FDR_GATE)
    print(f"  {len(sig_kinases)} significant kinases "
          f"(FDR < {icfg.PHOSPHO_FDR_GATE})")
    print(f"  {len(all_mea_nes)} total MEA kinases (background pool)")

    attr_weights = _load_attribution_weights(
        icfg.CONTRAST, icfg.SENDER, icfg.RECEIVER,
        icfg.SENDER_ATTRIBUTION_DISCOUNT)
    print(f"  {len(attr_weights)} kinases with attribution weights")

    both = set(sig_kinases) & set(attr_weights)
    print(f"  {len(both)} kinases significant AND attributed")

    # ------------------------------------------------------------------
    # 4. Pre-compute shared structures
    # ------------------------------------------------------------------
    sub_to_kins = build_substrate_kinase_map(kldata)
    idf_map = _compute_idf_map(sub_to_kins, mouse_to_abbrevs,
                               sig_kinases, attr_weights)

    # ------------------------------------------------------------------
    # 5. Compute scores
    # ------------------------------------------------------------------
    print("\nComputing kinase support scores...")
    scores_df = compute_scores(
        pathways, sub_to_kins, idf_map, sig_kinases, attr_weights,
        mouse_to_abbrevs)

    n_nonzero = (scores_df["kinase_support_score"] > 0).sum()
    n_zero = (scores_df["kinase_support_score"] == 0).sum()
    print(f"  Nonzero scores: {n_nonzero}")
    print(f"  Zero scores: {n_zero}")

    # ------------------------------------------------------------------
    # 5. Adjusted rankings
    # ------------------------------------------------------------------
    print("\nComputing adjusted rankings...")
    adj_df = compute_adjusted_rankings(scores_df, icfg.LAMBDA_VALUES)

    # Show rank divergence from TPDS at each lambda
    for lam in icfg.LAMBDA_VALUES:
        rank_col = f"adjusted_rank_lam{lam}"
        tau, _ = stats.kendalltau(adj_df["tpds_rank"], adj_df[rank_col])
        print(f"  lambda={lam}: Kendall tau vs TPDS = {tau:.4f}")

    # ------------------------------------------------------------------
    # 6. Sensitivity analyses
    # ------------------------------------------------------------------
    print("\nRunning sensitivity analyses...")
    summary = run_sensitivity_analyses(
        scores_df, results_full, icfg.LAMBDA_VALUES)

    # ------------------------------------------------------------------
    # 7. Write outputs
    # ------------------------------------------------------------------
    scores_path = os.path.join(
        icfg.INTERMEDIATES_DIR, "kinase_support_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"\nWrote {scores_path}")

    adj_path = os.path.join(icfg.INTERMEDIATES_DIR, "adjusted_rankings.csv")
    adj_df.to_csv(adj_path, index=False)
    print(f"Wrote {adj_path}")

    summary_path = os.path.join(
        icfg.INTERMEDIATES_DIR, "reranking_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # ------------------------------------------------------------------
    # 8. Optional: permutation tests
    # ------------------------------------------------------------------
    if args.permutations:
        perm_df = run_permutation_tests(
            pathways, sub_to_kins, idf_map, sig_kinases, attr_weights,
            mouse_to_abbrevs, all_mea_nes, icfg.N_PERMUTATIONS)
        perm_path = os.path.join(
            icfg.INTERMEDIATES_DIR, "permutation_pvalues.csv")
        perm_df.to_csv(perm_path, index=False)
        print(f"Wrote {perm_path}")

    print("\nAdapter 5.5 complete.")


if __name__ == "__main__":
    main()
