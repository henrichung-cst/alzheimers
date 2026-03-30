import ast
import os

import kinase_library as kl
import numpy as np
import pandas as pd
import requests
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def calculate_log2_fold_change(df, fg_col, bg_col):
    """Calculates log2 fold change between foreground and background columns."""
    return np.log2((df[fg_col] + 1) / (df[bg_col] + 1))

def run_kinase_enrichment(
    work_df, lfc_threshold, kin_type, kl_method, kl_thresh,
    title, output_dir_enrichment, output_dir_volcano,
    lff_thresh, pval_thresh,
):
    """Runs kinase enrichment analysis and saves volcano plot and results."""
    dpd = kl.DiffPhosData(work_df, seq_col='motif', lfc_col='log2_fold_change', lfc_thresh=lfc_threshold)
    enrich_data = dpd.kinase_enrichment(
        kin_type=kin_type,
        kl_method=kl_method,
        kl_thresh=kl_thresh
    )

    # Save volcano plot
    volcano_path = os.path.join(output_dir_volcano, f"{title}_volcano.png")
    enrich_data.plot_volcano(title=title, save_fig=volcano_path, sig_lff=lff_thresh, sig_pval=pval_thresh)

    # Save enrichment results
    enrichment_results_path = os.path.join(output_dir_enrichment, f"{title}_enrichment_results.csv")
    result_cols = [
        'most_sig_direction', 'most_sig_log2_freq_factor',
        'most_sig_fisher_pval', 'most_sig_fisher_adj_pval',
    ]
    results_df = enrich_data.combined_enrichment_results[result_cols]
    results_df.to_csv(enrichment_results_path)

    return enrich_data, results_df


def extract_substrate_lfc_stats(enrich_data, results_df, lff_thresh, pval_thresh):
    """Extract per-kinase substrate LFC distribution stats from enrichment results.

    For each significant kinase, retrieves the driving substrates via
    enrich_data.enriched_subs() and computes summary statistics on their
    |LFC| values. Adds columns to results_df in-place and returns it.
    """
    sig_mask = (
        (results_df["most_sig_log2_freq_factor"].abs() >= lff_thresh) &
        (results_df["most_sig_fisher_adj_pval"] <= pval_thresh)
    )
    sig_kinases = results_df.index[sig_mask].tolist()

    # Initialize substrate stat columns
    for col in ["n_substrates", "median_substrate_lfc", "q75_substrate_lfc", "max_substrate_lfc"]:
        results_df[col] = np.nan

    for kinase in sig_kinases:
        direction = results_df.loc[kinase, "most_sig_direction"]
        activity = "inhibited" if direction == "-" else "activated"
        try:
            subs = enrich_data.enriched_subs(
                [kinase], activity,
                data_columns=["log2_fold_change"],
                as_dataframe=True,
            )
            if subs is not None and not subs.empty:
                abs_lfc = subs["log2_fold_change"].abs()
                results_df.loc[kinase, "n_substrates"] = len(subs)
                results_df.loc[kinase, "median_substrate_lfc"] = abs_lfc.median()
                results_df.loc[kinase, "q75_substrate_lfc"] = abs_lfc.quantile(0.75)
                results_df.loc[kinase, "max_substrate_lfc"] = abs_lfc.max()
        except Exception:
            pass

    return results_df


def assign_evidence_tier(median_lfc, tier_boundaries):
    """Assign a qualitative evidence tier based on median substrate |LFC|.

    Parameters
    ----------
    median_lfc : float
        Median absolute LFC of driving substrates.
    tier_boundaries : tuple of (moderate, strong)
        LFC thresholds for tier boundaries.

    Returns
    -------
    str : 'Strong', 'Moderate', or 'Subtle'
    """
    if pd.isna(median_lfc):
        return np.nan
    moderate, strong = tier_boundaries
    if median_lfc >= strong:
        return "Strong"
    elif median_lfc >= moderate:
        return "Moderate"
    else:
        return "Subtle"

def get_mapping_cache(cache_file):
    """Loads the mapping cache from CSV or creates an empty one."""
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0).to_dict()['gene_symbol']
    return {}

def save_mapping_cache(cache_dict, cache_file):
    """Saves the mapping cache to CSV."""
    df = pd.DataFrame.from_dict(cache_dict, orient='index', columns=['gene_symbol'])
    df.index.name = 'kinase_abbreviation'
    df.to_csv(cache_file)

def resolve_kinase_symbol(kinase_name, cache):
    """Uses MyGene.info REST API to find the official gene symbol, with caching."""
    if kinase_name in cache:
        return cache[kinase_name]

    # Search specifically in symbol or alias fields to avoid greedy matches
    query = f"(symbol:{kinase_name} OR alias:{kinase_name})"
    url = f"https://mygene.info/v3/query?q={query}&species=mouse,human&fields=symbol,alias&size=5"

    try:
        response = requests.get(url, verify=False).json()
        if response.get('hits'):
            hits = response['hits']
            # 1. Exact Symbol match
            for hit in hits:
                if hit.get('symbol', '').upper() == kinase_name.upper():
                    symbol = hit['symbol']
                    cache[kinase_name] = symbol
                    return symbol
            # 2. Exact Alias match
            for hit in hits:
                aliases = hit.get('alias', [])
                if isinstance(aliases, str):
                    aliases = [aliases]
                if any(a.upper() == kinase_name.upper() for a in aliases):
                    symbol = hit['symbol']
                    cache[kinase_name] = symbol
                    return symbol
            # 3. Fallback to first hit
            symbol = hits[0]['symbol']
            cache[kinase_name] = symbol
            return symbol
    except Exception:
        pass

    # Default to itself if all else fails
    cache[kinase_name] = kinase_name
    return kinase_name

def map_kinases_to_genes(summary_path, cache_file):
    """Main function to map kinase abbreviations to official gene symbols."""
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}. Run the enrichment analysis first.")
        return

    # Extract all unique kinases from the enrichment summary
    df = pd.read_csv(summary_path)
    all_kinases = set()
    for k_list_str in df['top10_most_regulated_kinases']:
        all_kinases.update(ast.literal_eval(k_list_str))

    cache = get_mapping_cache(cache_file)
    new_mappings = 0

    print(f"Checking mappings for {len(all_kinases)} unique kinases...")
    for gene in sorted(list(all_kinases)):
        if gene not in cache:
            resolve_kinase_symbol(gene, cache)
            new_mappings += 1

    save_mapping_cache(cache, cache_file)
    print(f"Mapping complete. Total mappings: {len(cache)} ({new_mappings} new).")
    print(f"Result saved to: {cache_file}")

# --- Allen Institute Brain Atlas Expression Logic ---

ALLEN_API_BASE = "http://api.brain-map.org/api/v2/data"
PRODUCT_IDS = {
    "mouse": "1",
    "human": "9,10",
}
EXPRESSION_URL = (
    f"{ALLEN_API_BASE}/SectionDataSet/query.json"
    "?criteria=products[id$in{product_ids}],"
    "genes[acronym$eq'{gene_symbol}'],"
    "[expression$eqtrue],[failed$eqfalse]"
)

def get_expression_cache(cache_file):
    """Loads the Allen expression cache from CSV or creates an empty one."""
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0).to_dict('index')
    return {}

def save_expression_cache(cache_dict, cache_file):
    """Saves the Allen expression cache to CSV."""
    df = pd.DataFrame.from_dict(cache_dict, orient='index')
    df.index.name = 'gene_symbol'
    df.to_csv(cache_file)

def check_gene_expression(gene_symbol, organism="mouse", cache=None):
    """Check if a gene is expressed in the brain via the Allen Brain Atlas, with caching."""
    if cache is not None and gene_symbol in cache:
        return cache[gene_symbol]

    # Adjust casing based on organism (Allen API is case-sensitive)
    if organism.lower() == "mouse":
        search_symbol = gene_symbol.capitalize()
    elif organism.lower() == "human":
        search_symbol = gene_symbol.upper()
    else:
        search_symbol = gene_symbol

    url = EXPRESSION_URL.format(
        product_ids=PRODUCT_IDS[organism.lower()],
        gene_symbol=search_symbol,
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("msg", [])

        result = {
            "expressed": len(records) > 0,
            "num_experiments": len(records),
            "experiment_ids": str([r["id"] for r in records]), # Store as string for CSV
        }
        if cache is not None:
            cache[gene_symbol] = result
        return result
    except Exception as e:
        print(f"Error querying Allen API for {gene_symbol}: {e}")
        return {"expressed": False, "num_experiments": 0, "experiment_ids": "[]"}

def annotate_kinase_expression(kinase_list, kinase_to_gene_cache, expression_cache, organism="mouse"):
    """Annotate kinases with brain expression status (0/1) and experiment count."""
    kinase_expression_info = {}
    for kinase in kinase_list:
        gene_symbol = kinase_to_gene_cache.get(kinase, kinase)
        exp_result = check_gene_expression(gene_symbol, organism, expression_cache)
        kinase_expression_info[kinase] = {
            "brain_expressed": 1 if exp_result["expressed"] else 0,
            "num_experiments": exp_result["num_experiments"],
        }
    return kinase_list, kinase_expression_info
