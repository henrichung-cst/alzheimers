
import ast
import os

import pandas as pd
import requests
import urllib3

# Suppress SSL warnings for the environment
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import config

MAPPING_CACHE_FILE = config.MAPPING_CACHE_FILE

def get_mapping_cache():
    """Loads the mapping cache from CSV or creates an empty one."""
    if os.path.exists(MAPPING_CACHE_FILE):
        return pd.read_csv(MAPPING_CACHE_FILE, index_col=0).to_dict()['gene_symbol']
    return {}

def save_mapping_cache(cache_dict):
    """Saves the mapping cache to CSV."""
    df = pd.DataFrame.from_dict(cache_dict, orient='index', columns=['gene_symbol'])
    df.index.name = 'kinase_abbreviation'
    df.to_csv(MAPPING_CACHE_FILE)

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

def main():
    summary_path = "outputs/deconv/enrichment_summary_sig_kins.csv"
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}. Run the enrichment analysis first.")
        return

    # Extract all unique kinases from the enrichment summary
    df = pd.read_csv(summary_path)
    all_kinases = set()
    for k_list_str in df['top10_most_regulated_kinases']:
        all_kinases.update(ast.literal_eval(k_list_str))

    cache = get_mapping_cache()
    new_mappings = 0

    print(f"Checking mappings for {len(all_kinases)} unique kinases...")
    for gene in sorted(list(all_kinases)):
        if gene not in cache:
            resolve_kinase_symbol(gene, cache)
            new_mappings += 1

    save_mapping_cache(cache)
    print(f"Mapping complete. Total mappings: {len(cache)} ({new_mappings} new).")
    print(f"Result saved to: {MAPPING_CACHE_FILE}")

if __name__ == "__main__":
    main()
