"""Regenerable kinase-abbreviation → gene-symbol cache.

The kinase-library uses its own kinase abbreviation vocabulary (e.g. `AKT3`,
`DNAPK`, `CK1A2`) that does not always match the HGNC gene symbol. This
producer enumerates every kinase in `kinase_library.get_kinome_info()`
(the authoritative source — `kldata*.csv` exports are derived from it),
falls back to MyGene.info for anything the kinome_info table doesn't resolve
cleanly, and writes the result to `data/derived/caches/kinase_to_gene_mapping.csv`.

Manual curation lives in a sidecar `kinase_to_gene_overrides.csv` that takes
precedence. When the upstream returns a wrong ortholog, add the correct
mapping to the sidecar rather than editing the regenerable cache.

Outputs:
  data/derived/caches/kinase_to_gene_mapping.csv  (regenerable)

Usage:
  pixi run python alz/shared/map_kinases_to_genes.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import requests
import urllib3

# Suppress SSL warnings for the environment
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from alz.shared import config

MAPPING_CACHE_FILE = config.MAPPING_CACHE_FILE
OVERRIDES_FILE = os.path.join(
    config.DERIVED_CACHES_DIR, "kinase_to_gene_overrides.csv"
)


def _load_overrides() -> dict[str, str]:
    """Manual kinase → gene symbol overrides (precedence over MyGene)."""
    if not os.path.exists(OVERRIDES_FILE):
        return {}
    df = pd.read_csv(OVERRIDES_FILE)
    return dict(zip(df["kinase_abbreviation"].astype(str),
                    df["gene_symbol"].astype(str)))


def _resolve_via_mygene(kinase_name: str) -> str | None:
    """MyGene.info symbol/alias lookup. Returns None on failure."""
    query = f"(symbol:{kinase_name} OR alias:{kinase_name})"
    url = (f"https://mygene.info/v3/query?q={query}"
           f"&species=mouse,human&fields=symbol,alias&size=5")
    try:
        hits = requests.get(url, verify=False, timeout=10).json().get("hits") or []
    except Exception as e:
        print(f"  WARN: MyGene.info lookup failed for {kinase_name}: {e}")
        return None
    if not hits:
        return None
    # 1. Exact symbol match
    for h in hits:
        if str(h.get("symbol", "")).upper() == kinase_name.upper():
            return h["symbol"]
    # 2. Exact alias match
    for h in hits:
        aliases = h.get("alias") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if any(str(a).upper() == kinase_name.upper() for a in aliases):
            return h["symbol"]
    # 3. Fallback to first hit
    return hits[0].get("symbol")


def main() -> int:
    import kinase_library as kl
    overrides = _load_overrides()
    info = kl.get_kinome_info()[["KINASE", "GENE_NAME"]].drop_duplicates()
    info = info[info["KINASE"].notna()]
    kinases = sorted({str(k) for k in info["KINASE"].unique()})
    upstream_lookup = dict(zip(info["KINASE"].astype(str),
                               info["GENE_NAME"].astype(str)))
    print(f"Resolving {len(kinases)} unique kinase abbreviations from "
          f"kinase_library.get_kinome_info()")
    print(f"  Overrides: {len(overrides)} manual entries from {OVERRIDES_FILE}")

    mapping: dict[str, str] = {}
    n_overridden = n_upstream = n_mygene = n_self = 0
    for k in kinases:
        if k in overrides:
            mapping[k] = overrides[k]
            n_overridden += 1
            continue
        upstream_gene = upstream_lookup.get(k)
        if upstream_gene and upstream_gene.strip():
            mapping[k] = upstream_gene
            n_upstream += 1
            continue
        resolved = _resolve_via_mygene(k)
        if resolved:
            mapping[k] = resolved
            n_mygene += 1
        else:
            mapping[k] = k
            n_self += 1
            print(f"  WARN: {k} unresolved, falling back to self")

    df = pd.DataFrame(
        sorted(mapping.items()), columns=["kinase_abbreviation", "gene_symbol"]
    )
    os.makedirs(os.path.dirname(MAPPING_CACHE_FILE), exist_ok=True)
    df.to_csv(MAPPING_CACHE_FILE, index=False)
    print(f"Wrote {MAPPING_CACHE_FILE}")
    print(f"  total: {len(df)}  overridden: {n_overridden}  "
          f"upstream: {n_upstream}  mygene: {n_mygene}  "
          f"fell-back-to-self: {n_self}")
    if not os.path.exists(OVERRIDES_FILE):
        # Seed an empty overrides file so the existence is documented.
        pd.DataFrame(columns=["kinase_abbreviation", "gene_symbol"]).to_csv(
            OVERRIDES_FILE, index=False
        )
        print(f"  seeded empty {OVERRIDES_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
