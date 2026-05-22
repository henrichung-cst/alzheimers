"""SEA-AD transcriptomic agreement for the human (NBB / Mukesh) per-donor kinase MEA.

Goal: for each kinase scored by the per-donor MEA, give a single cohort-level
SEA-AD LFC so the viewer can ask "does the SEA-AD AD-vs-control transcript
signal agree, in direction, with the phospho MEA NES?".

We deliberately collapse across cell type. The human AD samples have no
per-cell-type resolution, so a cell-type-resolved SEA-AD comparison (as on
the mouse side) is not meaningful here. Instead we summarise each kinase's
gene-level LFC across SEA-AD's 139 MTG supertypes.

Inputs:
  data/external/sea_ad/effect_sizes.h5ad                          — full CPS effect sizes
  data/derived/caches/kinase_to_gene_mapping.csv    — kinase abbr → gene
  outputs/reports/kinase_attribution_human/perdonor/recurrence{,_pY}.csv
    — kinase list per residue track

Output:
  outputs/reports/kinase_attribution/human_seaad_agreement.csv
    Columns: kinase, gene_symbol, residue_type, sea_ad_lfc_median,
             sea_ad_lfc_mean, n_supertypes_finite, sea_ad_direction_agreement
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config
from alz.ingest.mukesh import HUMAN_KINASE_DIR

SEA_AD_H5AD = os.path.join(config.REPO_ROOT, "data", "external", "sea_ad", "effect_sizes.h5ad")
KINASE_GENE_CSV = os.path.join(
    config.REPO_ROOT, "data", "datasets", "song", "analysis_cache",
    "kinase_to_gene_mapping.csv",
)
PERDONOR_DIR = os.path.join(HUMAN_KINASE_DIR, "perdonor")
OUT_CSV = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "kinase_attribution",
    "human_seaad_agreement.csv",
)


def _load_kinase_list() -> pd.DataFrame:
    """Per-residue kinase list from the recurrence outputs."""
    rows = []
    for track_key, residue in (("st", "ST"), ("py", "Y")):
        suffix = config.PHOSPHO_TRACKS[track_key]["output_suffix"]
        path = os.path.join(PERDONOR_DIR, f"recurrence{suffix}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        rows.append(pd.DataFrame({
            "kinase": df["kinase"].astype(str),
            "residue_type": residue,
        }))
    if not rows:
        raise FileNotFoundError(
            f"No recurrence CSV found under {PERDONOR_DIR}; "
            "run alz/ingest/mukesh_perdonor.py first."
        )
    return pd.concat(rows, ignore_index=True)


def _kinase_gene_map() -> dict[str, str]:
    if not os.path.exists(KINASE_GENE_CSV):
        return {}
    df = pd.read_csv(KINASE_GENE_CSV)
    return {
        str(k): str(g) for k, g in zip(df["kinase_abbreviation"], df["gene_symbol"])
        if pd.notna(g)
    }


def _gene_lfc_summary(adata, gene_idx: int) -> dict:
    """Median / mean LFC across supertypes + direction-agreement audit."""
    row = adata.X[gene_idx, :]
    if hasattr(row, "toarray"):
        row = row.toarray().flatten()
    else:
        row = np.asarray(row).flatten()
    finite = np.isfinite(row)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {"median": np.nan, "mean": np.nan, "n_finite": 0, "agreement": np.nan}
    vals = row[finite]
    med = float(np.median(vals))
    mean = float(np.mean(vals))
    # Direction agreement: share of supertypes whose sign matches the mean's sign.
    sign_ref = np.sign(mean) if mean != 0 else 0
    if sign_ref == 0:
        agree = np.nan
    else:
        agree = float(np.mean(np.sign(vals) == sign_ref))
    return {"median": med, "mean": mean, "n_finite": n_finite, "agreement": agree}


def run() -> None:
    if not os.path.exists(SEA_AD_H5AD):
        raise FileNotFoundError(
            f"missing {SEA_AD_H5AD}; run `python alz/reference/atlas.py --sea-ad` first"
        )
    try:
        import anndata as ad_mod
    except ImportError as e:
        raise SystemExit("anndata is required: pixi run python ...") from e

    print(f"Loading SEA-AD effect sizes: {SEA_AD_H5AD}")
    adata = ad_mod.read_h5ad(SEA_AD_H5AD)
    gene_to_idx = {g: i for i, g in enumerate(adata.obs_names)}
    print(f"  genes={adata.shape[0]}  supertypes={adata.shape[1]}")

    kinase_df = _load_kinase_list()
    kinase_to_gene = _kinase_gene_map()
    print(f"  kinases (kin × residue rows): {len(kinase_df)}  "
          f"gene-map entries: {len(kinase_to_gene)}")

    out_rows = []
    missing_genes: list[str] = []
    for _, row in kinase_df.iterrows():
        kinase = row["kinase"]
        residue = row["residue_type"]
        gene = kinase_to_gene.get(kinase, kinase)
        gi = gene_to_idx.get(gene)
        if gi is None:
            missing_genes.append(gene)
            out_rows.append({
                "kinase": kinase, "gene_symbol": gene, "residue_type": residue,
                "sea_ad_lfc_median": None, "sea_ad_lfc_mean": None,
                "n_supertypes_finite": 0, "sea_ad_direction_agreement": None,
            })
            continue
        s = _gene_lfc_summary(adata, gi)
        out_rows.append({
            "kinase": kinase, "gene_symbol": gene, "residue_type": residue,
            "sea_ad_lfc_median": s["median"],
            "sea_ad_lfc_mean": s["mean"],
            "n_supertypes_finite": s["n_finite"],
            "sea_ad_direction_agreement": s["agreement"],
        })

    out = pd.DataFrame(out_rows)
    os.makedirs(HUMAN_KINASE_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    matched = out["n_supertypes_finite"].fillna(0).gt(0).sum()
    print(f"  wrote {OUT_CSV}  rows={len(out)}  with SEA-AD coverage={matched}")
    if missing_genes:
        uniq = sorted(set(missing_genes))
        print(f"  WARNING: {len(uniq)} genes not in SEA-AD obs_names "
              f"(first 5): {uniq[:5]}")


if __name__ == "__main__":
    run()
