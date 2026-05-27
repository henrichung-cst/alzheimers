"""Build the pair-mode wide-CSV inputs the Incytr R driver consumes, via the
*provenance* deconvolution (Yuyu `protein-ms-by-cell-type.py`):

    P_c = (N_total / N_c) x bulk x (specific_c / Sum_46 specific)

with min/10000 zero-imputation. This is the method that reproduces sce4
(Acvr1/Cholinergic ma_2mo = 50.74 / 28.22). See
`docs/plans/sce4_decomposition_reconciliation_2026-05-24.md` for why the prior
levy_t5 forward-projection share (log2(CPM+1) x bulk) could NOT reproduce it:
aggexp is `AggregateExpression(slot="data")` (a model-based / SCT normalization
baked into an upstream Seurat object not on this box), so the transcript share
cannot be regenerated from the h5ad counts. We therefore consume the frozen
`aggexp.csv` for the share (the one unrecoverable upstream step) but regenerate
the cell-count size factors from the Song h5ad (byte-exact clustering).

Inputs:
  - frozen aggexp:      transcript share, 46 cluster x 24 group, gene columns
  - Song h5ad:          per-(cluster, group) cell counts -> N_total / N_c
  - frozen group medians: per-group MS bulk per track
      pr  -> pr_median.csv   (key `Gene Symbol`)
      ps  -> imac_median.csv (keys `site_id`, `gene_symbol`)
      py  -> py_median.csv   (keys `site_id`, `gene_symbol`)

Outputs (wide, males-only, group-level):
  data/derived/incytr_inputs/pr_yuyu_deconvoluted.csv
  data/derived/incytr_inputs/ps_yuyu_deconvoluted.csv
  data/derived/incytr_inputs/py_yuyu_deconvoluted.csv

Wide schema: row keys + columns `<condition>_<cluster>` where
condition = `ma_<age>_<geno>` (12 male conditions x 31 spine clusters = 372
value columns). The share denominator is summed over ALL 46 clusters; only the
31-cluster levy_t5 spine is emitted.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from alz.shared import config  # noqa: E402
from alz.integration.config_integration import load_cluster_spine  # noqa: E402

PROV = os.path.join(REPO_ROOT, "data", "incytr_frozen", "v2_46clusters", "provenance")
AGGEXP = os.path.join(PROV, "aggexp.csv")
CLUSTERSIZE = os.path.join(PROV, "yuyu_clustersize.csv")
SAMPLEKEY = os.path.join(PROV, "yuyu_samplekey.csv")
PR_MEDIAN = os.path.join(PROV, "pr_median.csv")
IMAC_MEDIAN = os.path.join(PROV, "imac_median.csv")
PY_MEDIAN = os.path.join(PROV, "py_median.csv")

OUT_DIR = os.path.join(REPO_ROOT, "data", "derived", "incytr_inputs")

# 46 coarse cluster names (provenance basis). Sorted longest-first so the
# numeric per-group suffix on aggexp's cluster column is stripped against the
# longest matching coarse name (provenance `remove_number`).
_COARSE = sorted(
    pd.read_csv(CLUSTERSIZE, index_col=0).index.astype(str), key=len, reverse=True
)


def _to_coarse(raw: str) -> str:
    s = str(raw).replace('"', "").strip()
    for c in _COARSE:
        if s == c or (s.startswith(c) and s[len(c):].isdigit()):
            return c
    raise ValueError(f"aggexp cluster {raw!r} matches no coarse cluster")


def _ms_to_group() -> dict[str, str]:
    """MS_ID column name -> SCRNA group (e.g. 'M_2mo_APP' -> 'ma_2mo_AppP')."""
    sk = pd.read_csv(SAMPLEKEY)
    return dict(zip(sk["MS_ID"].astype(str), sk["Group"].astype(str)))


def _cell_counts_from_h5ad() -> pd.DataFrame:
    """(coarse cluster) x (group) cell-count matrix, regenerated from the Song
    h5ad via the frozen 46-cluster barcode map. Reproduces yuyu_clustersize.csv
    byte-exactly (verified) but keeps the pipeline from-raw. obs-only backed
    read — the expression matrix is never loaded."""
    import anndata as ad

    bc = config.load_barcode_to_cluster_map()
    adata = ad.read_h5ad(config.SONG_H5AD_FILE, backed="r")
    obs = adata.obs
    cluster = obs.index.to_series().map(bc)
    group = obs["sample"].astype(str).map(lambda s: "_".join(s.split("_")[-3:]))
    df = pd.DataFrame({"cluster": cluster.values, "group": group.values})
    df = df[df["cluster"].notna()]
    counts = (
        df.value_counts(["cluster", "group"]).rename("n").reset_index()
        .pivot(index="cluster", columns="group", values="n")
        .reindex(_COARSE).fillna(0.0)
    )
    return counts


def _load_share() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Returns (share, genes, gene_pos).

    share: index (coarse cluster, group) -> DataFrame of per-gene transcript
    share `specific_c / Sum_46 specific`, with min/10000 zero-imputation, summed
    over all 46 clusters per (gene, group). genes: the aggexp gene list.
    """
    with open(AGGEXP) as fh:
        header = next(csv.reader(fh))
    genes = [h.replace('"', "").strip() for h in header[1:-1]]
    # Read all rows; first column (empty header) = cluster, last = Sample/group.
    agg = pd.read_csv(
        AGGEXP, header=0, names=["_cluster"] + genes + ["group"],
        dtype={"_cluster": str, "group": str},
    )
    agg["group"] = agg["group"].str.replace('"', "").str.strip()
    agg["cluster"] = agg["_cluster"].map(_to_coarse)
    agg = agg.drop(columns=["_cluster"]).set_index(["cluster", "group"])
    gene_mat = agg[genes].astype("float32")

    # Reindex to the full 46 x 24 grid; synthesized-missing rows start at 0 and
    # are imputed to the floor below (provenance synthesizes missing cluster x
    # group rows as min/10000).
    all_groups = sorted(agg.index.get_level_values("group").unique())
    full_idx = pd.MultiIndex.from_product([_COARSE, all_groups],
                                          names=["cluster", "group"])
    gene_mat = gene_mat.reindex(full_idx).fillna(0.0)

    nz = gene_mat.values[gene_mat.values > 0]
    floor = float(nz.min()) / 10000.0
    arr = gene_mat.values
    arr[arr == 0.0] = floor

    # share = specific / Sum_46 specific, per (gene, group)
    share = pd.DataFrame(arr, index=gene_mat.index, columns=genes)
    totals = share.groupby(level="group").transform("sum")
    share = share / totals
    gene_pos = {g: i for i, g in enumerate(genes)}
    return share, genes, gene_pos


def _deconvolve(
    share: pd.DataFrame,
    genes_set: set[str],
    counts: pd.DataFrame,
    bulk: pd.DataFrame,
    bulk_gene_col: str,
    key_cols: list[str],
    ms2grp: dict[str, str],
    spine: list[str],
) -> pd.DataFrame:
    """Run the provenance deconvolution for one MS track.

    bulk: rows = proteins/sites, columns include key_cols + MS_ID bulk columns.
    Emits wide DataFrame: key_cols + `<group>_<cluster>` for male groups x spine.
    """
    male_ms = {ms: g for ms, g in ms2grp.items() if g.startswith("ma_")}
    n_total = counts.sum(axis=0)  # per group

    # Pre-extract per-group share sub-frames (cluster x gene) for speed.
    share_by_group = {
        g: share.xs(g, level="group") for g in set(male_ms.values())
    }

    out_rows = []
    bulk = bulk[bulk[bulk_gene_col].astype(str).isin(genes_set)].reset_index(drop=True)
    for _, brow in bulk.iterrows():
        gene = str(brow[bulk_gene_col])
        rec = {k: brow[k] for k in key_cols}
        for ms_id, group in male_ms.items():
            if ms_id not in bulk.columns:
                continue
            bval = brow[ms_id]
            sh = share_by_group[group]
            if gene not in sh.columns:
                continue
            gshare = sh[gene]  # indexed by cluster
            for cl in spine:
                nc = counts.at[cl, group]
                # Empty cluster in this group contributes no protein (0 cells).
                if nc <= 0:
                    rec[f"{group}_{cl}"] = 0.0
                    continue
                sf = n_total[group] / nc
                rec[f"{group}_{cl}"] = sf * float(bval) * float(gshare[cl])
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def export_protein(share, genes_set, counts, ms2grp, spine) -> str:
    bulk = pd.read_csv(PR_MEDIAN)
    wide = _deconvolve(share, genes_set, counts, bulk, "Gene Symbol",
                       ["Gene Symbol"], ms2grp, spine)
    wide.insert(0, "gene_symbol", wide["Gene Symbol"])
    out = os.path.join(OUT_DIR, "pr_yuyu_deconvoluted.csv")
    print(f"[export-pair]   writing {out}  shape={wide.shape}")
    wide.to_csv(out, index=False)
    return out


def export_phospho(share, genes_set, counts, ms2grp, spine,
                   median_file: str, out_name: str, key_cols: list[str]) -> str:
    # Driver collapses ps/py rows by `gene_symbol` (site keys are ignored), so
    # py_median (which lacks `site_id`) keys on gene_symbol alone.
    bulk = pd.read_csv(median_file)
    wide = _deconvolve(share, genes_set, counts, bulk, "gene_symbol",
                       key_cols, ms2grp, spine)
    out = os.path.join(OUT_DIR, out_name)
    print(f"[export-pair]   writing {out}  shape={wide.shape}")
    wide.to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=["pr", "ps", "py", "all"], default="all")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    spine = load_cluster_spine("levy_t5")
    print(f"[export-pair] spine={len(spine)} clusters; provenance deconvolution")
    print("[export-pair] regenerating cell counts from Song h5ad ...")
    counts = _cell_counts_from_h5ad()
    print("[export-pair] loading frozen aggexp transcript share ...")
    share, genes, _ = _load_share()
    genes_set = set(genes)
    ms2grp = _ms_to_group()

    if args.track in ("pr", "all"):
        export_protein(share, genes_set, counts, ms2grp, spine)
    if args.track in ("ps", "all"):
        export_phospho(share, genes_set, counts, ms2grp, spine,
                       IMAC_MEDIAN, "ps_yuyu_deconvoluted.csv",
                       ["site_id", "gene_symbol"])
    if args.track in ("py", "all"):
        export_phospho(share, genes_set, counts, ms2grp, spine,
                       PY_MEDIAN, "py_yuyu_deconvoluted.csv",
                       ["gene_symbol"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
