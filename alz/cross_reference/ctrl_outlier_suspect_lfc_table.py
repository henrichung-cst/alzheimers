"""Per-site LFC table for the three suspicious human controls (CTRL-07/08/10).

For each phosphosite we report how the *suspect* controls depart from the four
clean controls, alongside how the *AD* cases depart from the same clean baseline,
so the two can be compared row by row. A final column annotates the gene's
Human Protein Atlas secretome location.

Substrate: log2 stoichiometry matrices, so LFC = difference of NaN-aware group
means (sign convention: + = up vs clean controls, matching the pipeline-wide
"+ = up in disease" rule).

  lfc_suspect = mean(CTRL-07,08,10) - mean(CTRL-01,02,03,04)
  lfc_ad      = mean(AD x10)        - mean(CTRL-01,02,03,04)
  diff        = lfc_suspect - lfc_ad
  abs_diff    = |diff|

Ranked by |lfc_suspect| descending. Both st and pY tracks.

Read-only on derived matrices; no pipeline edits.
HPA secretome cached at ctrl_audit/hpa_secretome.tsv (protein_class:Predicted
secreted proteins, columns g,secl). See investigation_report/README.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HUMAN = Path("outputs/reports/kinase_attribution_human")
OUT = HUMAN / "ctrl_audit"

AD = ["AD-01", "AD-02", "AD-03", "AD-04", "AD-06", "AD-07", "AD-08", "AD-09", "AD-13", "AD-15"]
CTRL_CLEAN = ["CTRL-01", "CTRL-02", "CTRL-03", "CTRL-04"]
CTRL_SUSP = ["CTRL-07", "CTRL-08", "CTRL-10"]
META = ["site_id", "protein_id", "gene_symbol", "site_position", "motif"]


def load_secretome() -> dict[str, str]:
    sec = pd.read_csv(OUT / "hpa_secretome.tsv", sep="\t")
    loc = sec.set_index("Gene")["Secretome location"].dropna()
    return loc[loc.str.strip() != ""].to_dict()


def build(track: str, secretome: dict[str, str]) -> pd.DataFrame:
    fn = "stoichiometry_matrix.csv" if track == "st" else "stoichiometry_matrix_pY.csv"
    df = pd.read_csv(HUMAN / fn)
    susp = df[CTRL_SUSP].astype(float)
    clean = df[CTRL_CLEAN].astype(float)
    ad = df[AD].astype(float)

    out = df[META].copy()
    out["lfc_suspect"] = susp.mean(axis=1) - clean.mean(axis=1)
    out["lfc_ad"] = ad.mean(axis=1) - clean.mean(axis=1)
    out["diff_suspect_minus_ad"] = out["lfc_suspect"] - out["lfc_ad"]
    out["abs_diff"] = out["diff_suspect_minus_ad"].abs()
    # observation counts so single-measurement LFCs are transparent (not hidden)
    out["n_suspect"] = susp.notna().sum(axis=1).values
    out["n_clean"] = clean.notna().sum(axis=1).values
    out["n_ad"] = ad.notna().sum(axis=1).values
    out["secretome_location"] = out["gene_symbol"].map(secretome).fillna("")

    # rank by signed suspect LFC desc; NaN LFCs (group fully missing) sort to the bottom
    out = out.sort_values("lfc_suspect", ascending=False,
                          na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def collapse_to_gene(sites: pd.DataFrame, secretome: dict[str, str]) -> pd.DataFrame:
    """Average the per-site LFCs within each gene (phosphosite layer removed).

    diff/abs_diff are re-derived from the gene-mean LFCs so they keep the
    per-site definition (diff = suspect - ad) rather than being a mean of |.|.
    """
    g = sites.groupby("gene_symbol", sort=False)
    out = pd.DataFrame({
        "lfc_suspect": g["lfc_suspect"].mean(),
        "lfc_ad": g["lfc_ad"].mean(),
        "n_sites": g.size(),
    }).reset_index()
    out["diff_suspect_minus_ad"] = out["lfc_suspect"] - out["lfc_ad"]
    out["abs_diff"] = out["diff_suspect_minus_ad"].abs()
    out["secretome_location"] = out["gene_symbol"].map(secretome).fillna("")
    out = out[["gene_symbol", "lfc_suspect", "lfc_ad", "diff_suspect_minus_ad",
               "abs_diff", "n_sites", "secretome_location"]]
    out = out.sort_values("lfc_suspect", ascending=False,
                          na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def main():
    secretome = load_secretome()
    for track in ("st", "pY"):
        tbl = build(track, secretome)
        path = OUT / f"suspect_vs_ad_lfc_{track}.csv"
        tbl.to_csv(path, index=False)
        n_sec = (tbl["secretome_location"] != "").sum()
        n_nan = tbl["lfc_suspect"].isna().sum()
        print(f"[{track}] {len(tbl)} sites -> {path.name}  "
              f"({n_sec} secreted-gene sites, {n_nan} with NaN suspect LFC)")
        print(tbl.head(10)[["rank", "gene_symbol", "site_position", "lfc_suspect",
                            "lfc_ad", "diff_suspect_minus_ad", "secretome_location"]]
              .to_string(index=False))

        gene = collapse_to_gene(tbl, secretome)
        gpath = OUT / f"suspect_vs_ad_lfc_{track}_bygene.csv"
        gene.to_csv(gpath, index=False)
        print(f"[{track}] {len(gene)} genes -> {gpath.name}")
        print(gene.head(10)[["rank", "gene_symbol", "lfc_suspect", "lfc_ad",
                            "diff_suspect_minus_ad", "n_sites", "secretome_location"]]
              .to_string(index=False))
        print()


if __name__ == "__main__":
    main()
