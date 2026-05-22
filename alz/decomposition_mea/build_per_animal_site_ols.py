"""Levy-t5 per-animal site-level OLS publisher.

Consumes the per-track, per-cluster site OLS already computed by
`alz.decomposition_mea.enrich_celltype` (`site_level_ols_per_cluster{,_pY}.parquet`),
joins each row's `motif` from the matching `raw_phospho_normalized{,_pY}.csv`,
unions the `st` and `py` tracks with an explicit `track` column, renames
`cluster` → `cell_type` to match Levy-t5 vocabulary, and writes the canonical
single-file artifact the unified viewer reads:

    outputs/reports/decomposition/{spine}/per_animal/site_level_ols.parquet

Plus a `.audit.json` sidecar capturing spine, analysis_mode, and source
mtimes for staleness detection.

Usage:
    pixi run python alz/decomposition_mea/build_per_animal_site_ols.py --spine levy_t5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from alz.shared import config  # noqa: E402

REPO = Path(config.REPO_ROOT)
DEC_ROOT = REPO / "outputs/reports/decomposition"
BULK_DIR = REPO / "outputs/reports/kinase_attribution"

TRACKS = [
    {"name": "st", "suffix": "", "raw_csv": "raw_phospho_normalized.csv"},
    {"name": "py", "suffix": "_pY", "raw_csv": "raw_phospho_normalized_pY.csv"},
]


def _load_track(spine_dir: Path, track: dict) -> pd.DataFrame:
    site_ols_path = spine_dir / f"site_level_ols_per_cluster{track['suffix']}.parquet"
    if not site_ols_path.exists():
        raise FileNotFoundError(
            f"{site_ols_path} missing — run "
            f"`pixi run python -m alz.decomposition_mea.enrich_celltype "
            f"--spine {spine_dir.name} --track {track['name']}` first."
        )
    raw_path = BULK_DIR / track["raw_csv"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} missing — Stage 1 must emit raw phospho before Phase 4."
        )

    site_ols = pd.read_parquet(site_ols_path)
    motifs = (
        pd.read_csv(raw_path, usecols=["site_id", "motif"])
        .drop_duplicates(subset=["site_id"])
    )
    df = site_ols.merge(motifs, on="site_id", how="left")
    n_missing_motif = df["motif"].isna().sum()
    if n_missing_motif:
        raise RuntimeError(
            f"{track['name']}: {n_missing_motif} site rows have no motif "
            f"after join with {raw_path.name}. Site_id vocabulary drift."
        )
    df["track"] = track["name"]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spine", default=config.CLUSTER_SPINE_NAME)
    args = ap.parse_args()

    spine_dir = DEC_ROOT / args.spine
    out_dir = spine_dir / "per_animal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "site_level_ols.parquet"
    audit_path = out_dir / "site_level_ols.audit.json"

    parts = [_load_track(spine_dir, t) for t in TRACKS]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.rename(columns={"cluster": "cell_type"})

    # Keep only the columns the viewer + audits consume.
    cols = ["cell_type", "track", "contrast", "site_id", "gene_symbol", "motif",
            "lfc", "se", "t", "pval", "fdr", "n_obs"]
    combined = combined[cols]
    combined.to_parquet(out_path, index=False)

    n_cells = combined["cell_type"].nunique()
    n_sites = combined["site_id"].nunique()
    n_contrasts = combined["contrast"].nunique()
    print(f"wrote {out_path}  rows={len(combined):,}  cell_types={n_cells}  "
          f"sites={n_sites}  contrasts={n_contrasts}")

    audit = config.provenance_stamp(
        n_rows=int(len(combined)),
        n_cell_types=int(n_cells),
        n_sites=int(n_sites),
        n_contrasts=int(n_contrasts),
        tracks=[t["name"] for t in TRACKS],
        source_mtimes={
            f"site_level_ols_per_cluster{t['suffix']}.parquet": os.path.getmtime(
                spine_dir / f"site_level_ols_per_cluster{t['suffix']}.parquet")
            for t in TRACKS
        },
    )
    with open(audit_path, "w") as fh:
        json.dump(audit, fh, indent=2)
    print(f"wrote {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
