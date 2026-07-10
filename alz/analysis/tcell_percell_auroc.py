#!/usr/bin/env python3
"""Per-cell sub-state validation of ProjecTILs T-cell labels via AUROC.

Asks the sub-state question at the cell level: for a marker gene, how well does its
per-cell expression separate the cells of state s from the cells of its sibling
same-lineage states? That separation is exactly the AUROC (= Mann-Whitney U /
n_pos·n_neg = P(a random state-s cell out-expresses a random sibling cell)).
Comparing state *means* would smear this (30% of cells high == 100% low).

Two backgrounds, matched to the question:
  - state markers  -> sibling same-lineage cells (the sub-state question)
  - type  markers  -> the other lineage's cells (the cell-type question)

Inputs (per donor):
  - outputs/reports/tcell_labeling/auroc/{donor}_marker_cell_expr.csv
      cells x marker genes, log-normalized 'data' slot, written by
      tcell_export_marker_cells.R (the only step that touches the multi-GB .rds).
  - data/derived/tcells_incytr_inputs/{donor}/scrna/projectils_embeddings.csv
      per-cell functional.cluster label (already on disk; joined by barcode).

--write-markers PATH writes the marker-gene union (single source: SIGNATURES) for
the R extractor to consume, then exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcell_marker_sets import SIGNATURES, CORE_PANELS, _marker_class  # noqa: E402

DONORS = ("donor1", "donor2")
EMB_ROOT = Path("data/derived/tcells_incytr_inputs")
OUTDIR = Path("outputs/reports/tcell_labeling/auroc")

# Panel -> expected functional.cluster labels (the object's native vocabulary,
# e.g. CD8.TEX — distinct from the pseudobulk CD8Tex names). Lineage panels
# expand to the whole block from the "CD8."/"CD4." prefix at runtime.
_PANEL_LABELS = {
    "exhaustion": ["CD8.TEX", "CD4.CTL_Exh"],
    "progenitor_exhaustion": ["CD8.TPEX"],
    "cytotoxic": ["CD8.EM", "CD8.TEMRA", "CD8.TEX", "CD8.MAIT", "CD4.CTL_GNLY", "CD4.CTL_EOMES"],
    "th17": ["CD4.Th17"],
    "tfh": ["CD4.Tfh"],
    "treg": ["CD4.Treg"],
    "naive_memory": ["CD8.NaiveLike", "CD8.CM", "CD4.NaiveLike"],
}


def _lineage(label: str) -> str:
    return "CD8" if label.startswith("CD8") else "CD4"


def _expected_labels(panel: str, present: list[str]) -> list[str]:
    if panel == "cd8_lineage":
        return [s for s in present if _lineage(s) == "CD8"]
    if panel == "cd4_lineage":
        return [s for s in present if _lineage(s) == "CD4"]
    if panel in CORE_PANELS:
        return []
    want = set(_PANEL_LABELS.get(panel, []))
    return [s for s in present if s in want]


def _auroc(values: np.ndarray, pos: np.ndarray) -> float:
    """AUROC that `values` ranks positives (pos mask) above the rest."""
    n_pos = int(pos.sum())
    n_neg = int(pos.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(values)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _process(donor: str) -> None:
    expr = pd.read_csv(OUTDIR / f"{donor}_marker_cell_expr.csv").set_index("barcode")
    emb = pd.read_csv(EMB_ROOT / donor / "scrna" / "projectils_embeddings.csv",
                      usecols=["barcode", "functional.cluster"]).drop_duplicates("barcode")
    labels = emb.set_index("barcode")["functional.cluster"]
    expr = expr.join(labels, how="inner")
    state = expr.pop("functional.cluster").astype(str)
    lineage = state.map(_lineage)
    present_states = sorted(state.unique())
    markers = list(expr.columns)

    # --- per-marker AUROC (matched background) ---
    rows = []
    for panel, panel_genes in SIGNATURES.items():
        mclass = _marker_class(panel)
        expected = _expected_labels(panel, present_states)
        for gene in panel_genes:
            if gene not in markers:
                rows.append({"donor": donor, "gene": gene, "panel": panel, "marker_class": mclass,
                             "expected_labels": ";".join(expected), "present": False,
                             "best_state": "", "best_auroc": np.nan, "best_is_expected": False,
                             "expected_max_auroc": np.nan})
                continue
            v = expr[gene].to_numpy()
            if mclass == "type":
                # cell-type: this lineage's cells vs the other lineage's cells
                lin = "CD8" if panel == "cd8_lineage" else "CD4"
                auc = _auroc(v, (lineage == lin).to_numpy())
                best_state, best_auroc, exp_auc = lin, auc, auc
                is_expected = bool(auc >= 0.5)  # enriches in its own lineage
            else:
                # sub-state: each candidate state vs its SIBLING same-lineage cells
                per_state = {}
                for s in present_states:
                    same = (lineage == _lineage(s)).to_numpy()
                    pos = (state == s).to_numpy()
                    if pos.sum() == 0 or (same & ~pos).sum() == 0:
                        continue
                    per_state[s] = _auroc(v[same], pos[same])
                if not per_state:
                    continue
                best_state = max(per_state, key=per_state.get)
                best_auroc = per_state[best_state]
                exp_auc = max((per_state[s] for s in expected if s in per_state), default=np.nan)
                is_expected = best_state in expected
            rows.append({"donor": donor, "gene": gene, "panel": panel, "marker_class": mclass,
                         "expected_labels": ";".join(expected), "present": True,
                         "best_state": best_state, "best_auroc": round(float(best_auroc), 3),
                         "best_is_expected": is_expected,
                         "expected_max_auroc": round(float(exp_auc), 3) if not np.isnan(exp_auc) else np.nan})
    pd.DataFrame(rows).to_csv(OUTDIR / f"{donor}_percell_marker_auroc.csv", index=False)

    # --- per-panel signature-score AUROC (set level) ---
    # Per cell, signature score = mean of the panel's z-scored (across cells) marker
    # expression. AUROC of that score separates expected-label cells from the matched
    # background. This is the clean per-cell sub-state readout.
    z = (expr[markers] - expr[markers].mean()) / expr[markers].std(ddof=0).replace(0, np.nan)
    set_rows = []
    for panel, panel_genes in SIGNATURES.items():
        mclass = _marker_class(panel)
        genes = [g for g in panel_genes if g in markers]
        if not genes:
            continue
        score = z[genes].mean(axis=1).to_numpy()
        expected = set(_expected_labels(panel, present_states))
        for s in present_states:
            if mclass == "type":
                pos = (state == s).to_numpy()          # vs everything (lineage block signal)
                bg = np.ones(len(state), dtype=bool)
            else:
                same = (lineage == _lineage(s)).to_numpy()  # vs sibling same-lineage
                bg = same
                pos = (state == s).to_numpy() & same
            sub = score[bg]
            auc = _auroc(sub, pos[bg])
            set_rows.append({"donor": donor, "panel": panel, "marker_class": mclass, "state": s,
                             "n_markers": len(genes), "signature_auroc": round(float(auc), 3) if not np.isnan(auc) else np.nan,
                             "n_cells_state": int((state == s).sum()), "is_expected_state": s in expected})
    pd.DataFrame(set_rows).to_csv(OUTDIR / f"{donor}_percell_panel_auroc.csv", index=False)
    print(f"[{donor}] wrote per-cell AUROC ({len(expr)} cells, {len(markers)} markers, {len(present_states)} states)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-markers", type=Path, help="write marker-gene union for the R extractor, then exit")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.write_markers:
        genes = sorted({g for panel in SIGNATURES.values() for g in panel})
        args.write_markers.parent.mkdir(parents=True, exist_ok=True)
        args.write_markers.write_text("\n".join(genes) + "\n")
        print(f"wrote {len(genes)} marker genes to {args.write_markers}")
        return 0

    for donor in DONORS:
        expr_csv = OUTDIR / f"{donor}_marker_cell_expr.csv"
        if not expr_csv.exists():
            print(f"[{donor}] missing {expr_csv} — run tcell_export_marker_cells.R first", file=sys.stderr)
            continue
        _process(donor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
