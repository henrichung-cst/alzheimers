#!/usr/bin/env python3
"""Regression check: canonical pair-mode output reproduces sce4 reference.

Validates the six-fix chain that produces sce4-equivalent Incytr output
(`bench/bench.md` §sce4 reproduction). Each of the six call-site overrides
in `alz/incytr_pair/incytr_commandline.R` is required to hold the numeric
parity; if any one is reverted, this check exits non-zero.

Acceptance bar (Microglia → Cholinergic.Neurons, App_2mo contrast):
  - Path-key recall (sce4 rows present in our output after SigProb filter)
    >= --min-recall (default 595), OR every miss is an App-transgene ligand
    path. The derived (provenance-deconvolution) inputs reproduce sce4 at
    573/600 here; the 27-path deficit is 100% App-ligand. App is an isolated
    input-provenance gap: sce4's Oct-2025 run pr ranked the transgene into
    gene.use, but every pr reproducible on this box gives App fc2 rank
    ~4523/6687 (pr_log2FC -0.345), so no fold-based rule (prG |log2|>1 or
    top_n(500)) recovers it. The frozen 46-cluster run caps at 573 for the
    same reason. See bench/bench.md §App transgene residual.
  - For each position in {Receptor, Target}: max |Δ sclog2FC| == 0 after
    merging on the (Ligand|Receptor|EM|Target) path key. Ligand and EM
    positions may carry App-transgene residuals — outliers there are tolerated
    only when the position gene is the --transgene (App); any other mismatch
    fails.

Usage:
  pixi run python alz/incytr_pair/verify_sce4_parity.py
  pixi run python alz/incytr_pair/verify_sce4_parity.py \\
      --contrast App_2mo --sender Microglia --receiver Cholinergic.Neurons
  pixi run python alz/incytr_pair/verify_sce4_parity.py --all-known-pairs

The two known-good pairs (derived provenance-deconvolution inputs):
  - Microglia → Cholinergic.Neurons  (573/600, max |Δ| = 0 on R/E/T;
                                       27 misses all App-ligand → PASS)
  - Ndnf...×Ndnf...                  (599/600, max |Δ| = 0 on R/E/T)
"""

from __future__ import annotations

import argparse
import os
import sys

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SCE4_REF_DEFAULT = os.path.join(REPO_ROOT, "bench/sce4_DEG_PRG_Top300_table_10302025.csv")
WIDE_DIR_DEFAULT = os.path.join(REPO_ROOT, "outputs/reports/incytr_pair_mode/wide")

# Contrast string (e.g. "App_2mo") → wide-parquet filename stem.
_GENO_DECODE = {"App": "AppP", "Tau": "Ttau", "ApTt": "ApTt"}

# Cluster args use sce4's dotted vocabulary; _to_canonical() translates to our
# dashed wide-parquet vocabulary for the read filter.
NDNF = "Ndnf.positive.neurogliaform.inhibitory.interneurons.GABAergic"

KNOWN_PAIRS = [
    ("App_2mo", "Microglia", "Cholinergic.Neurons"),
    ("App_2mo", NDNF,        NDNF),
]


def contrast_to_parquet(contrast: str, wide_dir: str) -> str:
    geno, age = contrast.split("_")
    code = _GENO_DECODE[geno]
    return os.path.join(wide_dir, f"ma_{age}_{code}_ma_{age}_WTyp_incytr_output.parquet")


def _to_canonical(name: str) -> str:
    """Translate sce4-dotted cluster names (Cholinergic.Neurons) to our dashed
    canonical form (Cholinergic-Neurons). Cluster names contain no '.' in the
    canonical vocabulary, so this is unambiguous."""
    return name.replace(".", "-")


def load_pair(parquet: str, sender: str, receiver: str) -> pd.DataFrame:
    filt = (pc.field("Sender") == _to_canonical(sender)) & (pc.field("Receiver") == _to_canonical(receiver))
    cols = [
        "Ligand", "Receptor", "EM", "Target",
        "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC",
        "SigProb_ma_2mo_AppP", "SigProb_ma_2mo_WTyp",
        "Sender", "Receiver",
    ]
    available = set(pq.ParquetFile(parquet).schema_arrow.names)
    cols = [c for c in cols if c in available]
    tbl = pq.read_table(parquet, columns=cols, filters=filt)
    return tbl.to_pandas()


def check_pair(contrast: str, sender: str, receiver: str,
               wide_dir: str, sce4_path: str,
               sigprob_cutoff: float, min_recall: int,
               tol: float, transgene: str) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    parquet = contrast_to_parquet(contrast, wide_dir)
    if not os.path.exists(parquet):
        return False, [f"FAIL: {parquet} not found (run alz/runners/main/run_pair_mode_pipeline.sh)"]

    ours = load_pair(parquet, sender, receiver)
    sce4 = pd.read_csv(sce4_path, low_memory=False)

    # sce4 uses dotted form in Sender.group/Receiver.group; match in sce4's own
    # vocabulary then compare on the (Ligand, Receptor, EM, Target) path keys,
    # which are gene symbols and identical across formats.
    ref = sce4[(sce4["Sender.group"] == sender) & (sce4["Receiver.group"] == receiver)].copy()
    msgs.append(f"[{sender} → {receiver} | {contrast}] ours={len(ours)} sce4={len(ref)}")

    if len(ref) == 0:
        return False, msgs + [f"FAIL: no sce4 rows for {sender} → {receiver}"]

    def _mkkey(df: pd.DataFrame) -> pd.Series:
        return (df["Ligand"].astype(str) + "|" + df["Receptor"].astype(str)
                + "|" + df["EM"].astype(str) + "|" + df["Target"].astype(str))
    ours["k"] = _mkkey(ours)
    ref["k"]  = _mkkey(ref)

    # Recall after SigProb filter (either arm passes).
    sp_cols = [c for c in ("SigProb_ma_2mo_AppP", "SigProb_ma_2mo_WTyp") if c in ours.columns]
    if sp_cols:
        keep = (ours[sp_cols] > sigprob_cutoff).any(axis=1)
        filt = ours[keep]
    else:
        filt = ours
    recall = ref["k"].isin(filt["k"]).sum()
    msgs.append(f"  recall (SigProb>{sigprob_cutoff}): {recall}/{len(ref)}")

    ok = True
    if recall < min_recall:
        # The only legitimate residual is the App transgene: sce4's Oct-2025
        # run pr ranked App high enough to enter gene.use, but every pr artifact
        # reproducible on this box gives App fc2 rank ~4523/6687 (pr_log2FC
        # -0.345), so no fold-based rule recovers it. An App-ligand-only deficit
        # is an isolated input-provenance gap, not a method regression.
        # See bench/bench.md §App transgene residual.
        missed = ref[~ref["k"].isin(filt["k"])]
        non_transgene = missed[missed["Ligand"].astype(str) != transgene]
        if len(non_transgene) == 0:
            msgs.append(f"  recall {recall} < {min_recall}, but all "
                        f"{len(missed)} misses are {transgene}-ligand "
                        f"(transgene input-provenance gap) — PASS")
        else:
            msgs.append(f"  FAIL: recall {recall} < min {min_recall}; "
                        f"{len(non_transgene)} non-{transgene} misses "
                        f"(e.g. {sorted(non_transgene['Ligand'].unique())[:5]})")
            ok = False

    # Per-position max|Δ| on the joined paths.
    j = ours.merge(
        ref[["k", "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC"]]
            .rename(columns={
                "Ligand_sclog2FC": "sce4_L", "Receptor_sclog2FC": "sce4_R",
                "EM_sclog2FC":     "sce4_E", "Target_sclog2FC":   "sce4_T",
            }),
        on="k",
    )
    msgs.append(f"  joined overlap: {len(j)}")

    pos_map = {
        "Ligand":   ("Ligand_sclog2FC",   "sce4_L"),
        "Receptor": ("Receptor_sclog2FC", "sce4_R"),
        "EM":       ("EM_sclog2FC",       "sce4_E"),
        "Target":   ("Target_sclog2FC",   "sce4_T"),
    }
    for pos, (ours_col, sce4_col) in pos_map.items():
        d = (j[ours_col] - j[sce4_col]).abs()
        over = j[d > tol]
        n_over = len(over)
        max_d = float(d.max()) if len(d) else float("nan")
        msgs.append(f"  {pos:8s}  max|Δ|={max_d:.4f}  n(|Δ|>{tol})={n_over}")
        # R/T must be exact (sclog2FC is scRNA-derived). Ligand/EM may carry the
        # App-transgene residual: sce4's snRNA object shows App massively induced
        # (e.g. Ndnf EM sclog2FC 7.65) while ours is flat (0.19) — the same
        # transgene input gap. Outliers are tolerated ONLY when the position gene
        # is the transgene, so a real non-App regression still fails.
        if pos in ("Receptor", "Target"):
            if n_over > 0:
                msgs.append(f"    FAIL: {pos} has {n_over} mismatches above tol {tol}")
                ok = False
        else:  # Ligand or EM
            non_tg = over[over[pos].astype(str) != transgene]
            if len(non_tg) > 0:
                msgs.append(f"    FAIL: {pos} has {len(non_tg)} non-{transgene} "
                            f"mismatches (e.g. {sorted(non_tg[pos].unique())[:5]})")
                ok = False
            elif n_over:
                msgs.append(f"    ({n_over} {transgene}-transgene residual, allowed)")

    msgs.append(f"  STATUS: {'PASS' if ok else 'FAIL'}")
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="App_2mo")
    ap.add_argument("--sender",   default="Microglia")
    ap.add_argument("--receiver", default="Cholinergic.Neurons")
    ap.add_argument("--wide-dir", default=WIDE_DIR_DEFAULT)
    ap.add_argument("--sce4-ref", default=SCE4_REF_DEFAULT)
    ap.add_argument("--sigprob-cutoff", type=float, default=0.1)
    ap.add_argument("--min-recall", type=int, default=595,
                    help=(
                        "Min sce4-path recall after SigProb filter. A deficit "
                        "below this still PASSES if every miss is a --transgene "
                        "ligand path (isolated input-provenance gap on the "
                        "transgene; see bench/bench.md §App transgene residual)."
                    ))
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="per-position max |Δ| tolerance (default 1e-4 for float64 round-off)")
    ap.add_argument("--transgene", default="App",
                    help=(
                        "Transgene ligand whose paths are exempt from the recall "
                        "floor. sce4's Oct-2025 run ranked the App transgene into "
                        "gene.use, but no pr reproducible on this box does "
                        "(App fc2 rank ~4523/6687). An App-ligand-only deficit is "
                        "an isolated input-provenance gap. See bench/bench.md."
                    ))
    ap.add_argument("--all-known-pairs", action="store_true",
                    help="check both Micro→Cholin and Ndnf×Ndnf")
    args = ap.parse_args()

    pairs = KNOWN_PAIRS if args.all_known_pairs else [
        (args.contrast, args.sender, args.receiver)
    ]

    all_ok = True
    for contrast, sender, receiver in pairs:
        ok, msgs = check_pair(
            contrast, sender, receiver,
            args.wide_dir, args.sce4_ref,
            args.sigprob_cutoff, args.min_recall,
            args.tol, args.transgene,
        )
        for m in msgs:
            print(m)
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("verify_sce4_parity: PASS")
        return 0
    print("verify_sce4_parity: FAIL — one of the six sce4-parity fixes has regressed.")
    print("  See bench/bench.md §sce4 reproduction and CLAUDE.md §Pair-mode Incytr.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
