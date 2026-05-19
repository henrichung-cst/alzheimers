"""Generate spine-selection figure + summary table from cluster_spine.csv.

Reads `data/incytr_frozen/v2_46clusters/spines/levy_t5/cluster_spine.csv` and writes:
  outputs/reports/decomposition/levy_t5/cluster_spine_summary.csv
  outputs/reports/decomposition/levy_t5/cluster_spine_selection.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPINE_CSV = Path("data/incytr_frozen/v2_46clusters/spines/levy_t5/cluster_spine.csv")
OUT_DIR = Path("outputs/reports/decomposition/levy_t5")

TIER_ORDER = ["full_rank", "partial", "severe", "unnamed", "fails_gate"]
TIER_LABEL = {
    "full_rank": "Full rank (kept)",
    "partial": "Partial rank",
    "severe": "Severely deficient",
    "unnamed": "Unnamed",
    "fails_gate": "Fails cell-count gate",
}
TIER_COLOR = {
    "full_rank": "#2b8cbe",
    "partial": "#fdae61",
    "severe": "#d7191c",
    "unnamed": "#999999",
    "fails_gate": "#54278f",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SPINE_CSV)

    summary = (
        df.groupby("tier")
        .agg(n_clusters=("cluster_name", "size"), n_nuclei=("n_cells", "sum"))
        .reindex(TIER_ORDER)
        .reset_index()
    )
    total_nuclei = int(summary["n_nuclei"].sum())
    summary["pct_nuclei"] = (100 * summary["n_nuclei"] / total_nuclei).round(2)
    summary["in_spine"] = summary["tier"].eq("full_rank")
    summary.to_csv(OUT_DIR / "cluster_spine_summary.csv", index=False)

    fig, (ax_bar, ax_stack) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]}
    )

    df_sorted = df.sort_values("n_cells", ascending=False).reset_index(drop=True)
    colors = df_sorted["tier"].map(TIER_COLOR)
    x = np.arange(len(df_sorted))
    ax_bar.bar(x, df_sorted["n_cells"], color=colors, edgecolor="white", linewidth=0.4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(df_sorted["cluster_name"], rotation=90, fontsize=7)
    ax_bar.set_ylabel("Nuclei (n)")
    ax_bar.set_yscale("log")
    ax_bar.set_title("46 Song clusters by nucleus count — Levy-t5 spine selection")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIER_COLOR[t]) for t in TIER_ORDER
    ]
    labels = [
        f"{TIER_LABEL[t]} (n={int(summary.loc[summary.tier==t, 'n_clusters'].iloc[0])}, "
        f"{summary.loc[summary.tier==t, 'pct_nuclei'].iloc[0]:.1f}%)"
        for t in TIER_ORDER
    ]
    ax_bar.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    n_in_spine = int(df["in_spine"].sum())
    boundary_x = n_in_spine - 0.5
    ax_bar.axvline(boundary_x, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_bar.text(
        boundary_x, ax_bar.get_ylim()[1] * 0.7, "  spine boundary",
        fontsize=8, va="top", ha="left",
    )

    bottom = 0.0
    for tier in TIER_ORDER:
        row = summary[summary.tier == tier].iloc[0]
        height = float(row["pct_nuclei"])
        ax_stack.bar(
            0, height, bottom=bottom, color=TIER_COLOR[tier], width=0.6,
            edgecolor="white", linewidth=0.8,
        )
        if height >= 1.0:
            ax_stack.text(
                0, bottom + height / 2, f"{height:.1f}%",
                ha="center", va="center", fontsize=9,
                color="white" if tier != "unnamed" else "black",
            )
        bottom += height
    ax_stack.set_ylim(0, 100)
    ax_stack.set_xlim(-0.6, 0.6)
    ax_stack.set_xticks([])
    ax_stack.set_ylabel("% of total nuclei")
    ax_stack.set_title(f"Coverage\n(N = {total_nuclei:,} nuclei)")

    fig.tight_layout()
    out_png = OUT_DIR / "cluster_spine_selection.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_DIR/'cluster_spine_summary.csv'}")
    print(f"Wrote {out_png}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
