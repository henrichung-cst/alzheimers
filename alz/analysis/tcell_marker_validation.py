#!/usr/bin/env python3
"""Marker-gene validation for ProjecTILs T-cell state calls."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DONORS = ("donor1", "donor2")
DEFAULT_INPUT_ROOT = Path("data/derived/tcells_incytr_inputs")
DEFAULT_MARKERS = Path("tcell_markers.txt")
DEFAULT_OUTDIR = Path("outputs/reports/tcell_marker_validation")

SIGNATURES = {
    "curated_tcell_markers": [],
    "tcell_core": ["CD3D", "CD3E", "TRAC", "ZAP70"],
    "cd4_lineage": ["CD4", "IL7R", "CCR7", "LTB"],
    "cd8_lineage": ["CD8A", "CD8B", "NKG7"],
    "exhaustion": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"],
    "progenitor_exhaustion": ["TCF7", "LEF1", "SELL", "CCR7", "IL7R"],
    "cytotoxic": ["GZMB", "GZMH", "GNLY", "NKG7", "PRF1", "EOMES"],
    "th17": ["RORC", "CCR6", "IL17A", "IL17F", "IL23R", "KLRB1"],
    "tfh": ["CXCR5", "BCL6", "ICOS", "PDCD1", "IL21"],
    "treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "naive_memory": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R"],
}

EXTRA_MARKERS = sorted({
    g for genes in SIGNATURES.values() for g in genes
} | {
    "CD4", "CD8A", "SMCHD1", "PPP1R11", "ALB1", "KLRC1", "TYK2",
    "HSP90AA1", "RAN", "H4C1", "MAPK14", "NDC80", "FGFR1", "ZYX",
    "PTPN6", "LY9", "NCK1",
})

EXPECTED_SIGNATURES = {
    "CD4CTLeomes": ["cd4_lineage", "cytotoxic"],
    "CD4CTLexh": ["cd4_lineage", "cytotoxic", "exhaustion"],
    "CD4CTLgnly": ["cd4_lineage", "cytotoxic"],
    "CD4Naive": ["cd4_lineage", "naive_memory"],
    "CD4Tfh": ["cd4_lineage", "tfh"],
    "CD4Th17": ["cd4_lineage", "th17"],
    "Treg": ["cd4_lineage", "treg"],
    "CD8CM": ["cd8_lineage", "naive_memory"],
    "CD8EM": ["cd8_lineage", "cytotoxic"],
    "CD8MAIT": ["cd8_lineage", "cytotoxic"],
    "CD8Naive": ["cd8_lineage", "naive_memory"],
    "CD8TEMRA": ["cd8_lineage", "cytotoxic"],
    "CD8Tex": ["cd8_lineage", "exhaustion", "cytotoxic"],
    "CD8Tpex": ["cd8_lineage", "progenitor_exhaustion"],
}

STATE_MARKER_LABELS = {
    "CD4CTLeomes": ["CD4", "EOMES", "GZMB"],
    "CD4CTLexh": ["CD4", "PDCD1", "TOX", "GZMB"],
    "CD4CTLgnly": ["CD4", "GNLY", "NKG7"],
    "CD4Naive": ["CD4", "CCR7", "TCF7", "IL7R"],
    "CD4Tfh": ["CD4", "CXCR5", "BCL6", "ICOS"],
    "CD4Th17": ["CD4", "RORC", "CCR6", "IL17A"],
    "Treg": ["CD4", "FOXP3", "IL2RA", "CTLA4"],
    "CD8CM": ["CD8A", "CD8B", "CCR7", "TCF7"],
    "CD8EM": ["CD8A", "CD8B", "GZMB", "NKG7"],
    "CD8MAIT": ["CD8A", "KLRB1", "NKG7"],
    "CD8Naive": ["CD8A", "CCR7", "TCF7", "IL7R"],
    "CD8TEMRA": ["CD8A", "GZMB", "PRF1", "GNLY"],
    "CD8Tex": ["CD8A", "PDCD1", "TOX", "LAG3"],
    "CD8Tpex": ["CD8A", "TCF7", "LEF1", "IL7R"],
}


def _read_marker_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"marker file not found: {path}")
    out: list[str] = []
    for line in path.read_text().splitlines():
        gene = line.strip()
        if gene and not gene.startswith("#"):
            out.append(gene)
    return list(dict.fromkeys(out))


def _parse_group(col: str) -> tuple[str, int]:
    match = re.match(r"^(.+)__d(\d+)$", col)
    if not match:
        raise ValueError(f"unexpected aggexp state/day column: {col}")
    return match.group(1), int(match.group(2))


def _zscore_rows_leave_same_state_out(
    df: pd.DataFrame,
    group_info: dict[str, tuple[str, int]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    z = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    background_sizes: dict[str, int] = {}
    for col, (state, _day) in group_info.items():
        background_cols = [
            bg_col
            for bg_col, (bg_state, _bg_day) in group_info.items()
            if bg_state != state
        ]
        background_sizes[col] = len(background_cols)
        if not background_cols:
            z[col] = np.nan
            continue
        background = df[background_cols]
        mean = background.mean(axis=1)
        sd = background.std(axis=1).replace(0, np.nan)
        z[col] = df[col].sub(mean).div(sd)
    return z.fillna(0.0), background_sizes


def _coverage_rows(donor: str, all_markers: list[str], genes: set[str], panels: dict[str, list[str]]) -> list[dict]:
    rows: list[dict] = []
    for gene in all_markers:
        rows.append({
            "donor": donor,
            "panel": "all_requested_markers",
            "gene": gene,
            "present": gene in genes,
        })
    for panel, panel_genes in panels.items():
        for gene in panel_genes:
            rows.append({
                "donor": donor,
                "panel": panel,
                "gene": gene,
                "present": gene in genes,
            })
    return rows


def _plot_signature_heatmap(donor: str, sig: pd.DataFrame, outdir: Path) -> None:
    sig = sig.sort_values(["day", "state", "signature"])
    signatures = list(SIGNATURES.keys())
    groups = (
        sig[["state", "day"]]
        .drop_duplicates()
        .sort_values(["day", "state"])
        .apply(lambda r: f"{r.state}__d{int(r.day)}", axis=1)
        .tolist()
    )
    mat = pd.DataFrame(index=signatures, columns=groups, dtype=float)
    for row in sig.itertuples(index=False):
        mat.loc[row.signature, f"{row.state}__d{int(row.day)}"] = row.score
    mat = mat.fillna(0.0)

    fig_w = max(11, 0.22 * len(groups) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5.6))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    ax.set_yticks(range(len(mat.index)), mat.index)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=65, ha="right", fontsize=7)
    ax.set_title(f"{donor}: marker-signature z-score support by ProjecTILs state/day", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, extend="both")
    cbar.set_label("mean marker z-score vs other states")
    fig.tight_layout()
    fig.savefig(outdir / f"{donor}_signature_score_heatmap.png", dpi=220)
    fig.savefig(outdir / f"{donor}_signature_score_heatmap.pdf")
    plt.close(fig)


def _plot_expected_support(donor: str, expected: pd.DataFrame, outdir: Path) -> None:
    if expected.empty:
        return
    expected = expected.sort_values(["state", "day"])
    states = sorted(expected["state"].unique())
    days = sorted(expected["day"].unique())
    mat = pd.DataFrame(index=states, columns=days, dtype=float)
    for row in expected.itertuples(index=False):
        mat.loc[row.state, row.day] = row.expected_signature_score

    ylabels = [
        f"{state} ({', '.join(STATE_MARKER_LABELS.get(state, []))})"
        if STATE_MARKER_LABELS.get(state) else state
        for state in states
    ]

    fig, ax = plt.subplots(figsize=(10.8, max(4.5, 0.42 * len(states) + 2)))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(states)), ylabels)
    ax.set_xticks(range(len(days)), [f"d{d}" for d in days])
    ax.set_xlabel("Day")
    ax.set_title(f"{donor}: expected marker support for each ProjecTILs state", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, extend="both")
    cbar.set_label("mean expected-signature z-score vs other states")
    fig.tight_layout()
    fig.savefig(outdir / f"{donor}_expected_state_support_heatmap.png", dpi=220)
    fig.savefig(outdir / f"{donor}_expected_state_support_heatmap.pdf")
    plt.close(fig)


def _plot_expected_support_composition(
    donor: str,
    counts: pd.DataFrame,
    expected: pd.DataFrame,
    outdir: Path,
    label_fraction_threshold: float = 0.20,
) -> pd.DataFrame:
    if counts.empty:
        return pd.DataFrame()

    support_cols = [
        "state",
        "day",
        "state_marker_label",
        "expected_signature_score",
        "support",
    ]
    composition = counts.copy()
    composition["state"] = composition["state"].astype(str)
    composition["day"] = composition["day"].astype(int)
    composition["n_cells"] = composition["n_cells"].astype(int)
    if not expected.empty:
        composition = composition.merge(
            expected[support_cols],
            on=["state", "day"],
            how="left",
        )
    else:
        composition["state_marker_label"] = composition["state"]
        composition["expected_signature_score"] = np.nan
        composition["support"] = "not_tested"

    composition["state_marker_label"] = composition["state_marker_label"].fillna(composition["state"])
    composition["support"] = composition["support"].fillna("not_tested")
    totals = composition.groupby("day")["n_cells"].transform("sum")
    composition["day_total_cells"] = totals
    composition["day_fraction"] = composition["n_cells"] / totals.replace(0, np.nan)
    composition["day_percent"] = composition["day_fraction"] * 100.0
    composition["label_shown"] = composition["day_fraction"] > label_fraction_threshold
    composition = composition.sort_values(["day", "state"])
    composition.to_csv(
        outdir / f"{donor}_expected_state_support_composition.csv",
        index=False,
    )

    days = sorted(composition["day"].unique())
    states = [state for state in STATE_MARKER_LABELS if state in set(composition["state"])]
    states += sorted(set(composition["state"]) - set(states))
    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=-2.0, vmax=2.0)
    missing_color = "#d9d9d9"

    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(days) + 4.5), 7.2))
    bottoms = np.zeros(len(days), dtype=float)
    day_positions = np.arange(len(days))
    for state in states:
        sub = composition[composition["state"] == state].set_index("day")
        heights = np.array([sub.loc[day, "day_percent"] if day in sub.index else 0 for day in days])
        segment_colors = []
        for day in days:
            if day not in sub.index or pd.isna(sub.loc[day, "expected_signature_score"]):
                segment_colors.append(missing_color)
            else:
                segment_colors.append(cmap(norm(sub.loc[day, "expected_signature_score"])))
        bars = ax.bar(
            day_positions,
            heights,
            bottom=bottoms,
            width=0.68,
            color=segment_colors,
            edgecolor="white",
            linewidth=0.45,
        )
        for i, (day, height) in enumerate(zip(days, heights)):
            if height <= 0 or day not in sub.index:
                continue
            row = sub.loc[day]
            if not bool(row["label_shown"]):
                continue
            markers = STATE_MARKER_LABELS.get(state, [])
            marker_label = ", ".join(markers)
            marker_lines = textwrap.wrap(marker_label, width=18) if marker_label else []
            label = "\n".join([state] + marker_lines)
            ax.text(
                bars[i].get_x() + bars[i].get_width() / 2,
                bottoms[i] + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                linespacing=0.95,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )
        bottoms += heights

    ax.set_xticks(day_positions, [f"d{day}" for day in days])
    ax.set_xlim(-0.85, len(days) - 0.15)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Day")
    ax.set_ylabel("Percent of cells assigned to ProjecTILs state")
    fig.suptitle(
        f"{donor}: ProjecTILs composition with expected marker support labels",
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        f"Labels shown only for state/day segments > {label_fraction_threshold:.0%} of that day's cells.",
        fontsize=9,
        color="#555555",
        fontweight="normal",
        pad=8,
    )
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.035,
        pad=0.025,
        extend="both",
    )
    cbar.set_label("expected marker support z-score vs other states")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / f"{donor}_expected_state_support_composition_stacked.png", dpi=220)
    fig.savefig(outdir / f"{donor}_expected_state_support_composition_stacked.pdf")
    plt.close(fig)
    return composition


def _plot_trajectories(donor: str, expected: pd.DataFrame, sig: pd.DataFrame, outdir: Path) -> None:
    selected = {
        "CD8Tex": ["exhaustion", "cytotoxic", "cd8_lineage"],
        "CD8Tpex": ["progenitor_exhaustion", "cd8_lineage", "exhaustion"],
        "CD4Th17": ["th17", "cd4_lineage"],
        "CD4Tfh": ["tfh", "cd4_lineage"],
        "Treg": ["treg", "cd4_lineage"],
        "CD4CTLexh": ["exhaustion", "cytotoxic", "cd4_lineage"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=False)
    axes = axes.reshape(-1)
    for ax, (state, signatures) in zip(axes, selected.items()):
        sub = sig[(sig["state"] == state) & (sig["signature"].isin(signatures))]
        if sub.empty:
            ax.axis("off")
            continue
        for signature in signatures:
            line = sub[sub["signature"] == signature].sort_values("day")
            if not line.empty:
                ax.plot(line["day"], line["score"], marker="o", linewidth=1.8, label=signature)
        ax.axhline(0, color="#9e9e9e", linewidth=0.8)
        marker_genes = STATE_MARKER_LABELS.get(state, [])
        title = f"{state}\n({', '.join(marker_genes)})" if marker_genes else state
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("day")
        ax.set_ylabel("mean marker z-score vs other states")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{donor}: marker trajectories for key ProjecTILs states", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / f"{donor}_key_state_marker_trajectories.png", dpi=220)
    fig.savefig(outdir / f"{donor}_key_state_marker_trajectories.pdf")
    plt.close(fig)


def _support_label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.25:
        return "moderate"
    if score >= -0.25:
        return "weak_or_neutral"
    return "discordant"


def _process_donor(donor: str, input_root: Path, all_markers: list[str], panels: dict[str, list[str]], outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scrna = input_root / donor / "scrna"
    agg = pd.read_csv(scrna / "aggexp_data.csv")
    counts = pd.read_csv(scrna / "cell_counts.csv")
    count_lookup = {
        (str(r.state), int(r.day)): int(r.n_cells)
        for r in counts.itertuples(index=False)
    }

    genes = set(agg["gene"].astype(str))
    coverage = pd.DataFrame(_coverage_rows(donor, all_markers, genes, panels))

    marker_genes = [g for g in all_markers if g in genes]
    avg = agg[agg["gene"].astype(str).isin(marker_genes)].copy()
    avg["gene"] = avg["gene"].astype(str)
    group_cols = [c for c in avg.columns if c != "gene"]
    group_info = {c: _parse_group(c) for c in group_cols}
    for col, (state, day) in group_info.items():
        n_cells = count_lookup.get((state, day))
        if not n_cells:
            avg[col] = np.nan
        else:
            avg[col] = avg[col] / n_cells

    avg = avg.set_index("gene")
    z, z_background_sizes = _zscore_rows_leave_same_state_out(avg, group_info)

    expression_rows = []
    for gene in avg.index:
        for col, value in avg.loc[gene].items():
            state, day = group_info[col]
            expression_rows.append({
                "donor": donor, "gene": gene, "state": state, "day": day,
                "avg_log_expression": float(value) if pd.notna(value) else np.nan,
                "z": float(z.loc[gene, col]),
                "z_background": "leave_same_state_out",
                "z_background_n_groups": z_background_sizes.get(col, 0),
            })
    expression = pd.DataFrame(expression_rows)

    sig_rows = []
    for signature, genes_for_sig in panels.items():
        present = [g for g in genes_for_sig if g in z.index]
        if not present:
            continue
        score_by_group = z.loc[present].mean(axis=0)
        for col, score in score_by_group.items():
            state, day = group_info[col]
            sig_rows.append({
                "donor": donor,
                "signature": signature,
                "state": state,
                "day": day,
                "score": float(score),
                "z_background": "leave_same_state_out",
                "z_background_n_groups": z_background_sizes.get(col, 0),
                "n_markers_present": len(present),
                "markers_present": ";".join(present),
            })
    sig = pd.DataFrame(sig_rows)

    expected_rows = []
    for state, expected_sigs in EXPECTED_SIGNATURES.items():
        sub = sig[(sig["state"] == state) & (sig["signature"].isin(expected_sigs))]
        if sub.empty:
            continue
        by_day = sub.groupby("day")["score"].mean()
        for day, score in by_day.items():
            marker_genes = STATE_MARKER_LABELS.get(state, [])
            expected_rows.append({
                "donor": donor,
                "state": state,
                "state_marker_label": (
                    f"{state} ({', '.join(marker_genes)})"
                    if marker_genes else state
                ),
                "marker_genes_shown_in_label": ";".join(marker_genes),
                "day": int(day),
                "expected_signatures": ";".join(expected_sigs),
                "expected_signature_score": float(score),
                "z_background": "leave_same_state_out",
                "support": _support_label(float(score)),
            })
    expected = pd.DataFrame(expected_rows)

    _plot_signature_heatmap(donor, sig, outdir)
    _plot_expected_support(donor, expected, outdir)
    _plot_expected_support_composition(donor, counts, expected, outdir)
    _plot_trajectories(donor, expected, sig, outdir)
    return coverage, expression, sig, expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    curated = _read_marker_file(args.markers)
    panels = {k: v[:] for k, v in SIGNATURES.items()}
    panels["curated_tcell_markers"] = curated
    all_markers = list(dict.fromkeys(curated + EXTRA_MARKERS))

    coverage_all = []
    expression_all = []
    sig_all = []
    expected_all = []
    for donor in DONORS:
        coverage, expression, sig, expected = _process_donor(
            donor, args.input_root, all_markers, panels, args.outdir
        )
        coverage_all.append(coverage)
        expression_all.append(expression)
        sig_all.append(sig)
        expected_all.append(expected)

    coverage_df = pd.concat(coverage_all, ignore_index=True)
    expression_df = pd.concat(expression_all, ignore_index=True)
    sig_df = pd.concat(sig_all, ignore_index=True)
    expected_df = pd.concat(expected_all, ignore_index=True)

    coverage_df.to_csv(args.outdir / "marker_gene_coverage.csv", index=False)
    expression_df.to_csv(args.outdir / "marker_expression_by_state_day.csv", index=False)
    sig_df.to_csv(args.outdir / "signature_scores_by_state_day.csv", index=False)
    expected_df.to_csv(args.outdir / "expected_state_marker_support.csv", index=False)
    pd.DataFrame([
        {
            "state": state,
            "state_marker_label": f"{state} ({', '.join(genes)})",
            "marker_genes_shown_in_label": ";".join(genes),
            "expected_signatures": ";".join(EXPECTED_SIGNATURES.get(state, [])),
            "z_background": "leave_same_state_out",
            "label_source": (
                "Markers selected as canonical validation hints; "
                "ProjecTILs state names come from functional.cluster labels."
            ),
        }
        for state, genes in STATE_MARKER_LABELS.items()
    ]).to_csv(args.outdir / "state_marker_label_map.csv", index=False)

    coverage_summary = (
        coverage_df.groupby(["donor", "panel"])
        .agg(n_requested=("gene", "nunique"), n_present=("present", "sum"))
        .reset_index()
    )
    coverage_summary["coverage_fraction"] = (
        coverage_summary["n_present"] / coverage_summary["n_requested"]
    )
    coverage_summary.to_csv(args.outdir / "marker_gene_coverage_summary.csv", index=False)

    print("Wrote marker validation outputs to", args.outdir)
    print("\nCoverage summary:")
    print(coverage_summary.to_string(index=False, formatters={"coverage_fraction": "{:.3f}".format}))
    print("\nExpected support summary:")
    if expected_df.empty:
        print("No expected support rows generated.")
    else:
        summary = (
            expected_df.groupby(["donor", "state"])
            .agg(mean_expected_score=("expected_signature_score", "mean"),
                 min_expected_score=("expected_signature_score", "min"),
                 n_days=("day", "nunique"))
            .reset_index()
            .sort_values(["donor", "mean_expected_score"], ascending=[True, False])
        )
        print(summary.to_string(index=False, formatters={
            "mean_expected_score": "{:.2f}".format,
            "min_expected_score": "{:.2f}".format,
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
