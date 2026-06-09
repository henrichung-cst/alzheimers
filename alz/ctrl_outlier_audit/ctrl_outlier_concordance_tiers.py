"""Combinatorial AD/suspect/clean concordance table for human kinase behavior.

Inputs are existing audit outputs:
  - ctrl_audit/reanalysis_mea/suspect_vs_AD_kinase_wide.csv
  - perdonor/kinase_donor_nes{,_pY}.csv
  - perdonor/kinase_donor_fdr{,_pY}.csv

Writes two read-only-derived CSVs under ctrl_audit/reanalysis_mea/:
  - concordance_tier_summary.csv
  - concordance_tier_kinases.csv

All combinations start from the same base requirement:
  - AD_vs_cleanCTRL group FDR < 0.25
  - suspect_vs_cleanCTRL group FDR < 0.25
  - AD and suspect group NES share direction

Then a 4 x 4 x 4 rule grid is applied to AD, suspect controls, and clean controls.
For AD and suspect controls, the expected individual direction is the shared
AD/suspect group direction. For clean controls, the expected direction is opposite.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

HUMAN = Path("outputs/reports/kinase_attribution_human")
PER = HUMAN / "perdonor"
OUT = HUMAN / "ctrl_audit" / "reanalysis_mea"

WIDE_PATH = OUT / "suspect_vs_AD_kinase_wide.csv"
SUMMARY_PATH = OUT / "concordance_tier_summary.csv"
KINASES_PATH = OUT / "concordance_tier_kinases.csv"

FDR_THRESH = 0.25

AD = ["AD-01", "AD-02", "AD-03", "AD-04", "AD-06", "AD-07", "AD-08", "AD-09", "AD-13", "AD-15"]
SUSPECT = ["CTRL-07", "CTRL-08", "CTRL-10"]
CLEAN = ["CTRL-01", "CTRL-02", "CTRL-03", "CTRL-04"]


def _normalize_ad_sample(sample: str) -> str:
    sample = sample.strip()
    if sample.isdigit():
        return f"AD-{int(sample):02d}"
    return sample


@dataclass(frozen=True)
class Rule:
    name: str
    label: str
    description_same: str
    description_opposite: str
    rank: int


RULES = [
    Rule(
        name="median_all_direction",
        label="median-all direction",
        description_same="median individual NES has the AD/suspect direction",
        description_opposite="median individual NES has the opposite direction",
        rank=1,
    ),
    Rule(
        name="all_direction",
        label="all direction",
        description_same="all individual NES values have the AD/suspect direction",
        description_opposite="all individual NES values have the opposite direction",
        rank=2,
    ),
    Rule(
        name="median_sig_only_direction",
        label="median-significant-only direction",
        description_same=(
            "median NES among individually significant samples has the AD/suspect direction"
        ),
        description_opposite=(
            "median NES among individually significant samples has the opposite direction"
        ),
        rank=3,
    ),
    Rule(
        name="all_sig_direction",
        label="all significant direction",
        description_same="all individuals are FDR < 0.25 and have the AD/suspect direction",
        description_opposite="all individuals are FDR < 0.25 and have the opposite direction",
        rank=4,
    ),
]


def _float(value: str) -> float:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.12g}"


def _read_matrix(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        out = {}
        for row in csv.DictReader(handle):
            kinase = row.pop("kinase")
            out[kinase] = {col: _float(value) for col, value in row.items()}
        return out


def _read_wide() -> list[dict[str, str | float]]:
    numeric = {
        "NES_suspect",
        "NES_AD",
        "delta_NES_suspect_minus_AD",
        "abs_delta_NES",
        "FDR_suspect",
        "FDR_AD",
        "NES_ADwithSuspect",
        "NES_AD_allCTRL",
        "median_nes_sig_only_allCTRL",
    }
    with WIDE_PATH.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            for col in numeric:
                row[col] = _float(row.get(col, ""))
            rows.append(row)
        return rows


def _track_files(track: str) -> tuple[Path, Path]:
    suffix = "" if track == "st" else "_pY"
    return PER / f"kinase_donor_nes{suffix}.csv", PER / f"kinase_donor_fdr{suffix}.csv"


def _values(matrix: dict[str, dict[str, float]], kinase: str, samples: list[str]) -> list[float]:
    vals = matrix.get(kinase, {})
    return [vals.get(sample, math.nan) for sample in samples]


def _group_stats(
    nes: dict[str, dict[str, float]],
    fdr: dict[str, dict[str, float]],
    kinase: str,
    samples: list[str],
    expected_sign: int,
) -> dict[str, float | int]:
    nes_values = _values(nes, kinase, samples)
    fdr_values = _values(fdr, kinase, samples)
    finite_nes = [value for value in nes_values if math.isfinite(value)]
    pairs = [(n, q) for n, q in zip(nes_values, fdr_values) if math.isfinite(n)]
    sig_nes = [n for n, q in pairs if q < FDR_THRESH]

    n_expected = sum(_sign(n) == expected_sign for n, _q in pairs)
    n_sig_expected = sum(_sign(n) == expected_sign and q < FDR_THRESH for n, q in pairs)
    n_sig = sum(q < FDR_THRESH for _n, q in pairs)

    return {
        "median_nes": median(finite_nes) if finite_nes else math.nan,
        "median_sig_only_nes": median(sig_nes) if sig_nes else math.nan,
        "n_tested": len(pairs),
        "n_expected_direction": n_expected,
        "n_sig_expected_direction": n_sig_expected,
        "n_sig_any_direction": n_sig,
    }


def _base_group_concordant(row: dict[str, str | float]) -> bool:
    nes_ad = row["NES_AD"]
    nes_suspect = row["NES_suspect"]
    return (
        isinstance(nes_ad, float)
        and isinstance(nes_suspect, float)
        and math.isfinite(nes_ad)
        and math.isfinite(nes_suspect)
        and row["FDR_AD"] < FDR_THRESH
        and row["FDR_suspect"] < FDR_THRESH
        and nes_ad * nes_suspect > 0
    )


def _annotate_rows() -> list[dict[str, str | float | int | bool]]:
    donor = {}
    for track in ("st", "py"):
        nes_path, fdr_path = _track_files(track)
        donor[track] = {"nes": _read_matrix(nes_path), "fdr": _read_matrix(fdr_path)}

    annotated = []
    for row in _read_wide():
        track = str(row["track"])
        kinase = str(row["kinase"])
        is_base = _base_group_concordant(row)
        direction_sign = _sign(row["NES_AD"]) if is_base else 0
        opposite_sign = -direction_sign if direction_sign else 0

        nes = donor[track]["nes"]
        fdr = donor[track]["fdr"]
        ad = _group_stats(nes, fdr, kinase, AD, direction_sign)
        suspect = _group_stats(nes, fdr, kinase, SUSPECT, direction_sign)
        clean = _group_stats(nes, fdr, kinase, CLEAN, opposite_sign)

        annotated.append({
            **row,
            "direction": "up" if direction_sign > 0 else "down" if direction_sign < 0 else "",
            "direction_sign": direction_sign,
            "is_group_concordant": is_base,
            "NES_clean_median": clean["median_nes"],
            "median_NES_AD_donors": ad["median_nes"],
            "median_NES_suspect_controls": suspect["median_nes"],
            "median_NES_clean_controls": clean["median_nes"],
            "median_sig_only_NES_AD_donors": ad["median_sig_only_nes"],
            "median_sig_only_NES_suspect_controls": suspect["median_sig_only_nes"],
            "median_sig_only_NES_clean_controls": clean["median_sig_only_nes"],
            "n_AD_same_direction": ad["n_expected_direction"],
            "n_AD_sig_same_direction": ad["n_sig_expected_direction"],
            "n_AD_sig_any_direction": ad["n_sig_any_direction"],
            "n_suspect_same_direction": suspect["n_expected_direction"],
            "n_suspect_sig_same_direction": suspect["n_sig_expected_direction"],
            "n_suspect_sig_any_direction": suspect["n_sig_any_direction"],
            "n_clean_opposite_direction": clean["n_expected_direction"],
            "n_clean_sig_opposite_direction": clean["n_sig_expected_direction"],
            "n_clean_sig_any_direction": clean["n_sig_any_direction"],
        })
    return annotated


def _passes_rule(rule: Rule, row: dict[str, str | float | int | bool], group: str) -> bool:
    if not row["is_group_concordant"]:
        return False

    if group == "ad":
        median_key = "median_NES_AD_donors"
        n_key = "n_AD_same_direction"
        n_sig_key = "n_AD_sig_same_direction"
        n_total = len(AD)
        expected_sign = int(row["direction_sign"])
    elif group == "suspect":
        median_key = "median_NES_suspect_controls"
        n_key = "n_suspect_same_direction"
        n_sig_key = "n_suspect_sig_same_direction"
        n_total = len(SUSPECT)
        expected_sign = int(row["direction_sign"])
    elif group == "clean":
        median_key = "median_NES_clean_controls"
        n_key = "n_clean_opposite_direction"
        n_sig_key = "n_clean_sig_opposite_direction"
        n_total = len(CLEAN)
        expected_sign = -int(row["direction_sign"])
    else:
        raise ValueError(group)

    sig_median_key = median_key.replace("median_NES", "median_sig_only_NES")
    median_ok = _sign(row[median_key]) == expected_sign
    sig_median_ok = _sign(row[sig_median_key]) == expected_sign
    all_ok = row[n_key] == n_total
    all_sig_ok = row[n_sig_key] == n_total

    if rule.name == "median_all_direction":
        return median_ok
    if rule.name == "all_direction":
        return all_ok
    if rule.name == "median_sig_only_direction":
        return sig_median_ok
    if rule.name == "all_sig_direction":
        return all_sig_ok
    raise ValueError(rule.name)


def _combination_id(ad_rule: Rule, suspect_rule: Rule, clean_rule: Rule) -> str:
    return f"ad_{_rule_code(ad_rule)}__sus_{_rule_code(suspect_rule)}__clean_{_rule_code(clean_rule, clean=True)}"


def _combination_label(ad_rule: Rule, suspect_rule: Rule, clean_rule: Rule) -> str:
    return f"AD: {ad_rule.label}; suspect: {suspect_rule.label}; clean: {_clean_rule_label(clean_rule)}"


def _rule_code(rule: Rule, clean: bool = False) -> str:
    codes = {
        "median_all_direction": "med_all",
        "all_direction": "all",
        "median_sig_only_direction": "med_sig",
        "all_sig_direction": "all_sig",
    }
    code = codes[rule.name]
    return f"{code}_opp" if clean else code


def _clean_rule_label(rule: Rule) -> str:
    return rule.label.replace("direction", "opposite")


def _members(
    rows: list[dict[str, str | float | int | bool]],
    ad_rule: Rule,
    suspect_rule: Rule,
    clean_rule: Rule,
) -> list[dict[str, str | float | int | bool]]:
    return [
        row for row in rows
        if _passes_rule(ad_rule, row, "ad")
        and _passes_rule(suspect_rule, row, "suspect")
        and _passes_rule(clean_rule, row, "clean")
    ]


def _write_summary(rows: list[dict[str, str | float | int | bool]]) -> None:
    fieldnames = [
        "combo",
        "ad_rule",
        "suspect_rule",
        "clean_rule",
        "total",
        "up",
        "down",
        "st",
        "py",
    ]
    with SUMMARY_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ad_rule in RULES:
            for suspect_rule in RULES:
                for clean_rule in RULES:
                    members = _members(rows, ad_rule, suspect_rule, clean_rule)
                    writer.writerow({
                        "combo": _combination_id(ad_rule, suspect_rule, clean_rule),
                        "ad_rule": _rule_code(ad_rule),
                        "suspect_rule": _rule_code(suspect_rule),
                        "clean_rule": _rule_code(clean_rule, clean=True),
                        "total": len(members),
                        "up": sum(row["direction"] == "up" for row in members),
                        "down": sum(row["direction"] == "down" for row in members),
                        "st": sum(row["track"] == "st" for row in members),
                        "py": sum(row["track"] == "py" for row in members),
                    })


def _write_kinases(rows: list[dict[str, str | float | int | bool]]) -> None:
    combo_fields = [
        _combination_id(ad_rule, suspect_rule, clean_rule)
        for ad_rule in RULES
        for suspect_rule in RULES
        for clean_rule in RULES
    ]
    fieldnames = [
        "kinase",
        "track",
        "residue",
        "direction",
        "base",
        "ad_nes",
        "sus_nes",
        "clean_nes",
        "ad_fdr",
        "sus_fdr",
        "delta_sus_ad",
        "abs_delta",
        "ad_med_all_nes",
        "sus_med_all_nes",
        "clean_med_all_nes",
        "ad_med_sig_nes",
        "sus_med_sig_nes",
        "clean_med_sig_nes",
        "ad_same",
        "ad_sig_same",
        "sus_same",
        "sus_sig_same",
        "clean_opp",
        "clean_sig_opp",
    ] + combo_fields
    with KINASES_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (str(r["track"]) != "st", not bool(r["is_group_concordant"]),
                           str(r["direction"]) != "up", r["abs_delta_NES"], str(r["kinase"])),
        ):
            out = {
                "kinase": row["kinase"],
                "track": row["track"],
                "residue": row["residue_type"],
                "direction": row["direction"],
                "base": int(bool(row["is_group_concordant"])),
                "ad_nes": _fmt(row["NES_AD"]),
                "sus_nes": _fmt(row["NES_suspect"]),
                "clean_nes": _fmt(row["NES_clean_median"]),
                "ad_fdr": _fmt(row["FDR_AD"]),
                "sus_fdr": _fmt(row["FDR_suspect"]),
                "delta_sus_ad": _fmt(row["delta_NES_suspect_minus_AD"]),
                "abs_delta": _fmt(row["abs_delta_NES"]),
                "ad_med_all_nes": _fmt(row["median_NES_AD_donors"]),
                "sus_med_all_nes": _fmt(row["median_NES_suspect_controls"]),
                "clean_med_all_nes": _fmt(row["median_NES_clean_controls"]),
                "ad_med_sig_nes": _fmt(row["median_sig_only_NES_AD_donors"]),
                "sus_med_sig_nes": _fmt(row["median_sig_only_NES_suspect_controls"]),
                "clean_med_sig_nes": _fmt(row["median_sig_only_NES_clean_controls"]),
                "ad_same": row["n_AD_same_direction"],
                "ad_sig_same": row["n_AD_sig_same_direction"],
                "sus_same": row["n_suspect_same_direction"],
                "sus_sig_same": row["n_suspect_sig_same_direction"],
                "clean_opp": row["n_clean_opposite_direction"],
                "clean_sig_opp": row["n_clean_sig_opposite_direction"],
            }
            for ad_rule in RULES:
                for suspect_rule in RULES:
                    for clean_rule in RULES:
                        combo_id = _combination_id(ad_rule, suspect_rule, clean_rule)
                        out[combo_id] = int(
                            _passes_rule(ad_rule, row, "ad")
                            and _passes_rule(suspect_rule, row, "suspect")
                            and _passes_rule(clean_rule, row, "clean")
                        )
            writer.writerow(out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AD/suspect/clean concordance tier tables."
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT),
        help="Directory for concordance_tier_summary.csv and concordance_tier_kinases.csv.",
    )
    parser.add_argument(
        "--wide-path",
        default=str(WIDE_PATH),
        help="Path to suspect_vs_AD_kinase_wide.csv.",
    )
    parser.add_argument(
        "--exclude-ad-samples",
        nargs="*",
        default=[],
        help="AD samples to remove from per-donor AD concordance checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    global AD, OUT, WIDE_PATH, SUMMARY_PATH, KINASES_PATH
    exclude_ad_samples = {_normalize_ad_sample(s) for s in args.exclude_ad_samples}
    unknown = sorted(exclude_ad_samples - set(AD))
    if unknown:
        raise ValueError(f"Excluded AD samples not in concordance AD list: {unknown}")
    AD = [sample for sample in AD if sample not in exclude_ad_samples]
    OUT = Path(args.out_dir)
    WIDE_PATH = Path(args.wide_path)
    SUMMARY_PATH = OUT / "concordance_tier_summary.csv"
    KINASES_PATH = OUT / "concordance_tier_kinases.csv"
    OUT.mkdir(parents=True, exist_ok=True)

    rows = _annotate_rows()
    _write_summary(rows)
    _write_kinases(rows)

    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {KINASES_PATH}")
    for ad_rule, suspect_rule, clean_rule in [
        (RULES[0], RULES[0], RULES[0]),
        (RULES[1], RULES[1], RULES[1]),
        (RULES[2], RULES[2], RULES[2]),
        (RULES[3], RULES[3], RULES[3]),
    ]:
        members = _members(rows, ad_rule, suspect_rule, clean_rule)
        print(
            f"{_combination_id(ad_rule, suspect_rule, clean_rule)} "
            f"total={len(members):3d} "
            f"up={sum(row['direction'] == 'up' for row in members):3d} "
            f"down={sum(row['direction'] == 'down' for row in members):3d}"
        )


if __name__ == "__main__":
    main()
