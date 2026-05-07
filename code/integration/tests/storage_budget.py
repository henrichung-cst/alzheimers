"""Storage budget checks for normalized Incytr outputs.

Usage:
  python code/integration/tests/storage_budget.py --root outputs/incytr
"""

from __future__ import annotations

import argparse
import os


GB = 1024 ** 3


def tree_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def child_dirs(path):
    if not os.path.isdir(path):
        return []
    return [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, name))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join("outputs", "incytr"))
    parser.add_argument("--universe-total-gb", type=float, default=25.0)
    parser.add_argument("--scoring-avg-gb", type=float, default=2.0)
    parser.add_argument("--config-avg-gb", type=float, default=0.2)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    universes = child_dirs(os.path.join(root, "universes"))
    scorings = child_dirs(os.path.join(root, "scoring"))
    configs = child_dirs(os.path.join(root, "configs"))

    universe_total = sum(tree_size(p) for p in universes)
    scoring_total = sum(tree_size(p) for p in scorings)
    config_total = sum(tree_size(p) for p in configs)

    scoring_avg = scoring_total / max(len(scorings), 1)
    config_avg = config_total / max(len(configs), 1)

    print(f"universes: {len(universes)} dirs, {universe_total / GB:.2f} GB total")
    print(f"scoring:   {len(scorings)} dirs, {scoring_avg / GB:.2f} GB avg")
    print(f"configs:   {len(configs)} dirs, {config_avg / GB:.2f} GB avg")

    failures = []
    if universe_total > args.universe_total_gb * GB:
        failures.append(
            f"universe total {universe_total / GB:.2f} GB > {args.universe_total_gb:.2f} GB"
        )
    if scorings and scoring_avg > args.scoring_avg_gb * GB:
        failures.append(
            f"scoring avg {scoring_avg / GB:.2f} GB > {args.scoring_avg_gb:.2f} GB"
        )
    if configs and config_avg > args.config_avg_gb * GB:
        failures.append(
            f"config avg {config_avg / GB:.2f} GB > {args.config_avg_gb:.2f} GB"
        )
    if failures:
        raise SystemExit("storage budget exceeded:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
