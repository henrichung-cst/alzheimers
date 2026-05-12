"""Self-contained sanity test for omics_loaders.

Runs `python alz/integration/test_omics_loaders.py` (no test framework
dependency). Exits non-zero on failure. Exercises:

- canonical_animal_id corner cases
- descriptive-header animal extraction (incl. Ref_Pool skip)
- intersect_and_rekey: drops, ordering, and column rename
- end-to-end load on the bundled Song xlsx files (slow, ~12s)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import omics_loaders as ol  # noqa: E402


def test_canonical_animal_id():
    cases = [
        ("D092", "D92"),
        ("D92", "D92"),
        ("C198", "C198"),
        ("E049", "E49"),
        ("E50", "E50"),
        ("e137", "E137"),
        ("", None),
        (None, None),
    ]
    for inp, expected in cases:
        got = ol.canonical_animal_id(inp)
        assert got == expected, f"canonical_animal_id({inp!r}) = {got!r}, expected {expected!r}"
    print("  canonical_animal_id: ok")


def test_descriptive_header_extraction():
    cases = [
        ("1_C198(L)_M_2mo_WT_P1_128N", "C198"),
        ("37_E50(L)_M_4mo_WT_P1_128C", "E50"),
        ("4_C200 (N)_M_2mo_T22_P1_130N", "C200"),
        ("Ref_Pool_P1_126", None),
        ("", None),
        (None, None),
    ]
    for inp, expected in cases:
        got = ol.extract_animal_from_descriptive(inp)
        assert got == expected, f"extract_animal_from_descriptive({inp!r}) = {got!r}"
    print("  descriptive_header_extraction: ok")


def test_intersect_and_rekey_basic():
    # Two omics layers, three transcript animals — one drops out.
    transcript_meta = pd.DataFrame(
        {"animal_id": ["D092_x", "E049_y", "E137_z"]}
    )
    omics = {
        "pr": pd.DataFrame(
            {"D92": [1.0, 2.0], "E49": [3.0, 4.0]},
            index=["GeneA", "GeneB"],
        ),
        "ps": pd.DataFrame(
            {"D92": [5.0, 6.0], "E49": [7.0, 8.0]},
            index=["GeneA", "GeneB"],
        ),
    }
    transcript_map = ol.transcript_animal_canon_map(transcript_meta)
    assert transcript_map == {"D92": "D092_x", "E49": "E049_y", "E137": "E137_z"}

    rekeyed, kept, dropped = ol.intersect_and_rekey(omics, transcript_map)
    assert kept == ["D092_x", "E049_y"], kept
    assert dropped == ["E137_z"], dropped
    for layer in ("pr", "ps"):
        assert list(rekeyed[layer].columns) == kept
        assert rekeyed[layer].shape == (2, 2)
    # Spot-check value passthrough.
    assert rekeyed["pr"].loc["GeneA", "D092_x"] == 1.0
    assert rekeyed["pr"].loc["GeneB", "E049_y"] == 4.0
    print("  intersect_and_rekey_basic: ok")


def test_full_load_on_song_xlsx():
    """Slow: parses all three xlsx files. Skipped if files are missing."""
    missing = [s.path for s in ol.OMICS_SCHEMAS.values() if not os.path.isfile(s.path)]
    if missing:
        print(f"  full_load_on_song_xlsx: SKIPPED (missing files: {missing})")
        return
    omics = ol.load_omics_matrices()
    # All three layers share the same animal set in the Song dataset.
    cols_pr = set(omics["pr"].columns)
    cols_ps = set(omics["ps"].columns)
    cols_py = set(omics["py"].columns)
    assert cols_pr == cols_ps == cols_py, (
        f"Song dataset layers disagree on animal coverage: "
        f"pr-only={cols_pr - cols_ps}, ps-only={cols_ps - cols_pr}, py-only={cols_py - cols_pr}"
    )
    n = len(cols_pr)
    assert 60 <= n <= 80, f"unexpected n_animals in Song omics: {n}"
    # Sum-collapse keeps row indices unique (genes, not site_ids).
    for layer in ("ps", "py"):
        assert omics[layer].index.is_unique, f"{layer} has duplicate gene rows after collapse"
    print(f"  full_load_on_song_xlsx: ok ({n} animals; pr={omics['pr'].shape}, "
          f"ps={omics['ps'].shape}, py={omics['py'].shape})")


def main():
    print("omics_loaders sanity tests")
    test_canonical_animal_id()
    test_descriptive_header_extraction()
    test_intersect_and_rekey_basic()
    test_full_load_on_song_xlsx()
    print("All tests passed.")


if __name__ == "__main__":
    main()
