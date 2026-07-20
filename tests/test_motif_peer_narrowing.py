import pandas as pd
import pytest

from alz.cross_reference.motif_peer_narrowing import (
    DETECTION_FRAC_MIN,
    build_peer_sets,
    centered_cosine,
    compute_narrowing,
    narrowing_sections,
)


def _motif(seed: float, kin_type: str = "ser_thr") -> dict:
    # The rows are normalized PSSM-like vectors; the exact values are only
    # fixture evidence for the centered-cosine seam.
    return {
        "kin_type": kin_type,
        "matrix": [[seed, 1 - seed], [1 - seed, seed]],
    }


def test_centered_cosine_removes_uniform_background_and_type_partitions():
    motifs = {
        "A": _motif(0.9),
        "B": _motif(0.8),
        "Y": _motif(0.9, "tyrosine"),
    }
    peers = build_peer_sets(motifs, cosine_cut=0.60)

    assert centered_cosine(
        pd.DataFrame([[0.9, 0.1], [0.1, 0.9]]).to_numpy(),
        pd.DataFrame([[0.8, 0.2], [0.2, 0.8]]).to_numpy(),
    ) > 0.60
    assert [peer["kinase"] for peer in peers["A"]] == ["B"]
    assert peers["Y"] == []


def test_compute_narrowing_excludes_undetected_peers_and_flags_sole_source():
    detections = pd.DataFrame([
        {"kinase": "A", "cell_type": "c1", "detection_fraction": 0.50},
        {"kinase": "A", "cell_type": "c2", "detection_fraction": 0.20},
        {"kinase": "B", "cell_type": "c1", "detection_fraction": 0.25},
        {"kinase": "B", "cell_type": "c2", "detection_fraction": 0.00},
        {"kinase": "C", "cell_type": "c1", "detection_fraction": 0.00},
        {"kinase": "D", "cell_type": "c1", "detection_fraction": 0.25},
        {"kinase": "E", "cell_type": "c1", "detection_fraction": 0.25},
    ])
    peer_sets = {
        "A": [{"kinase": "B", "centered_cosine": 0.70},
              {"kinase": "C", "centered_cosine": 0.66},
              {"kinase": "D", "centered_cosine": 0.62},
              {"kinase": "E", "centered_cosine": 0.61}],
        "B": [], "C": [], "D": [], "E": [],
    }

    rows, stats = compute_narrowing("song", detections, peer_sets)

    by = {(r.kinase, r.cell_type): r for r in rows.itertuples(index=False)}
    # C is detected nowhere, so it is excluded from A's candidate set entirely.
    a_c1 = by[("A", "c1")]
    assert a_c1.motif_peers_informative == 4
    assert [p["kinase"] for p in a_c1.motif_peer_roster] == ["B", "D", "E"]
    # In c2 the twins are absent, so A is the sole plausible source there.
    assert by[("A", "c2")].motif_peers_detected == 1
    assert by[("A", "c2")].motif_peers_informative == 4
    # No floor: unique-motif detected centers (B, D, E) each emit a 1-of-1 row.
    for kinase in ("B", "D", "E"):
        assert by[(kinase, "c1")].motif_peers_detected == 1
        assert by[(kinase, "c1")].motif_peers_informative == 1
    assert "rows_suppressed_n_lt_4" not in stats
    assert stats["kinases_unique_motif"] == 3
    assert stats["sole_source_kinases"] == 4  # A (in c2), B, D, E
    assert stats["sole_source_rows"] == 4
    assert DETECTION_FRAC_MIN == 0.10


def test_compute_narrowing_excludes_undetected_center():
    detections = pd.DataFrame([
        {"kinase": "A", "cell_type": "c1", "detection_fraction": 0.0},
        {"kinase": "B", "cell_type": "c1", "detection_fraction": 0.2},
    ])
    rows, stats = compute_narrowing(
        "song",
        detections,
        {"A": [{"kinase": "B", "centered_cosine": 0.7}], "B": []},
    )
    # A is detected nowhere → no A rows. B is detected → a 1-of-1 sole-source row.
    assert set(rows["kinase"]) == {"B"}
    assert stats["kinases_detected_anywhere"] == 1
    assert stats["sole_source_rows"] == 1


def test_detection_grain_conflicts_are_not_silently_deduplicated():
    from alz.cross_reference.motif_peer_narrowing import _validate_detection_grain

    frame = pd.DataFrame([
        {"kinase": "A", "cell_type": "c1", "fraction": 0.1},
        {"kinase": "A", "cell_type": "c1", "fraction": 0.2},
    ])
    with pytest.raises(ValueError, match="contrast-invariant"):
        _validate_detection_grain(frame, cohort="song", fraction_column="fraction")


def test_narrowing_sections_scopes_cohorts_and_rehydrates_losslessly():
    # The viewer slice must carry only the requested cohorts, and the compacted
    # form must reconstruct exactly the roster the fat form carried.
    payload = {
        "schema_version": 1,
        "motif_cosine_cut": 0.60,
        "detection_fraction_min": DETECTION_FRAC_MIN,
        "peer_roster": {"A": []},
        "cohorts": {
            "song": {"stats": {"rows_emitted": 2}, "rows": [
                {"kinase": "A", "cell_type": "c1", "detection_fraction": 0.5,
                 "motif_peers_detected": 2, "motif_peers_informative": 4,
                 "motif_peer_roster": [
                     {"kinase": "B", "detection_fraction": 0.4},
                     {"kinase": "C", "detection_fraction": 0.02},
                     {"kinase": "D", "detection_fraction": 0.01}]},
                {"kinase": "A", "cell_type": "c2", "detection_fraction": 0.3,
                 "motif_peers_detected": 1, "motif_peers_informative": 4,
                 "motif_peer_roster": [
                     {"kinase": "B", "detection_fraction": 0.01},
                     {"kinase": "C", "detection_fraction": 0.03},
                     {"kinase": "D", "detection_fraction": 0.05}]},
            ]},
            "tcell": {"stats": {}, "rows": []},
        },
    }

    scoped = narrowing_sections(payload, ("song",))
    assert set(scoped["cohorts"]) == {"song"}
    assert "peer_roster" not in scoped
    assert "stats" not in scoped["cohorts"]["song"]

    block = scoped["cohorts"]["song"]
    assert block["peer_names"] == {"A": ["B", "C", "D"]}
    for compact, original in zip(block["rows"], payload["cohorts"]["song"]["rows"]):
        rehydrated = [
            {"kinase": name, "detection_fraction": fraction}
            for name, fraction in zip(block["peer_names"][compact["kinase"]],
                                      compact["motif_peer_fractions"])
        ]
        assert rehydrated == original["motif_peer_roster"]

    with pytest.raises(KeyError):
        narrowing_sections(payload, ("no_such_cohort",))


def test_narrowing_sections_rejects_unstable_peer_order():
    # Positional alignment is the whole compaction contract; a reordered roster
    # would silently mislabel every fraction.
    payload = {"cohorts": {"song": {"rows": [
        {"kinase": "A", "cell_type": "c1", "motif_peers_detected": 1,
         "motif_peers_informative": 4,
         "motif_peer_roster": [{"kinase": "B", "detection_fraction": 0.4}]},
        {"kinase": "A", "cell_type": "c2", "motif_peers_detected": 1,
         "motif_peers_informative": 4,
         "motif_peer_roster": [{"kinase": "C", "detection_fraction": 0.4}]},
    ]}}}
    with pytest.raises(ValueError, match="not stable"):
        narrowing_sections(payload, ("song",))
