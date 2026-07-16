import pandas as pd

from alz.viewer.shared.trajectory import annotate_trajectory_columns


def _rows(contrasts, pds, path="L|R|EM|T", series="donor1"):
    return pd.DataFrame(
        {
            "sender": ["sender"] * len(contrasts),
            "receiver": ["receiver"] * len(contrasts),
            "Path": [path] * len(contrasts),
            "contrast": contrasts,
            "PDS": pds,
            "donor": [series] * len(contrasts),
        }
    )


def test_custom_axis_classifies_tcell_series_and_gates_incomplete_paths():
    complete_up = _rows(
        ["d13_d2", "d17_d2", "d20_d2"], [1.0, 2.0, 3.0], path="up"
    )
    incomplete = _rows(
        ["d13_d2", "d20_d2"], [1.0, 2.0], path="missing"
    )
    complete_mixed = _rows(
        ["d13_d2", "d17_d2", "d20_d2"], [1.0, -2.0, 3.0], path="mixed"
    )
    other_series = _rows(
        ["d13_d2", "d17_d2", "d20_d2"], [1.0, 2.0, 3.0],
        path="other", series="donor2"
    )
    frame = pd.concat(
        [complete_up, incomplete, complete_mixed, other_series], ignore_index=True
    )

    annotated, recur_index, summary = annotate_trajectory_columns(
        frame,
        series_key=lambda df: df["donor"],
        axis_value=lambda df: df["contrast"].str.split("_", n=1).str[0],
        ordered_axis=("d13", "d17", "d20"),
        valid_series={"donor1"},
    )

    labels = annotated.groupby("Path", sort=False)["traj_labels"].first().to_dict()
    assert labels["up"] == "always-up;monotonic-up"
    assert labels["mixed"] == "mixed"
    assert labels["missing"] == ""
    assert labels["other"] == ""
    assert recur_index == {
        "sender||receiver||up": ["donor1"],
        "sender||receiver||mixed": ["donor1"],
    }
    assert summary == {
        "always-up": 1,
        "always-down": 0,
        "monotonic-up": 1,
        "monotonic-down": 0,
        "mixed": 1,
    }


def test_contrast_split_axis_classifies_disease_series():
    # AD/5xFAD shape: contrast = "{disease}_{timepoint}", series = disease.
    frame = pd.concat(
        [
            _rows(["App_2mo", "App_4mo", "App_6mo"], [1.0, 2.0, 3.0], path="app"),
            _rows(["Tau_2mo", "Tau_4mo", "Tau_6mo"], [-1.0, -2.0, -3.0], path="tau"),
        ],
        ignore_index=True,
    )
    annotated, _, summary = annotate_trajectory_columns(
        frame,
        series_key=lambda df: df["contrast"].str.split("_", n=1).str[0],
        axis_value=lambda df: df["contrast"].str.split("_", n=1).str[1],
        ordered_axis=("2mo", "4mo", "6mo"),
        valid_series={"App", "Tau"},
    )

    labels = annotated.groupby("Path", sort=False)["traj_labels"].first().to_dict()
    assert labels["app"] == "always-up;monotonic-up"
    assert labels["tau"] == "always-down;monotonic-down"
    assert summary == {
        "always-up": 1,
        "always-down": 1,
        "monotonic-up": 1,
        "monotonic-down": 1,
        "mixed": 0,
    }
