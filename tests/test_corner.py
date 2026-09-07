from __future__ import annotations

import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ciri.corner import (
    corner_parameters,
    prepare_corner_samples,
    prepare_marginal_samples,
    weighted_quantiles,
)
from ciri.palette import RACS_SURVEY_COLOURS


SCRIPT = Path(__file__).parents[1] / "scripts" / "figx-corner.py"
PARAMETER_NAMES = [name for name, _ in corner_parameters("galactic")]


class FakePosterior:
    def __init__(self, columns, weights):
        self.columns = columns
        self.weights = np.asarray(weights, dtype=np.float64)

    def __getitem__(self, key):
        return self.columns[key]

    def get_weights(self):
        return self.weights


def posterior_from_matrix(matrix, weights):
    matrix = np.asarray(matrix, dtype=np.float64)
    return FakePosterior(
        {name: matrix[:, index] for index, name in enumerate(PARAMETER_NAMES)},
        weights,
    )


def load_script(monkeypatch):
    # Running the file normally puts scripts/ on sys.path for paperplot.py.
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return runpy.run_path(str(SCRIPT))


def test_corner_cli_defaults_to_galactic_and_accepts_equatorial(monkeypatch):
    module = load_script(monkeypatch)
    required = [
        str(SCRIPT),
        "--mid1-artifact", "mid1.zip",
        "--low3-artifact", "low3.zip",
        "--output", "corner.pdf",
    ]
    monkeypatch.setattr(sys, "argv", required)
    assert module["parse_args"]().coordinates == "galactic"
    monkeypatch.setattr(sys, "argv", [*required, "--coordinates", "equatorial"])
    assert module["parse_args"]().coordinates == "equatorial"


def test_corner_uses_the_shared_racs_survey_colours(monkeypatch):
    module = load_script(monkeypatch)
    assert RACS_SURVEY_COLOURS == {
        "low3": "#7897C3",
        "mid1": "#B66737",
    }
    assert module["SAMPLE_COLORS"] == (
        RACS_SURVEY_COLOURS["mid1"],
        RACS_SURVEY_COLOURS["low3"],
    )


def test_marginal_interval_uses_equal_width_sign_boxes(monkeypatch):
    load_script(monkeypatch)
    from paperplot import PaperPlotter

    class Formatter:
        _round_to_sig = staticmethod(PaperPlotter._round_to_sig)
        _format_fixed = staticmethod(PaperPlotter._format_fixed)

    interval = PaperPlotter._format_interval(Formatter(), 4.02, 3.28, 4.77)
    sign_width = r"\widthof{$\scriptstyle-$}"

    assert interval == (
        rf"4.02^{{\mathmakebox[{sign_width}][c]{{+}}0.75}}"
        rf"_{{\mathmakebox[{sign_width}][c]{{-}}0.74}}"
    )


@pytest.mark.parametrize(
    ("frame", "longitude_label", "latitude_label"),
    [
        ("galactic", r"\ell^\circ", r"b^\circ"),
        ("equatorial", r"\mathrm{RA}^\circ", r"\mathrm{Dec}^\circ"),
    ],
)
def test_corner_parameters_are_frame_aware_with_stable_names(
    frame,
    longitude_label,
    latitude_label,
):
    parameters = corner_parameters(frame)
    assert [name for name, _ in parameters] == PARAMETER_NAMES
    assert dict(parameters)["dipole_longitude"] == longitude_label
    assert dict(parameters)["dipole_latitude"] == latitude_label


def test_prepare_corner_samples_preserves_galactic_rows_columns_and_weights():
    matrix = np.array(
        [
            [1.0, 100.0, 10.0, 20.0, 0.1, 3.0],
            [2.0, 200.0, 300.0, -40.0, 0.2, 4.0],
        ]
    )
    prepared, weights = prepare_corner_samples(
        posterior_from_matrix(matrix, [0.75, 0.25]),
        "galactic",
    )
    np.testing.assert_array_equal(prepared, matrix)
    np.testing.assert_array_equal(weights, [0.75, 0.25])


def test_prepare_corner_samples_converts_known_galactic_direction_only():
    matrix = np.array([[1.0, 100.0, 0.0, 0.0, 0.1, 3.0]])
    prepared, weights = prepare_corner_samples(
        posterior_from_matrix(matrix, [2.5]),
        "equatorial",
    )
    np.testing.assert_allclose(prepared[0, 2:4], [266.40499, -28.93617], atol=2e-4)
    np.testing.assert_array_equal(prepared[0, [0, 1, 4, 5]], matrix[0, [0, 1, 4, 5]])
    np.testing.assert_array_equal(weights, [2.5])


def test_prepare_corner_samples_filters_invalid_rows_coherently():
    matrix = np.array(
        [
            [1.0, 10.0, -10.0, 20.0, 0.1, 1.0],
            [2.0, np.nan, 20.0, 30.0, 0.2, 2.0],
            [3.0, 30.0, 30.0, 91.0, 0.3, 3.0],
            [4.0, 40.0, 40.0, 40.0, 0.4, 4.0],
            [5.0, 50.0, 50.0, 50.0, 0.5, 5.0],
        ]
    )
    prepared, weights = prepare_corner_samples(
        posterior_from_matrix(matrix, [1.0, 2.0, 3.0, np.nan, -1.0]),
        "galactic",
    )
    np.testing.assert_array_equal(prepared, [[1.0, 10.0, 350.0, 20.0, 0.1, 1.0]])
    np.testing.assert_array_equal(weights, [1.0])


def test_getdist_samples_uses_frame_labels_and_preserves_prepared_weights(monkeypatch):
    module = load_script(monkeypatch)
    posterior = posterior_from_matrix(
        [
            [1.0, 100.0, 0.0, 0.0, 0.1, 3.0],
            [2.0, 200.0, 10.0, 20.0, 0.2, 4.0],
            [3.0, 300.0, 20.0, 30.0, 0.3, 5.0],
        ],
        [0.75, 0.0, 0.25],
    )
    getdist_samples = module["getdist_samples"]
    monkeypatch.setitem(
        getdist_samples.__globals__,
        "load_final_posterior",
        lambda artifact, label: posterior,
    )
    samples = getdist_samples(Path("unused.zip"), "test", "equatorial")
    assert samples.paramNames.parWithName("dipole_longitude").label == r"\mathrm{RA}^\circ"
    assert samples.paramNames.parWithName("dipole_latitude").label == r"\mathrm{Dec}^\circ"
    np.testing.assert_array_equal(samples.weights, [0.75, 0.25])
    np.testing.assert_array_equal(
        samples.samples[:, [0, 1, 4, 5]],
        [[1, 100, .1, 3], [3, 300, .3, 5]],
    )


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
def test_marginal_titles_use_selected_frame_labels_after_exclusion(frame, monkeypatch):
    module = load_script(monkeypatch)
    posterior = posterior_from_matrix(
        [
            [1.0, 100.0, 10.0, 20.0, 0.1, 3.0],
            [2.0, 200.0, 20.0, 30.0, 0.2, 4.0],
        ],
        [0.75, 0.25],
    )
    getdist_samples = module["getdist_samples"]
    monkeypatch.setitem(
        getdist_samples.__globals__,
        "load_final_posterior",
        lambda artifact, label: posterior,
    )
    samples = [
        getdist_samples(Path("unused.zip"), "first", frame),
        getdist_samples(Path("unused.zip"), "second", frame),
    ]
    parameters = corner_parameters(frame)[1:]
    figure, axes = plt.subplots(len(parameters), len(parameters), squeeze=False)

    class Settings:
        title_limit_fontsize = 10
        axes_fontsize = 10

    class Plotter:
        subplots = axes
        settings = Settings()
        marginal_title_x = 0.0

        @staticmethod
        def _scaled_fontsize(*args):
            return 10

        @staticmethod
        def _weighted_quantiles(values, weights, quantiles):
            return np.quantile(values, quantiles)

        @staticmethod
        def _format_interval(median, lower, upper):
            return "interval"

    try:
        module["stack_marginal_titles"](Plotter(), samples, parameters)
        longitude_index = [name for name, _ in parameters].index("dipole_longitude")
        latitude_index = [name for name, _ in parameters].index("dipole_latitude")
        assert dict(parameters)["dipole_longitude"] in axes[longitude_index, longitude_index].texts[0].get_text()
        assert dict(parameters)["dipole_latitude"] in axes[latitude_index, latitude_index].texts[0].get_text()
        assert all("log_{10}" not in text.get_text() for axis in axes.flat for text in axis.texts)
    finally:
        plt.close(figure)


def test_prepare_corner_samples_rejects_when_no_positive_valid_weight_remains():
    matrix = np.array([[1.0, 100.0, 10.0, 20.0, 0.1, 3.0]])
    with pytest.raises(ValueError, match="no finite samples with positive weight"):
        prepare_corner_samples(posterior_from_matrix(matrix, [0.0]), "galactic")


def test_marginal_samples_omit_fixed_parameters_and_weighted_quantiles_match_titles():
    posterior = FakePosterior(
        {
            "observer_speed": np.array([1.0, 3.0]),
            "temp_beta": np.array([0.1, 0.3]),
        },
        [0.75, 0.25],
    )
    parameters, matrix, weights = prepare_marginal_samples(posterior, "equatorial")
    assert [name for name, _ in parameters] == ["observer_speed", "temp_beta"]
    np.testing.assert_array_equal(matrix, [[1.0, 0.1], [3.0, 0.3]])
    lower, median, upper = weighted_quantiles(
        matrix[:, 0], weights, [0.158655254, 0.5, 0.841344746]
    )
    assert lower < median < upper
    assert median == pytest.approx(1.5)
