from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import runpy
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from ciri.artifact_io import (
    PER_ROUND_LOG_EVIDENCE_PATH,
    load_per_round_log_evidence,
)
from ciri.bayes_factors import (
    BayesFactor,
    EvidenceSummary,
    comparison_log_bayes_factor,
    format_value_uncertainty,
    render_log_bayes_table,
    summarize_log_evidence,
    write_log_bayes_json,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "tablex-log-bayes.py"
INITIAL_TABLE = """\
\\begin{tabular}{@{}lr@{}}
\\toprule
Model & $\\ln B$ \\\\
\\midrule
\\textit{Fiducial:} Free dipole; temperature response; clustering & \\textemdash{} \\\\
\\bottomrule
\\end{tabular}
"""


def write_evidence_artifact(path: Path, evidence: np.ndarray) -> Path:
    payload = BytesIO()
    np.save(payload, evidence)
    with ZipFile(path, "w") as archive:
        archive.writestr(PER_ROUND_LOG_EVIDENCE_PATH, payload.getvalue())
    return path


def evidence(logz, errors=None):
    logz = np.asarray(logz, dtype=np.float64)
    if errors is None:
        errors = np.zeros_like(logz)
    return np.vstack((logz, np.asarray(errors, dtype=np.float64)))


def test_load_per_round_log_evidence_valid(tmp_path):
    expected = evidence([-3.0, -2.0, -1.0], [0.3, 0.2, 0.1])
    artifact = write_evidence_artifact(tmp_path / "valid.zip", expected)
    actual = load_per_round_log_evidence(artifact)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64


def test_load_per_round_log_evidence_missing_member(tmp_path):
    artifact = tmp_path / "missing.zip"
    with ZipFile(artifact, "w") as archive:
        archive.writestr("unrelated.txt", "nothing")
    with pytest.raises(ValueError, match="missing posterior/per-round-log-evidence.npy"):
        load_per_round_log_evidence(artifact)


def test_load_per_round_log_evidence_rejects_corrupt_npy(tmp_path):
    artifact = tmp_path / "corrupt.zip"
    with ZipFile(artifact, "w") as archive:
        archive.writestr(PER_ROUND_LOG_EVIDENCE_PATH, b"not a numpy file")
    with pytest.raises(ValueError, match="not a valid NumPy array"):
        load_per_round_log_evidence(artifact)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.ones(4), "shape"),
        (np.ones((3, 4)), "shape"),
        (np.array([[1.0, np.nan], [0.1, 0.2]]), "must be finite"),
        (np.array([[1.0, 2.0], [0.1, -0.2]]), "must be nonnegative"),
    ],
)
def test_load_per_round_log_evidence_rejects_malformed_values(
    tmp_path,
    array,
    message,
):
    artifact = write_evidence_artifact(tmp_path / "invalid.zip", array)
    with pytest.raises(ValueError, match=message):
        load_per_round_log_evidence(artifact)


def test_load_per_round_log_evidence_rejects_non_numeric_array(tmp_path):
    artifact = write_evidence_artifact(
        tmp_path / "object.zip",
        np.array([["a", "b"], ["c", "d"]]),
    )
    with pytest.raises(ValueError, match="real numeric array"):
        load_per_round_log_evidence(artifact)


def test_summarize_discards_positionally_and_uses_ddof_one():
    per_round = evidence([100.0, 200.0, 1.0, 3.0, 5.0], [9, 9, 9, 9, 9])
    summary = summarize_log_evidence(per_round, discard_rounds=2)
    assert summary.mean == pytest.approx(3.0)
    assert summary.scale == pytest.approx(2.0 / np.sqrt(3.0))
    assert summary.degrees_freedom == 2


@pytest.mark.parametrize("discard", [-1, 4, 5])
def test_summarize_rejects_invalid_discard_or_too_few_retained(discard):
    with pytest.raises(ValueError, match="nonnegative|At least two"):
        summarize_log_evidence(evidence([1.0, 2.0, 3.0, 4.0, 5.0]), discard)


def test_comparison_sign_and_independent_student_t_uncertainty():
    result = comparison_log_bayes_factor(
        "Comparison",
        EvidenceSummary(mean=-8.0, scale=3.0, degrees_freedom=30),
        EvidenceSummary(mean=-10.0, scale=4.0, degrees_freedom=30),
    )
    assert result.value == pytest.approx(2.0)
    assert result.uncertainty > 5.0


@pytest.mark.parametrize(
    ("value", "uncertainty", "expected"),
    [
        (1.2345, 0.1234, r"1.23 \pm 0.12"),
        (-2.345, 1.163, r"-2.3 \pm 1.2"),
        (1234.0, 123.0, r"1230 \pm 120"),
        (2.345, 0.0, r"2.3 \pm 0.0"),
    ],
)
def test_uncertainty_led_formatter(value, uncertainty, expected):
    assert format_value_uncertainty(value, uncertainty) == expected


def test_uncertainty_led_formatter_accepts_configurable_significant_figures():
    assert format_value_uncertainty(1.2345, 0.1234, 1) == r"1.2 \pm 0.1"
    assert format_value_uncertainty(1.2345, 0.1234, 3) == r"1.234 \pm 0.123"
    with pytest.raises(ValueError, match="positive integer"):
        format_value_uncertainty(1.0, 0.1, 0)


def test_uncertainty_led_formatter_can_use_decimal_places():
    assert format_value_uncertainty(
        1.2345, 12.34, 2, use_decimal_places=True
    ) == r"1.23 \pm 12.34"


def test_render_initial_table_exactly():
    assert render_log_bayes_table() == INITIAL_TABLE


def test_render_comparisons_preserves_order_and_trusted_latex():
    rendered = render_log_bayes_table(
        [
            BayesFactor(r"First $M_1$", 1.0, 0.1),
            BayesFactor(r"Second \textbf{model}", -2.0, 0.2),
        ]
    )
    assert rendered.index(r"First $M_1$") < rendered.index(r"Second \textbf{model}")
    assert r"$1.00 \pm 0.10$" in rendered
    assert r"$-2.00 \pm 0.20$" in rendered


def test_log_bayes_json_is_indexed_by_dataset_and_stable_model_key(tmp_path):
    output = tmp_path / "log-bayes.json"
    write_log_bayes_json(
        output,
        mid1={"notemp": BayesFactor("No temperature", -1.5, 0.2)},
        low3={"notemp": BayesFactor("No temperature", -3.0, 0.4)},
    )
    assert json.loads(output.read_text()) == {
        "low3": {
            "fiducial": {"ln_b": 0.0, "uncertainty": 0.0},
            "notemp": {"ln_b": -3.0, "uncertainty": 0.4},
        },
        "mid1": {
            "fiducial": {"ln_b": 0.0, "uncertainty": 0.0},
            "notemp": {"ln_b": -1.5, "uncertainty": 0.2},
        },
    }


def test_cli_defaults_and_comparison_validation(monkeypatch):
    module = runpy.run_path(str(SCRIPT))
    base = [
        str(SCRIPT),
        "--mid1-fiducial-artifact", "mid.zip",
        "--low3-fiducial-artifact", "low.zip",
    ]
    monkeypatch.setattr(sys, "argv", base)
    args = module["parse_args"]()
    assert args.discard_rounds == 4
    assert args.uncertainty_significant_figures == 2
    assert args.use_dp is False
    assert args.comparisons == ()
    assert args.mid1_output == Path("output/tablex-log-bayes-mid1.tex")
    assert args.low3_output == Path("output/tablex-log-bayes-low3.tex")

    with pytest.raises(ValueError, match="must not be blank"):
        module["validate_comparisons"]([["   ", "mid.zip", "low.zip"]])
    with pytest.raises(ValueError, match="Duplicate"):
        module["validate_comparisons"](
            [["same", "a.zip", "b.zip"], [" same ", "c.zip", "d.zip"]]
        )


def test_cli_dual_write_with_multiple_comparisons_in_order(tmp_path):
    mid_fid = write_evidence_artifact(tmp_path / "mid-fid.zip", evidence([0, 0, 1, 3]))
    low_fid = write_evidence_artifact(tmp_path / "low-fid.zip", evidence([0, 0, 10, 14]))
    mid_a = write_evidence_artifact(tmp_path / "mid-a.zip", evidence([0, 0, 4, 6]))
    low_a = write_evidence_artifact(tmp_path / "low-a.zip", evidence([0, 0, 7, 11]))
    mid_b = write_evidence_artifact(tmp_path / "mid-b.zip", evidence([0, 0, -2, 0]))
    low_b = write_evidence_artifact(tmp_path / "low-b.zip", evidence([0, 0, 15, 19]))
    mid_output = tmp_path / "nested" / "mid.tex"
    low_output = tmp_path / "other" / "low.tex"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mid1-fiducial-artifact", str(mid_fid),
            "--low3-fiducial-artifact", str(low_fid),
            "--discard-rounds", "2",
            "--mid1-output", str(mid_output),
            "--low3-output", str(low_output),
            "--comparison", "First model", str(mid_a), str(low_a),
            "--comparison", r"Second $M_2$", str(mid_b), str(low_b),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    mid_table = mid_output.read_text()
    low_table = low_output.read_text()
    for table in (mid_table, low_table):
        assert table.index("First model") < table.index(r"Second $M_2$")
        assert table.endswith("\\end{tabular}\n")
    assert r"$3.0 \pm 2.0$" in mid_table
    assert r"$-3.0 \pm 2.0$" in mid_table
    assert r"$-3.0 \pm 4.0$" in low_table
    assert r"$5.0 \pm 4.0$" in low_table


def unique_cached_fiducial(sample: str) -> Path:
    matches = tuple(
        Path("artifacts/cache").glob(f"racs_{sample}_fiducial.*.analysis.zip")
    )
    assert len(matches) == 1, f"Expected one cached {sample} fiducial, got {matches}"
    return matches[0]


def test_cached_fiducial_evidence_summaries_match_reference():
    mid1 = summarize_log_evidence(
        load_per_round_log_evidence(unique_cached_fiducial("mid1")), 4
    )
    low3 = summarize_log_evidence(
        load_per_round_log_evidence(unique_cached_fiducial("low3")), 4
    )
    assert mid1.mean == pytest.approx(-928.97136, abs=1e-5)
    assert mid1.uncertainty == pytest.approx(1.16361, abs=1e-5)
    assert low3.mean == pytest.approx(-960.08738, abs=1e-5)
    assert low3.uncertainty == pytest.approx(1.40643, abs=1e-5)
