"""Summarize repeated log evidences and render log Bayes-factor tables."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


FIDUCIAL_LABEL = (
    r"\textit{Fiducial:} Free dipole; temperature response; clustering"
)


@dataclass(frozen=True)
class EvidenceSummary:
    """Mean and round-to-round sample standard deviation of log evidence."""

    mean: float
    uncertainty: float


@dataclass(frozen=True)
class ComparisonArtifacts:
    """One comparison label and its matching MID1/LOW3 artifacts."""

    label: str
    mid1_artifact: Path
    low3_artifact: Path
    flag: str = ""


@dataclass(frozen=True)
class BayesFactor:
    """A comparison-minus-fiducial log Bayes factor."""

    label: str
    value: float
    uncertainty: float
    flag: str = ""


def summarize_log_evidence(
    evidence: ArrayLike,
    discard_rounds: int = 4,
) -> EvidenceSummary:
    """Discard positional rounds and summarize retained log evidences.

    Stored estimator errors are validated by the artifact loader but are
    deliberately not combined into this round-to-round uncertainty.
    """
    if isinstance(discard_rounds, bool) or not isinstance(discard_rounds, int):
        raise ValueError("discard_rounds must be a nonnegative integer")
    if discard_rounds < 0:
        raise ValueError("discard_rounds must be a nonnegative integer")

    array = np.asarray(evidence, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError("Per-round log evidence must have shape (2, n_rounds)")
    retained = array[0, discard_rounds:]
    if retained.size < 2:
        raise ValueError("At least two log-evidence rounds must remain after discarding")
    if not np.all(np.isfinite(retained)):
        raise ValueError("Retained log evidences must be finite")
    return EvidenceSummary(
        mean=float(np.mean(retained)),
        uncertainty=float(np.std(retained, ddof=1)),
    )


def comparison_log_bayes_factor(
    label: str,
    comparison: EvidenceSummary,
    fiducial: EvidenceSummary,
    flag: str = "",
) -> BayesFactor:
    """Return comparison-minus-fiducial log Bayes factor and uncertainty."""
    return BayesFactor(
        label=label,
        value=comparison.mean - fiducial.mean,
        uncertainty=math.hypot(comparison.uncertainty, fiducial.uncertainty),
        flag=flag,
    )


def format_value_uncertainty(value: float, uncertainty: float) -> str:
    """Format an uncertainty to two significant figures and match its value."""
    if not math.isfinite(value) or not math.isfinite(uncertainty):
        raise ValueError("Bayes-factor value and uncertainty must be finite")
    if uncertainty < 0.0:
        raise ValueError("Bayes-factor uncertainty must be nonnegative")
    if uncertainty == 0.0:
        return rf"{value:.1f} \pm {uncertainty:.1f}"

    exponent = math.floor(math.log10(uncertainty))
    decimal_places = 1 - exponent
    rounded_value = round(value, decimal_places)
    rounded_uncertainty = round(uncertainty, decimal_places)
    displayed_places = max(decimal_places, 0)
    return (
        f"{rounded_value:.{displayed_places}f}"
        rf" \pm {rounded_uncertainty:.{displayed_places}f}"
    )


def render_log_bayes_table(comparisons: Sequence[BayesFactor] = ()) -> str:
    """Render a booktabs table ordered by descending log Bayes factor."""
    lines = [
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"Model & $\ln \mathcal{B}$ \\",
        r"\midrule",
    ]
    rows = [(0.0, rf"{FIDUCIAL_LABEL} & \textemdash{{}} \\")]
    rows.extend(
        (
            comparison.value,
            rf"{comparison.label}$^{{{comparison.flag}}}$ & "
            rf"${format_value_uncertainty(comparison.value, comparison.uncertainty)}$ \\"
            if comparison.flag
            else rf"{comparison.label} & "
            rf"${format_value_uncertainty(comparison.value, comparison.uncertainty)}$ \\",
        )
        for comparison in comparisons
    )
    lines.extend(
        row for _, row in sorted(rows, key=lambda item: item[0], reverse=True)
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def write_log_bayes_json(
    output: Path,
    *,
    mid1: dict[str, BayesFactor],
    low3: dict[str, BayesFactor],
) -> None:
    """Write machine-readable log Bayes factors indexed by dataset and model."""
    payload = {
        "low3": {
            "fiducial": {"ln_b": 0.0, "uncertainty": 0.0},
            **{
                key: {"ln_b": factor.value, "uncertainty": factor.uncertainty}
                for key, factor in low3.items()
            },
        },
        "mid1": {
            "fiducial": {"ln_b": 0.0, "uncertainty": 0.0},
            **{
                key: {"ln_b": factor.value, "uncertainty": factor.uncertainty}
                for key, factor in mid1.items()
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
