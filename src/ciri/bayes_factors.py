"""Summarize repeated log evidences and render log Bayes-factor tables."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import t as student_t


FIDUCIAL_LABEL = (
    r"\textit{Fiducial:} Free dipole; temperature response; clustering"
)


@dataclass(frozen=True)
class EvidenceSummary:
    """Student-t posterior summary of the true log evidence."""

    mean: float
    scale: float
    degrees_freedom: int

    @property
    def uncertainty(self) -> float:
        """Central one-sigma posterior half-width for the true log evidence."""
        return student_t.ppf(0.841344746, self.degrees_freedom) * self.scale


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

    Treat retained rounds as independent Gaussian measurements of an unknown
    true log evidence with an unknown noise scale. Under the standard Jeffreys
    prior, the posterior of that true value is Student-t with ``n - 1`` degrees
    of freedom, location equal to the retained mean, and scale ``s / sqrt(n)``.
    The stored estimator errors are deliberately not combined into this model.
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
        scale=float(np.std(retained, ddof=1) / math.sqrt(retained.size)),
        degrees_freedom=retained.size - 1,
    )


def _difference_student_t_uncertainty(
    comparison: EvidenceSummary,
    fiducial: EvidenceSummary,
) -> float:
    """Return the central one-sigma half-width of two independent t draws."""
    target_probability = 0.841344746
    location = comparison.mean - fiducial.mean

    def cdf(value: float) -> float:
        standardized_difference = value - location

        def integrand(fiducial_standardized: float) -> float:
            return student_t.cdf(
                (standardized_difference + fiducial.scale * fiducial_standardized)
                / comparison.scale,
                comparison.degrees_freedom,
            ) * student_t.pdf(fiducial_standardized, fiducial.degrees_freedom)

        return quad(integrand, -np.inf, np.inf, epsabs=1e-9, epsrel=1e-9)[0]

    characteristic_scale = comparison.scale + fiducial.scale
    lower = location
    upper = location + characteristic_scale
    while cdf(upper) < target_probability:
        upper = location + 2.0 * (upper - location)
    quantile = brentq(
        lambda value: cdf(value) - target_probability,
        lower,
        upper,
        xtol=1e-10,
    )
    return quantile - location


def comparison_log_bayes_factor(
    label: str,
    comparison: EvidenceSummary,
    fiducial: EvidenceSummary,
    flag: str = "",
) -> BayesFactor:
    """Return a log Bayes factor and its central one-sigma posterior width."""
    return BayesFactor(
        label=label,
        value=comparison.mean - fiducial.mean,
        uncertainty=_difference_student_t_uncertainty(comparison, fiducial),
        flag=flag,
    )


def format_value_uncertainty(
    value: float,
    uncertainty: float,
    significant_figures: int = 2,
    *,
    use_decimal_places: bool = False,
) -> str:
    """Round an uncertainty and its value by significant figures or decimals."""
    if not math.isfinite(value) or not math.isfinite(uncertainty):
        raise ValueError("Bayes-factor value and uncertainty must be finite")
    if uncertainty < 0.0:
        raise ValueError("Bayes-factor uncertainty must be nonnegative")
    if isinstance(significant_figures, bool) or not isinstance(significant_figures, int):
        raise ValueError("Rounding precision must be a nonnegative integer")
    if significant_figures < 0 or (significant_figures == 0 and not use_decimal_places):
        raise ValueError("Rounding precision must be a positive integer")
    if uncertainty == 0.0:
        decimal_places = significant_figures if use_decimal_places else 1
        return rf"{value:.{decimal_places}f} \pm {uncertainty:.{decimal_places}f}"

    decimal_places = (
        significant_figures
        if use_decimal_places
        else significant_figures - 1 - math.floor(math.log10(uncertainty))
    )
    rounded_value = round(value, decimal_places)
    rounded_uncertainty = round(uncertainty, decimal_places)
    displayed_places = max(decimal_places, 0)
    return (
        f"{rounded_value:.{displayed_places}f}"
        rf" \pm {rounded_uncertainty:.{displayed_places}f}"
    )


def render_log_bayes_table(
    comparisons: Sequence[BayesFactor] = (),
    *,
    significant_figures: int = 2,
    use_decimal_places: bool = False,
) -> str:
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
            rf"${format_value_uncertainty(comparison.value, comparison.uncertainty, significant_figures, use_decimal_places=use_decimal_places)}$ \\"
            if comparison.flag
            else rf"{comparison.label} & "
            rf"${format_value_uncertainty(comparison.value, comparison.uncertainty, significant_figures, use_decimal_places=use_decimal_places)}$ \\",
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
