"""Prepare weighted posterior samples and metadata for corner plots."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ciri.skymap import SkyFrame, convert_galactic_directions


class WeightedPosterior(Protocol):
    """Posterior columns and weights used by the corner preparation stage."""

    def __getitem__(self, key: str) -> ArrayLike: ...

    def get_weights(self) -> ArrayLike: ...


_BASE_PARAMETERS = (
    ("log10_n_initial_samples", r"\log_{10} N_{}"),
    ("observer_speed", r"\hat{v}_{o}"),
    ("dipole_longitude", ""),
    ("dipole_latitude", ""),
    ("temp_beta", r"\beta_{t}"),
    ("lambda_clus", r"\lambda_{c}"),
)

_DIRECTION_LABELS: dict[SkyFrame, tuple[str, str]] = {
    "galactic": (r"\ell^\circ", r"b^\circ"),
    "equatorial": (r"\mathrm{RA}^\circ", r"\mathrm{Dec}^\circ"),
}


def corner_parameters(frame: SkyFrame) -> tuple[tuple[str, str], ...]:
    """Return stable parameter names with frame-aware display labels."""
    try:
        longitude_label, latitude_label = _DIRECTION_LABELS[frame]
    except KeyError as exc:
        raise ValueError(f"Unsupported output frame: {frame!r}") from exc

    parameters = list(_BASE_PARAMETERS)
    parameters[2] = ("dipole_longitude", longitude_label)
    parameters[3] = ("dipole_latitude", latitude_label)
    return tuple(parameters)


def prepare_corner_samples(
    posterior: WeightedPosterior,
    output_frame: SkyFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Filter rows coherently and convert Galactic direction columns.

    Artifact directions are always Galactic degrees. Every retained row has
    finite parameter values, a valid Galactic latitude, and a finite
    positive nested weight. The retained nested weights are passed through
    unchanged and remain aligned row-for-row with every parameter column.
    """
    parameters = corner_parameters(output_frame)
    names = [name for name, _ in parameters]
    matrix = np.column_stack(
        [np.asarray(posterior[name], dtype=np.float64) for name in names]
    )
    weights = np.asarray(posterior.get_weights(), dtype=np.float64)

    if matrix.ndim != 2 or weights.ndim != 1 or matrix.shape[0] != weights.size:
        raise ValueError("Posterior columns and weights must have equal row counts")

    longitude_index = names.index("dipole_longitude")
    latitude_index = names.index("dipole_latitude")
    input_latitude = matrix[:, latitude_index]
    valid = (
        np.all(np.isfinite(matrix), axis=1)
        & np.isfinite(weights)
        & (weights > 0.0)
        & (input_latitude >= -90.0)
        & (input_latitude <= 90.0)
    )
    matrix = matrix[valid].copy()
    weights = weights[valid].copy()
    if matrix.size == 0 or not np.any(weights > 0.0):
        raise ValueError("Posterior contains no finite samples with positive weight")

    longitude, latitude = convert_galactic_directions(
        matrix[:, longitude_index],
        matrix[:, latitude_index],
        output_frame,
    )
    longitude = np.mod(longitude, 360.0)
    converted_valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= 0.0)
        & (longitude < 360.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )
    if not np.all(converted_valid):
        raise ValueError("Coordinate conversion produced invalid sky directions")

    matrix[:, longitude_index] = longitude
    matrix[:, latitude_index] = latitude
    return matrix, weights


def prepare_marginal_samples(
    posterior: WeightedPosterior,
    output_frame: SkyFrame,
) -> tuple[tuple[tuple[str, str], ...], NDArray[np.float64], NDArray[np.float64]]:
    """Return the available physical posterior parameters for marginal summaries.

    Comparison models fix some fiducial parameters, so absent parameters are
    intentionally omitted rather than represented by artificial uncertainties.
    """
    parameters = []
    columns = []
    for name, label in corner_parameters(output_frame):
        try:
            column = np.asarray(posterior[name], dtype=np.float64)
        except KeyError:
            continue
        parameters.append((name, label))
        columns.append(column)
    if not columns:
        raise ValueError("Posterior contains none of the supported parameters")

    matrix = np.column_stack(columns)
    weights = np.asarray(posterior.get_weights(), dtype=np.float64)
    if matrix.ndim != 2 or weights.ndim != 1 or matrix.shape[0] != weights.size:
        raise ValueError("Posterior columns and weights must have equal row counts")

    valid = np.all(np.isfinite(matrix), axis=1) & np.isfinite(weights) & (weights > 0.0)
    names = [name for name, _ in parameters]
    if "dipole_latitude" in names:
        latitude = matrix[:, names.index("dipole_latitude")]
        valid &= (latitude >= -90.0) & (latitude <= 90.0)
    matrix = matrix[valid].copy()
    weights = weights[valid].copy()
    if matrix.size == 0:
        raise ValueError("Posterior contains no finite samples with positive weight")

    if "dipole_longitude" in names:
        longitude_index = names.index("dipole_longitude")
        latitude_index = names.index("dipole_latitude")
        longitude, latitude = convert_galactic_directions(
            matrix[:, longitude_index], matrix[:, latitude_index], output_frame
        )
        matrix[:, longitude_index] = np.mod(longitude, 360.0)
        matrix[:, latitude_index] = latitude
    return tuple(parameters), matrix, weights


def weighted_quantiles(
    values: ArrayLike,
    weights: ArrayLike,
    quantiles: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the weighted quantiles used in the corner marginal titles."""
    data = np.asarray(values, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    qs = np.asarray(quantiles, dtype=np.float64)
    if data.ndim != 1 or sample_weights.size != data.size:
        raise ValueError("Values and weights must be equally sized vectors")
    mask = np.isfinite(data) & np.isfinite(sample_weights) & (sample_weights > 0.0)
    if not np.any(mask):
        raise ValueError("Weighted quantiles require at least one positive finite weight")
    data = data[mask]
    sample_weights = sample_weights[mask]
    sorter = np.argsort(data)
    data = data[sorter]
    sample_weights = sample_weights[sorter]
    cumulative = np.cumsum(sample_weights) - 0.5 * sample_weights
    cumulative /= cumulative[-1] + 0.5 * sample_weights[-1]
    return np.interp(qs, cumulative, data, left=data[0], right=data[-1])
