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
