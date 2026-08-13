"""Build and display weighted directional posteriors on a HEALPix sky.

The functions in this module intentionally expose each numerical stage used by
the sky-map figure: sample extraction, coordinate conversion, HEALPix binning,
spherical smoothing, credible-level calculation, and Mollweide projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence, cast

from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d

from ciri.artifact_io import load_final_posterior
from ciri.constants import CMB_DIPOLE_EQUATORIAL_DEG, CMB_DIPOLE_GALACTIC_DEG


SkyFrame = Literal["galactic", "equatorial"]


class WeightedSamples(Protocol):
    """The subset of the anesthetic ``Samples`` interface used here."""

    def __getitem__(self, key: str) -> ArrayLike: ...

    def get_weights(self) -> ArrayLike: ...


@dataclass(frozen=True)
class FrameLabels:
    """Plot labels for longitude and latitude in a celestial frame."""

    longitude: str
    latitude: str
    name: str


FRAME_LABELS: dict[SkyFrame, FrameLabels] = {
    "galactic": FrameLabels(r"$\ell$", r"$b$", "Galactic"),
    "equatorial": FrameLabels(r"$\mathrm{RA}$", r"$\mathrm{Dec}$", "Equatorial"),
}

MOLLWEIDE_QUADRANTS = (1, 2, 3, 4)
MOLLWEIDE_LONGITUDE_TICKS = (150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210)
MOLLWEIDE_LATITUDE_TICKS = (-60, -30, 0, 30, 60)
CMB_MARKER_SIZE = 120.0
CMB_MARKER_ZORDER = 20
ADDITIONAL_POINT_MARKERS = ("o", "+", "s", "^", "D", "x")
ADDITIONAL_POINT_MARKER_SIZE = 7.0
ADDITIONAL_POINT_ZORDER = 19
SKYMAP_LEGEND_GAP = 0.02


@dataclass(frozen=True)
class SkyPoint:
    """A labelled direction with an explicitly declared celestial frame."""

    source_frame: SkyFrame
    longitude_deg: float
    latitude_deg: float
    label: str


def extract_weighted_directions(
    samples: WeightedSamples,
    *,
    longitude_column: str = "dipole_longitude",
    latitude_column: str = "dipole_latitude",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract finite Galactic directions and their positive nested weights.

    Longitudes and latitudes are expected in degrees. Invalid rows and weights
    that cannot contribute probability mass (zero or negative) are discarded;
    the remaining weights are returned unchanged and are normalized only when
    they are accumulated into a sky map.
    """
    longitude = np.asarray(samples[longitude_column], dtype=np.float64)
    latitude = np.asarray(samples[latitude_column], dtype=np.float64)
    weights = np.asarray(samples.get_weights(), dtype=np.float64)

    if longitude.ndim != 1 or latitude.ndim != 1 or weights.ndim != 1:
        raise ValueError("Sky coordinates and posterior weights must be vectors")
    if not (longitude.shape == latitude.shape == weights.shape):
        raise ValueError("Sky coordinates and posterior weights must have equal length")

    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(weights)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
        & (weights > 0.0)
    )
    if not np.any(valid):
        raise ValueError("Posterior contains no valid directions with positive weight")

    return longitude[valid], latitude[valid], weights[valid]


def convert_directions(
    longitude_deg: ArrayLike,
    latitude_deg: ArrayLike,
    source_frame: SkyFrame,
    output_frame: SkyFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert longitude/latitude arrays between Galactic and ICRS frames."""
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    if longitude.shape != latitude.shape:
        raise ValueError("Longitude and latitude must have the same shape")
    if source_frame not in FRAME_LABELS:
        raise ValueError(f"Unsupported source frame: {source_frame!r}")
    if output_frame not in FRAME_LABELS:
        raise ValueError(f"Unsupported output frame: {output_frame!r}")

    if source_frame == output_frame:
        return np.mod(longitude, 360.0), latitude.copy()

    if source_frame == "galactic":
        source = SkyCoord(l=longitude * u.deg, b=latitude * u.deg, frame="galactic")
        converted = source.icrs
        return (
            np.asarray(converted.ra.to_value(u.deg), dtype=np.float64),
            np.asarray(converted.dec.to_value(u.deg), dtype=np.float64),
        )

    source = SkyCoord(ra=longitude * u.deg, dec=latitude * u.deg, frame="icrs")
    converted = source.galactic
    return (
        np.asarray(converted.l.to_value(u.deg), dtype=np.float64),
        np.asarray(converted.b.to_value(u.deg), dtype=np.float64),
    )


def convert_galactic_directions(
    longitude_deg: ArrayLike,
    latitude_deg: ArrayLike,
    output_frame: SkyFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Galactic inputs in Galactic or equatorial (ICRS) coordinates."""
    return convert_directions(
        longitude_deg,
        latitude_deg,
        "galactic",
        output_frame,
    )


def cmb_dipole_direction(frame: SkyFrame) -> tuple[float, float]:
    """Return the canonical CMB dipole longitude and latitude in degrees."""
    if frame == "galactic":
        return CMB_DIPOLE_GALACTIC_DEG
    if frame == "equatorial":
        return CMB_DIPOLE_EQUATORIAL_DEG
    raise ValueError(f"Unsupported output frame: {frame!r}")


def parse_additional_points(
    values: Sequence[Sequence[object]] | None,
) -> tuple[SkyPoint, ...]:
    """Validate CLI-style ``(frame, longitude, latitude, label)`` points."""
    points: list[SkyPoint] = []
    for index, value in enumerate(() if values is None else values, start=1):
        if len(value) != 4:
            raise ValueError(
                f"Point {index} must contain frame, longitude, latitude, and label"
            )
        raw_frame, raw_longitude, raw_latitude, raw_label = value
        frame_name = str(raw_frame)
        if frame_name not in FRAME_LABELS:
            raise ValueError(
                f"Point {index} frame must be exactly 'galactic' or 'equatorial'"
            )
        source_frame = cast(SkyFrame, frame_name)
        try:
            longitude = float(raw_longitude)
            latitude = float(raw_latitude)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Point {index} longitude and latitude must be numeric"
            ) from exc
        if not np.isfinite(longitude) or not np.isfinite(latitude):
            raise ValueError(f"Point {index} longitude and latitude must be finite")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(
                f"Point {index} latitude must lie between -90 and 90 degrees"
            )
        label = str(raw_label)
        if not label.strip():
            raise ValueError(f"Point {index} label must not be empty")
        points.append(SkyPoint(source_frame, longitude % 360.0, latitude, label))
    return tuple(points)


def additional_point_direction(
    point: SkyPoint,
    output_frame: SkyFrame,
) -> tuple[float, float]:
    """Return a point in the requested display frame."""
    longitude, latitude = convert_directions(
        [point.longitude_deg],
        [point.latitude_deg],
        point.source_frame,
        output_frame,
    )
    return float(longitude[0]), float(latitude[0])


def astronomical_mollweide_coordinates(
    longitude_deg: ArrayLike,
    latitude_deg: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project sky degrees to Matplotlib's astronomical Mollweide convention.

    Matplotlib's positive x direction is opposite increasing astronomical
    longitude. Longitudes are therefore negated and wrapped onto ``[-pi, pi)``;
    latitude maps directly to y in radians.
    """
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    if longitude.shape != latitude.shape:
        raise ValueError("Longitude and latitude must have the same shape")
    if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
        raise ValueError("Longitude and latitude must be finite")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("Latitude must lie between -90 and 90 degrees")

    plot_longitude = np.mod(-longitude + 180.0, 360.0) - 180.0
    return (
        np.asarray(np.deg2rad(plot_longitude), dtype=np.float64),
        np.asarray(np.deg2rad(latitude), dtype=np.float64),
    )


def draw_cmb_dipole_marker(ax: Axes, frame: SkyFrame) -> PathCollection:
    """Draw the frame-appropriate CMB dipole as DipoleSBI's black star."""
    longitude, latitude = cmb_dipole_direction(frame)
    x, y = astronomical_mollweide_coordinates(longitude, latitude)
    return ax.scatter(
        float(x),
        float(y),
        marker="*",
        s=CMB_MARKER_SIZE,
        color="black",
        linewidths=0.0,
        zorder=CMB_MARKER_ZORDER,
        label="CMB",
    )


def cmb_legend_handle() -> Line2D:
    """Return a legend proxy matching the plotted CMB star marker."""
    return Line2D(
        [],
        [],
        linestyle="None",
        marker="*",
        markersize=np.sqrt(CMB_MARKER_SIZE),
        markerfacecolor="black",
        markeredgecolor="black",
        label="CMB",
    )


def additional_point_marker(index: int) -> str:
    """Return the deterministic non-star marker for a zero-based point index."""
    if index < 0:
        raise ValueError("Point marker index must be non-negative")
    return ADDITIONAL_POINT_MARKERS[index % len(ADDITIONAL_POINT_MARKERS)]


def _additional_point_line(
    marker: str,
    label: str,
    x: Sequence[float],
    y: Sequence[float],
) -> Line2D:
    return Line2D(
        x,
        y,
        linestyle="None",
        marker=marker,
        markersize=ADDITIONAL_POINT_MARKER_SIZE,
        color="black",
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=1.0,
        zorder=ADDITIONAL_POINT_ZORDER,
        label=label,
    )


def draw_additional_points(
    ax: Axes,
    points: Sequence[SkyPoint],
    output_frame: SkyFrame,
) -> tuple[Line2D, ...]:
    """Draw labelled point inputs after converting to the display frame."""
    artists = []
    for index, point in enumerate(points):
        longitude, latitude = additional_point_direction(point, output_frame)
        x, y = astronomical_mollweide_coordinates(longitude, latitude)
        artist = _additional_point_line(
            additional_point_marker(index),
            point.label,
            [float(x)],
            [float(y)],
        )
        ax.add_line(artist)
        artists.append(artist)
    return tuple(artists)


def additional_point_legend_handles(
    points: Sequence[SkyPoint],
) -> tuple[Line2D, ...]:
    """Return marker-matched legend proxies in point input order."""
    return tuple(
        _additional_point_line(
            additional_point_marker(index),
            point.label,
            [],
            [],
        )
        for index, point in enumerate(points)
    )


def place_skymap_legend(
    ax: Axes,
    handles: Sequence[object],
    selected_axes_bounds: tuple[float, float, float, float] | None = None,
    *,
    columns: int = 3,
) -> Legend:
    """Place a fixed-column sky-map legend wholly below the visible sky.

    ``selected_axes_bounds`` uses axes coordinates and is the value returned by
    :func:`layout_mollweide_quadrants`. Anchoring the legend's *upper* edge
    below that region lets added point entries grow downward over as many rows
    as needed without obscuring the map.
    """
    if columns < 1:
        raise ValueError("Legend column count must be positive")

    if selected_axes_bounds is None:
        x_center = 0.5
        y_min = 0.0
    else:
        x_min, x_max, y_min, _ = selected_axes_bounds
        x_center = (x_min + x_max) / 2.0

    return ax.legend(
        handles=handles,
        loc="upper center",
        ncol=columns,
        bbox_to_anchor=(x_center, y_min - SKYMAP_LEGEND_GAP),
        frameon=False,
    )


def bin_weighted_directions(
    longitude_deg: ArrayLike,
    latitude_deg: ArrayLike,
    weights: ArrayLike,
    *,
    nside: int,
) -> NDArray[np.float64]:
    """Accumulate posterior weights into a normalized RING-order HEALPix map."""
    if not hp.isnsideok(nside, nest=False):
        raise ValueError(f"Invalid HEALPix nside: {nside}")

    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if longitude.ndim != 1 or longitude.shape != latitude.shape:
        raise ValueError("Longitude and latitude must be equally sized vectors")
    if sample_weights.shape != longitude.shape:
        raise ValueError("Weights must have the same shape as the coordinates")
    if not (
        np.all(np.isfinite(longitude))
        and np.all(np.isfinite(latitude))
        and np.all(np.isfinite(sample_weights))
        and np.all((-90.0 <= latitude) & (latitude <= 90.0))
        and np.all(sample_weights >= 0.0)
    ):
        raise ValueError("Coordinates and weights passed to binning must be valid")

    total_weight = float(np.sum(sample_weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Weights must have a finite, positive sum")

    # lonlat=True makes the units/order explicit: (longitude, latitude), degrees.
    # nest=False makes the map ordering explicit: standard HEALPix RING ordering.
    pixel_indices = hp.ang2pix(
        nside,
        np.mod(longitude, 360.0),
        latitude,
        lonlat=True,
        nest=False,
    )
    probability = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    np.add.at(probability, pixel_indices, sample_weights)
    probability /= probability.sum()
    return probability


def smooth_probability_map(
    probability: ArrayLike,
    *,
    sigma_rad: float,
) -> NDArray[np.float64]:
    """Gaussian-smooth a RING map on the sphere, clip ringing, and renormalize."""
    sky_probability = np.asarray(probability, dtype=np.float64)
    if sky_probability.ndim != 1 or not hp.isnpixok(sky_probability.size):
        raise ValueError("Probability must be a one-dimensional HEALPix map")
    if not np.all(np.isfinite(sky_probability)) or np.any(sky_probability < 0.0):
        raise ValueError("Probability map must contain finite non-negative values")
    if not np.isfinite(sigma_rad) or sigma_rad < 0.0:
        raise ValueError("Smoothing sigma must be finite and non-negative")
    if sky_probability.sum() <= 0.0:
        raise ValueError("Probability map must have positive total mass")

    normalized = sky_probability / sky_probability.sum()
    if sigma_rad == 0.0:
        return normalized

    smoothed = np.asarray(
        hp.smoothing(normalized, sigma=sigma_rad),
        dtype=np.float64,
    )
    # A truncated harmonic reconstruction can create tiny negative sidelobes.
    smoothed = np.clip(smoothed, 0.0, None)
    total = float(smoothed.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Smoothing produced an invalid probability map")
    smoothed /= total
    return smoothed


def posterior_probability_map(
    artifact: str | Path,
    *,
    label: str,
    output_frame: SkyFrame,
    nside: int,
    smoothing_sigma: float,
) -> NDArray[np.float64]:
    """Load an artifact and turn its weighted directions into a sky map.

    This deliberately spells out the full reusable pipeline rather than hiding
    it behind plotting code: load the nested samples, retain their weights,
    transform the coordinates, bin into equal-area pixels, and smooth on the
    sphere.
    """
    nested_samples = load_final_posterior(artifact, label=label)
    galactic_longitude, galactic_latitude, weights = extract_weighted_directions(
        nested_samples
    )
    longitude, latitude = convert_galactic_directions(
        galactic_longitude,
        galactic_latitude,
        output_frame,
    )
    binned_probability = bin_weighted_directions(
        longitude,
        latitude,
        weights,
        nside=nside,
    )
    return smooth_probability_map(
        binned_probability,
        sigma_rad=smoothing_sigma,
    )


def gaussian_enclosed_probability(sigma: float) -> float:
    """Enclosed probability at ``sigma`` for an isotropic 2D Gaussian."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Sigma must be finite and positive")
    return float(1.0 - np.exp(-0.5 * sigma**2))


def credible_density_thresholds(
    probability: ArrayLike,
    enclosed_probabilities: ArrayLike,
) -> NDArray[np.float64]:
    """Find HPD density thresholds from equal-area HEALPix pixel masses.

    Every HEALPix pixel has the same solid angle, so sorting normalized pixel
    masses is equivalent to sorting probability densities. Each returned value
    is the lowest pixel mass included in the corresponding highest-posterior-
    density region.
    """
    sky_probability = np.asarray(probability, dtype=np.float64)
    enclosed = np.asarray(enclosed_probabilities, dtype=np.float64)
    if sky_probability.ndim != 1 or not hp.isnpixok(sky_probability.size):
        raise ValueError("Probability must be a one-dimensional HEALPix map")
    if not np.all(np.isfinite(sky_probability)) or np.any(sky_probability < 0.0):
        raise ValueError("Probability map must contain finite non-negative values")
    if enclosed.ndim != 1 or not np.all((enclosed > 0.0) & (enclosed <= 1.0)):
        raise ValueError("Enclosed probabilities must be a vector in (0, 1]")

    total = float(sky_probability.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Probability map must have positive total mass")
    ranked = np.sort(sky_probability / total)[::-1]
    cumulative = np.cumsum(ranked)
    # Advance by one representable float so exact decimal boundaries such as
    # 0.4 + 0.3 + 0.2 are not missed due only to round-off.
    indices = np.searchsorted(
        np.nextafter(cumulative, np.inf), enclosed, side="left"
    )
    indices = np.minimum(indices, ranked.size - 1)
    return ranked[indices]


def project_healpix_mollweide(
    probability: ArrayLike,
    *,
    xsize: int = 500,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Project and normalize a HEALPix map using Healpy's Mollweide grid.

    This is the full-sky projection path used by DipoleSBI's
    ``sky_probability``: Healpy supplies the longitude/latitude grids and the
    projected map, which is sanitized and normalized before contouring.
    """
    sky_probability = np.asarray(probability, dtype=np.float64)
    if sky_probability.ndim != 1 or not hp.isnpixok(sky_probability.size):
        raise ValueError("Probability must be a one-dimensional HEALPix map")
    if xsize < 2:
        raise ValueError("Projection xsize must be at least two")

    x_grid, y_grid, projected = hp.projview(
        sky_probability,
        return_only_data=True,
        xsize=xsize,
    )
    projected_probability = np.asarray(projected, dtype=np.float64)
    projected_probability = np.nan_to_num(
        projected_probability,
        nan=0.0,
        neginf=0.0,
        posinf=0.0,
    )
    projected_probability = np.clip(projected_probability, 0.0, None)
    total = float(projected_probability.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Mollweide projection contains no probability mass")
    projected_probability /= total
    return (
        np.asarray(x_grid, dtype=np.float64),
        np.asarray(y_grid, dtype=np.float64),
        projected_probability,
    )


def compute_2d_contours(
    projected_probability: ArrayLike,
    contour_sigmas: ArrayLike,
) -> NDArray[np.float64]:
    """Port DipoleSBI's projected-map ``compute_2D_contours`` semantics."""
    probability = np.asarray(projected_probability, dtype=np.float64)
    sigmas = np.asarray(contour_sigmas, dtype=np.float64)
    if probability.ndim != 2:
        raise ValueError("Projected probability must be a two-dimensional array")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("Projected probability must be finite and non-negative")
    if probability.sum() <= 0.0:
        raise ValueError("Projected probability must have positive mass")
    if sigmas.ndim != 1 or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Contour sigmas must be a vector of positive values")

    probability = probability / probability.sum()
    probability_levels = np.linspace(0.0, probability.max(), 1000)
    mask = probability >= probability_levels[:, None, None]
    probability_integral = (mask * probability).sum(axis=(1, 2))
    density_at_enclosed_probability = interp1d(
        probability_integral,
        probability_levels,
    )
    enclosed = np.array(
        [gaussian_enclosed_probability(float(sigma)) for sigma in sigmas]
    )
    return np.asarray(
        np.flip(density_at_enclosed_probability(enclosed)),
        dtype=np.float64,
    )


def _coordinate_tick_text(
    frame: SkyFrame,
    coordinate: Literal["longitude", "latitude"],
    value: int,
    *,
    annotated: bool,
) -> str:
    """Format a plain degree tick or one frame-aware coordinate annotation."""
    if not annotated:
        return rf"${value}^\circ$"
    symbol = (
        FRAME_LABELS[frame].longitude
        if coordinate == "longitude"
        else FRAME_LABELS[frame].latitude
    )
    return rf"${symbol.strip('$')} = {value}^\circ$"


def normalize_longitude_tick_removals(
    values: Sequence[float] | None,
) -> frozenset[int]:
    """Validate removed longitude labels, normalizing values modulo 360.

    Longitude zero is always suppressed. The tick itself and its graticule are
    retained; this function controls display text only.
    """
    removed = {0}
    for value in () if values is None else values:
        if not np.isfinite(value) or not np.isclose(value, round(value)):
            raise ValueError(
                f"Longitude tick removals must be whole degrees; got {value}"
            )
        normalized = int(round(value)) % 360
        if normalized not in MOLLWEIDE_LONGITUDE_TICKS:
            supported = ", ".join(str(tick) for tick in sorted(MOLLWEIDE_LONGITUDE_TICKS))
            raise ValueError(
                f"Unsupported longitude tick {value:g} degrees "
                f"(normalized to {normalized}); supported labels are {supported}"
            )
        removed.add(normalized)
    return frozenset(removed)


def normalize_latitude_tick_removals(
    values: Sequence[float] | None,
) -> frozenset[int]:
    """Validate latitude labels to suppress without altering graticules."""
    removed: set[int] = set()
    for value in () if values is None else values:
        if not np.isfinite(value) or not np.isclose(value, round(value)):
            raise ValueError(f"Latitude tick removals must be whole degrees; got {value}")
        normalized = int(round(value))
        if normalized not in MOLLWEIDE_LATITUDE_TICKS:
            supported = ", ".join(str(tick) for tick in MOLLWEIDE_LATITUDE_TICKS)
            raise ValueError(
                f"Unsupported latitude tick {value:g} degrees; "
                f"supported labels are {supported}"
            )
        removed.add(normalized)
    return frozenset(removed)


def preferred_coordinate_ticks(
    quadrants: Sequence[int] | None,
    *,
    frame: SkyFrame = "galactic",
    remove_longitude_ticks: Sequence[float] | None = None,
    remove_latitude_ticks: Sequence[float] | None = None,
) -> tuple[int | None, int | None]:
    """Choose visible, nonremoved coordinate-bearing tick annotations."""
    selected = normalize_quadrants(quadrants)
    selected_set = set(MOLLWEIDE_QUADRANTS if selected is None else selected)
    removed_longitude = normalize_longitude_tick_removals(remove_longitude_ticks)
    removed_latitude = normalize_latitude_tick_removals(remove_latitude_ticks)

    right_visible = bool(selected_set & {2, 4})
    left_visible = bool(selected_set & {1, 3})
    longitude_candidates: list[int] = []
    if frame == "equatorial":
        # Healpy-style Equatorial maps identify RA near the outer sides: 330
        # on the right and 30 on the left. Prefer right when both are visible,
        # then retain the established within-side fallbacks if both special
        # ticks are unavailable.
        if right_visible:
            longitude_candidates.append(330)
        if left_visible:
            longitude_candidates.append(30)
        if right_visible:
            longitude_candidates.extend((300, 270, 240, 210))
        if left_visible:
            longitude_candidates.extend((120, 90, 150, 60))
    elif frame == "galactic":
        if right_visible:
            longitude_candidates.extend((300, 270, 330, 240, 210))
        if left_visible:
            longitude_candidates.extend((120, 90, 150, 60, 30))
    else:
        raise ValueError(f"Unsupported output frame: {frame!r}")
    longitude = next(
        (value for value in longitude_candidates if value not in removed_longitude),
        None,
    )

    latitude_candidates: list[int] = []
    if selected_set & {1, 2}:
        latitude_candidates.extend((30, 60))
    if selected_set & {3, 4}:
        latitude_candidates.extend((-30, -60))
    latitude = next(
        (value for value in latitude_candidates if value not in removed_latitude),
        None,
    )
    return longitude, latitude


def configure_mollweide_axes(
    ax: Axes,
    frame: SkyFrame,
    quadrants: Sequence[int] | None = None,
    *,
    remove_longitude_ticks: Sequence[float] | None = None,
    remove_latitude_ticks: Sequence[float] | None = None,
) -> None:
    """Apply frame-aware ticks and explicit pole-to-pole dotted graticules."""
    ticks_deg = np.arange(-150, 180, 30)
    ax.set_xticks(np.deg2rad(ticks_deg))
    sky_longitudes = np.mod(-ticks_deg, 360)
    removed_longitude = normalize_longitude_tick_removals(remove_longitude_ticks)
    removed_latitude = normalize_latitude_tick_removals(remove_latitude_ticks)
    annotated_longitude, annotated_latitude = preferred_coordinate_ticks(
        quadrants,
        frame=frame,
        remove_longitude_ticks=removed_longitude,
        remove_latitude_ticks=removed_latitude,
    )
    ax.set_xticklabels(
        [
            ""
            if int(value) in removed_longitude
            else _coordinate_tick_text(
                frame,
                "longitude",
                int(value),
                annotated=value == annotated_longitude,
            )
            for value in sky_longitudes
        ]
    )
    latitude_ticks = np.array(MOLLWEIDE_LATITUDE_TICKS, dtype=int)
    ax.set_yticks(np.deg2rad(latitude_ticks))
    # The selected coordinate-bearing ticks identify both axes, matching the
    # DipoleSBI presentation without redundant standalone labels.
    ax.set_xlabel("")
    ax.set_ylabel("")
    _place_central_latitude_labels(
        ax,
        frame,
        quadrants,
        annotated_latitude=annotated_latitude,
        removed_latitude_ticks=removed_latitude,
    )
    ax._ciri_removed_longitude_ticks = removed_longitude  # type: ignore[attr-defined]
    ax._ciri_removed_latitude_ticks = removed_latitude  # type: ignore[attr-defined]
    ax._ciri_frame_label = add_mollweide_frame_label(ax, frame, quadrants)  # type: ignore[attr-defined]
    ax.grid(False)

    grid_style = {
        "linestyle": ":",
        "linewidth": 0.7,
        "color": "black",
        "alpha": 0.35,
        "zorder": 4,
    }
    latitude_curve = np.linspace(-np.pi / 2.0, np.pi / 2.0, 720)
    longitude_graticules = []
    for longitude in np.deg2rad(ticks_deg):
        (line,) = ax.plot(
            np.full_like(latitude_curve, longitude),
            latitude_curve,
            **grid_style,
        )
        longitude_graticules.append(line)

    longitude_curve = np.linspace(-np.pi, np.pi, 1440)
    latitude_graticules = []
    for latitude in ax.get_yticks():
        (line,) = ax.plot(
            longitude_curve,
            np.full_like(longitude_curve, latitude),
            **grid_style,
        )
        latitude_graticules.append(line)
    ax._ciri_longitude_graticules = tuple(longitude_graticules)  # type: ignore[attr-defined]
    ax._ciri_latitude_graticules = tuple(latitude_graticules)  # type: ignore[attr-defined]


def _place_central_latitude_labels(
    ax: Axes,
    frame: SkyFrame,
    quadrants: Sequence[int] | None,
    *,
    annotated_latitude: int | None,
    removed_latitude_ticks: frozenset[int],
) -> None:
    """Place one latitude-label set beside selected sides of longitude zero."""
    selected = normalize_quadrants(quadrants)
    selected_set = set(MOLLWEIDE_QUADRANTS if selected is None else selected)
    for label in ax.get_yticklabels():
        label.set_visible(False)

    latitude_labels = []
    for latitude in ax.get_yticks():
        latitude_degrees = int(round(np.rad2deg(latitude)))
        if latitude_degrees in removed_latitude_ticks:
            continue
        if latitude > 1e-12:
            left_selected = 1 in selected_set
            right_selected = 2 in selected_set
        elif latitude < -1e-12:
            left_selected = 3 in selected_set
            right_selected = 4 in selected_set
        else:
            left_selected = bool(selected_set & {1, 3})
            right_selected = bool(selected_set & {2, 4})
        if not left_selected and not right_selected:
            continue

        # Put labels on the exterior/opposite side of the selected region. A
        # right-hand quadrant therefore labels left of longitude zero, while a
        # left-hand quadrant labels right. If both are selected, prefer left.
        place_left_of_meridian = right_selected
        offset = -8.0 if place_left_of_meridian else 8.0
        rotation = 90.0 if place_left_of_meridian else -90.0
        latitude_labels.append(
            ax.annotate(
                _coordinate_tick_text(
                    frame,
                    "latitude",
                    latitude_degrees,
                    annotated=latitude_degrees == annotated_latitude,
                ),
                xy=(0.0, latitude),
                xytext=(offset, 0.0),
                textcoords="offset points",
                ha="center",
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                zorder=6,
                bbox={
                    "boxstyle": "square,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "linewidth": 0.0,
                    "alpha": 1.0,
                },
                clip_on=False,
                annotation_clip=False,
            )
        )
    ax._ciri_quadrant_latitude_labels = tuple(latitude_labels)  # type: ignore[attr-defined]


def add_mollweide_frame_label(
    ax: Axes,
    frame: SkyFrame,
    quadrants: Sequence[int] | None,
):
    """Add a Healpy-style horizontal frame label within a selected corner."""
    selected = normalize_quadrants(quadrants)
    selected_set = set(MOLLWEIDE_QUADRANTS if selected is None else selected)
    # Match Healpy newvisufunc's rlabel/llabel placement. Prefer a selected top
    # quadrant, then fall back to the corresponding bottom corner.
    if 2 in selected_set:
        position = (0.975, 0.925)
        horizontal_alignment = "right"
    elif 1 in selected_set:
        position = (0.025, 0.925)
        horizontal_alignment = "left"
    elif 4 in selected_set:
        position = (0.975, 0.075)
        horizontal_alignment = "right"
    else:
        position = (0.025, 0.075)
        horizontal_alignment = "left"
    return ax.text(
        *position,
        FRAME_LABELS[frame].name,
        transform=ax.transAxes,
        ha=horizontal_alignment,
        va="center",
        rotation=0.0,
        zorder=6,
        clip_on=False,
    )


def posterior_legend_patch(label: str, color: str) -> PathPatch:
    """Create a translucent run swatch with an opaque contour-like border."""
    return PathPatch(
        MplPath.unit_rectangle(),
        facecolor=to_rgba(color, alpha=0.4),
        edgecolor=to_rgba(color, alpha=1.0),
        linewidth=1.3,
        label=label,
    )


def posterior_outline_legend_patch(
    label: str,
    color: str,
    *,
    linestyle: str = "--",
) -> PathPatch:
    """Create an unfilled line-style swatch for a contour-only posterior."""
    return PathPatch(
        MplPath.unit_rectangle(),
        facecolor="none",
        edgecolor=to_rgba(color, alpha=1.0),
        linestyle=linestyle,
        linewidth=1.3,
        label=label,
    )


def normalize_quadrants(quadrants: Sequence[int] | None) -> tuple[int, ...] | None:
    """Validate and canonicalize Mollweide quadrant numbers."""
    if quadrants is None:
        return None
    selected = tuple(quadrants)
    if not selected:
        raise ValueError("At least one quadrant must be supplied")
    invalid = sorted({value for value in selected if value not in MOLLWEIDE_QUADRANTS})
    if invalid:
        values = ", ".join(str(value) for value in invalid)
        raise ValueError(f"Quadrants must be integers from 1 to 4; got {values}")
    if len(set(selected)) != len(selected):
        raise ValueError("Quadrant values must be unique")
    return tuple(sorted(selected))


def mollweide_quadrant_path(quadrant: int, *, points: int = 512) -> MplPath:
    """Trace one curved quadrant in longitude/latitude data coordinates."""
    selected = normalize_quadrants([quadrant])
    assert selected is not None
    if points < 2:
        raise ValueError("Quadrant boundary requires at least two points")

    horizontal_sign = -1.0 if quadrant in (1, 3) else 1.0
    vertical_sign = 1.0 if quadrant in (1, 2) else -1.0
    exterior_latitude = np.linspace(
        vertical_sign * np.pi / 2.0,
        0.0,
        points,
    )
    vertices = [
        (0.0, 0.0),
        (0.0, vertical_sign * np.pi / 2.0),
        *zip(
            np.full(points, horizontal_sign * np.pi),
            exterior_latitude,
            strict=True,
        ),
        (0.0, 0.0),
    ]
    codes = (
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(vertices) - 2)
        + [MplPath.CLOSEPOLY]
    )
    return MplPath(vertices, codes)


def mollweide_quadrant_bounds(
    quadrants: Sequence[int],
) -> tuple[float, float, float, float]:
    """Return the selected longitude/latitude bounds in radians."""
    selected = normalize_quadrants(quadrants)
    assert selected is not None
    x_min = 0.0 if not any(q in selected for q in (1, 3)) else -np.pi
    x_max = 0.0 if not any(q in selected for q in (2, 4)) else np.pi
    y_min = 0.0 if not any(q in selected for q in (3, 4)) else -np.pi / 2.0
    y_max = 0.0 if not any(q in selected for q in (1, 2)) else np.pi / 2.0
    return x_min, x_max, y_min, y_max


def _clip_mollweide_artists(ax: Axes, clip_patch: PathPatch) -> None:
    """Clip existing plot, graticule, and marker artists to a compound patch."""
    for artist in (*ax.collections, *ax.images, *ax.lines):
        artist.set_clip_path(clip_patch)
    for gridline in (*ax.get_xgridlines(), *ax.get_ygridlines()):
        gridline.set_clip_path(clip_patch)
    for patch in ax.patches:
        if patch is not clip_patch:
            patch.set_clip_path(clip_patch)


def _draw_quadrant_outline(ax: Axes, selected: tuple[int, ...]) -> None:
    """Draw only exposed union boundaries, without shared internal strokes."""
    boundary_style = {
        "color": "black",
        "linewidth": 0.8,
        "zorder": 5,
        "clip_on": False,
    }
    exterior_latitude = np.linspace(0.0, np.pi / 2.0, 512)
    for quadrant in selected:
        horizontal_sign = -1.0 if quadrant in (1, 3) else 1.0
        vertical_sign = 1.0 if quadrant in (1, 2) else -1.0
        ax.plot(
            np.full_like(exterior_latitude, horizontal_sign * np.pi),
            vertical_sign * exterior_latitude,
            **boundary_style,
        )

    selected_set = set(selected)
    for left, right, vertical_sign in ((1, 2, 1.0), (3, 4, -1.0)):
        if (left in selected_set) != (right in selected_set):
            ax.plot(
                [0.0, 0.0],
                [0.0, vertical_sign * np.pi / 2.0],
                **boundary_style,
            )
    for top, bottom, horizontal_sign in ((1, 3, -1.0), (2, 4, 1.0)):
        if (top in selected_set) != (bottom in selected_set):
            ax.plot(
                [0.0, horizontal_sign * np.pi],
                [0.0, 0.0],
                **boundary_style,
            )


def configure_mollweide_quadrants(
    ax: Axes,
    quadrants: Sequence[int],
) -> PathPatch | None:
    """Clip a Mollweide axes to any union of numbered sky quadrants.

    All four quadrants intentionally return ``None`` and leave the original
    full-sky axes unchanged.
    """
    selected = normalize_quadrants(quadrants)
    assert selected is not None
    if selected == MOLLWEIDE_QUADRANTS:
        return None

    compound_path = MplPath.make_compound_path(
        *(mollweide_quadrant_path(quadrant) for quadrant in selected)
    )
    clip_patch = PathPatch(
        compound_path,
        transform=ax.transData,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(clip_patch)
    _clip_mollweide_artists(ax, clip_patch)

    ax.patch.set_edgecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    _draw_quadrant_outline(ax, selected)
    return clip_patch


def layout_mollweide_quadrants(
    ax: Axes,
    quadrants: Sequence[int],
) -> tuple[float, float, float, float]:
    """Position labels for a selected union and hide out-of-region ticks."""
    selected = normalize_quadrants(quadrants)
    assert selected is not None
    x_min, x_max, y_min, y_max = mollweide_quadrant_bounds(selected)
    axes_x_min = (x_min + np.pi) / (2.0 * np.pi)
    axes_x_max = (x_max + np.pi) / (2.0 * np.pi)
    axes_y_min = (y_min + np.pi / 2.0) / np.pi
    axes_y_max = (y_max + np.pi / 2.0) / np.pi
    center_x = (axes_x_min + axes_x_max) / 2.0
    center_y = (axes_y_min + axes_y_max) / 2.0

    for position, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=True):
        label.set_visible(x_min - 1e-12 <= position <= x_max + 1e-12)

    ax.xaxis.set_label_coords(center_x, axes_y_min - 0.055)
    ax.yaxis.set_label_coords(axes_x_min - 0.045, center_y)
    return axes_x_min, axes_x_max, axes_y_min, axes_y_max


def selected_mollweide_bbox(
    fig,
    clip_patch: PathPatch,
    *,
    extra_artists: Sequence = (),
    pad_inches: float = 0.06,
) -> Bbox:
    """Return the tight display bbox around a selected curved sky union."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [clip_patch.get_window_extent(renderer)]
    for artist in extra_artists:
        if artist.get_visible():
            bbox = artist.get_window_extent(renderer)
            if bbox.width > 0.0 and bbox.height > 0.0:
                bboxes.append(bbox)
    display_bbox = Bbox.union(bboxes)
    inches_bbox = display_bbox.transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(
        inches_bbox.x0 - pad_inches,
        inches_bbox.y0 - pad_inches,
        inches_bbox.x1 + pad_inches,
        inches_bbox.y1 + pad_inches,
    )


def posterior_density_colormap(
    color: str,
    *,
    maximum_alpha: float = 0.4,
) -> LinearSegmentedColormap:
    """Build DipoleSBI's 512-step transparent-white posterior colormap."""
    if not np.isfinite(maximum_alpha) or not 0.0 < maximum_alpha <= 1.0:
        raise ValueError("Maximum alpha must be in (0, 1]")
    red, green, blue, _ = to_rgba(color)
    return LinearSegmentedColormap.from_list(
        "sky_probability",
        [
            (1.0, 1.0, 1.0, 0.0),
            (red, green, blue, maximum_alpha),
        ],
        512,
    )


def draw_mollweide_posterior(
    ax: Axes,
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    projected_probability: ArrayLike,
    credible_thresholds: ArrayLike,
    *,
    color: str,
    draw_mesh: bool,
    draw_fills: bool = True,
    contour_linestyle: str = "-",
    rasterize_mesh: bool = False,
):
    """Draw a posterior's mesh, HPD fills, and contour boundaries.

    Set ``draw_fills=False`` with a non-solid ``contour_linestyle`` to overlay
    a comparison posterior as contours only.
    """
    longitude = np.asarray(x_grid, dtype=np.float64)
    latitude = np.asarray(y_grid, dtype=np.float64)
    density = np.asarray(projected_probability, dtype=np.float64)
    thresholds = np.asarray(credible_thresholds, dtype=np.float64)
    if longitude.ndim != 1 or latitude.ndim != 1:
        raise ValueError("Projected longitude and latitude grids must be vectors")
    if density.shape != (latitude.size, longitude.size):
        raise ValueError("Projected probability shape must match its coordinate grids")
    if thresholds.ndim != 1 or thresholds.size < 2:
        raise ValueError("At least two credible thresholds are required")
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0.0):
        raise ValueError("Credible thresholds must be finite and non-negative")

    contour_levels = np.unique(np.sort(thresholds))
    x_mesh, y_mesh = np.meshgrid(longitude, latitude)
    mesh = None
    if draw_mesh:
        mesh = ax.pcolormesh(
            x_mesh,
            y_mesh,
            density,
            cmap=posterior_density_colormap(color),
            rasterized=rasterize_mesh,
        )
    # A single filled-contour call can combine regions touching -pi and +pi
    # into one compound polygon. On Matplotlib's Mollweide transform its close
    # path then cuts diagonally across the projection. Split at the continuous
    # central meridian so neither contour artist ever contains both sides of
    # the longitude discontinuity. The pcolormesh remains the faithful, full
    # Healpy projection and needs no split.
    central_index = int(np.argmin(np.abs(longitude)))
    longitude_slices = (
        slice(0, central_index + 1),
        slice(central_index, longitude.size),
    )
    filled_contours = []
    contours = []
    for longitude_slice in longitude_slices:
        if draw_fills:
            filled_contours.append(
                ax.contourf(
                    x_mesh[:, longitude_slice],
                    y_mesh[:, longitude_slice],
                    density[:, longitude_slice],
                    levels=contour_levels,
                    cmap=posterior_density_colormap(color),
                    zorder=1,
                    extend="both",
                )
            )
        contours.append(
            ax.contour(
                x_mesh[:, longitude_slice],
                y_mesh[:, longitude_slice],
                density[:, longitude_slice],
                levels=contour_levels,
                colors=[color],
                linestyles=contour_linestyle,
                zorder=1,
                extend="both",
            )
        )
    return mesh, tuple(filled_contours), tuple(contours)
