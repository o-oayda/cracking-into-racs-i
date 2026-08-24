"""Plot LOW3 and MID1 directional posteriors on a two-dimensional sky map."""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from ciri.palette import MUTED_TEAL_DARK, RACS_SURVEY_COLOURS, ROSEWOOD
from ciri.plot_style import configure_plot_style
from ciri.skymap import (
    SkyFrame,
    ADDITIONAL_POINT_MARKER_SIZE,
    ADDITIONAL_POINT_ZORDER,
    CMB_MARKER_SIZE,
    CMB_MARKER_ZORDER,
    add_mollweide_frame_label,
    additional_point_direction,
    additional_point_legend_handles,
    additional_point_marker,
    cmb_dipole_direction,
    cmb_legend_handle,
    compute_2d_contours,
    credible_density_thresholds,
    configure_mollweide_axes,
    configure_mollweide_quadrants,
    draw_mollweide_posterior,
    draw_cmb_dipole_marker,
    draw_additional_points,
    gaussian_enclosed_probability,
    layout_mollweide_quadrants,
    normalize_annotated_longitude_tick,
    normalize_latitude_tick_removals,
    normalize_longitude_tick_removals,
    normalize_quadrants,
    parse_additional_points,
    place_skymap_legend,
    posterior_outline_legend_patch,
    posterior_legend_patch,
    posterior_density_colormap,
    posterior_probability_map,
    project_healpix_mollweide,
    selected_mollweide_bbox,
)
configure_plot_style()

POSTERIORS = (
    ("RACS-low3", RACS_SURVEY_COLOURS["low3"]),
    ("RACS-mid1", RACS_SURVEY_COLOURS["mid1"]),
)
NOTEMP_POSTERIORS = (
    ("No temp.", RACS_SURVEY_COLOURS["low3"], "--"),
    ("No temp.", RACS_SURVEY_COLOURS["mid1"], ":"),
)
SIGMAS = (1.0, 2.0)
NOTEMP_CONTOUR_ALPHA = 0.6
GLOBE_LONGITUDE_TICKS = tuple(range(0, 360, 30))
GLOBE_LABEL_LATITUDE_STEP_DEG = 1.0
GLOBE_LABEL_OFFSET_POINTS = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example mixed-frame overlays: --point galactic 264.021 48.253 "
            "'Galactic literature value' --point equatorial 167.942 -6.944 "
            "'Equatorial literature value'"
        ),
    )
    parser.add_argument("--low3-artifact", type=Path, required=True)
    parser.add_argument("--mid1-artifact", type=Path, required=True)
    parser.add_argument("--low3-notemp-artifact", type=Path)
    parser.add_argument("--mid1-notemp-artifact", type=Path)
    parser.add_argument(
        "--plot-notemp",
        action="store_true",
        help=(
            "Overlay no-temperature-response posterior contours. Requires "
            "--low3-notemp-artifact and --mid1-notemp-artifact."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=("mollweide", "globe"),
        default="mollweide",
        help=(
            "Sky projection to use: the existing Healpy-backed Mollweide map "
            "or a LIGO orthographic globe (default: mollweide)."
        ),
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=3,
        metavar="COUNT",
        help="Number of columns in the sky-map legend (default: 3).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH_IN", "HEIGHT_IN"),
        help=(
            "Output figure width and height in inches. Defaults to 8 by 4 for "
            "Mollweide or 6 by 6 for globe."
        ),
    )
    parser.add_argument(
        "--coordinates",
        choices=("galactic", "equatorial"),
        default="galactic",
        help="Coordinates shown on the output map (default: galactic).",
    )
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument(
        "--quadrants",
        type=int,
        nargs="+",
        help=(
            "Mollweide quadrants to show: 1=top-left, 2=top-right, "
            "3=bottom-left, 4=bottom-right. Omit for the full sky."
        ),
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=0.05,
        metavar="RADIANS",
        help="Spherical Gaussian smoothing sigma (default: 0.05 rad).",
    )
    parser.add_argument(
        "--label-contours",
        action="store_true",
        help=(
            "Label the solid posterior contours with their enclosed "
            "probability mass (for example, 39% and 86% for 1 and 2 sigma)."
        ),
    )
    parser.add_argument(
        "--contour-label-position-angle",
        type=float,
        metavar="DEGREES",
        help=(
            "Place each contour label at this position angle from its "
            "posterior peak, measured north through east. Requires "
            "--label-contours; by default Matplotlib chooses the placement."
        ),
    )
    parser.add_argument(
        "--remove-longitude-ticks",
        type=float,
        nargs="+",
        metavar="DEGREES",
        help=(
            "Hide one or more longitude tick labels. Values are normalized "
            "modulo 360; longitude 0 is always hidden."
        ),
    )
    parser.add_argument(
        "--remove-latitude-ticks",
        type=float,
        nargs="+",
        metavar="DEGREES",
        help="Hide one or more latitude tick labels.",
    )
    parser.add_argument(
        "--annotated-longitude-tick",
        type=float,
        metavar="DEGREES",
        help=(
            "Longitude tick that carries the frame annotation, such as "
            "'RA = 240 degrees'. Overrides automatic placement."
        ),
    )
    parser.add_argument(
        "--point",
        action="append",
        nargs=4,
        metavar=("FRAME", "LONGITUDE_DEG", "LATITUDE_DEG", "LABEL"),
        help=(
            "Overlay a labelled direction whose FRAME is exactly galactic or "
            "equatorial; repeat for multiple points and quote labels containing "
            "spaces."
        ),
    )
    parser.add_argument(
        "--rot",
        nargs=2,
        type=float,
        metavar=("LONGITUDE_DEG", "LATITUDE_DEG"),
        help=(
            "Projection center in the output coordinate frame. Mollweide only "
            "supports longitude rotation, so its LATITUDE_DEG must be 0."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figx-skymap.pdf"),
    )
    args = parser.parse_args()
    try:
        args.quadrants = normalize_quadrants(args.quadrants)
        args.remove_longitude_ticks = normalize_longitude_tick_removals(
            args.remove_longitude_ticks
        )
        args.remove_latitude_ticks = normalize_latitude_tick_removals(
            args.remove_latitude_ticks
        )
        if args.legend_columns < 1:
            raise ValueError("--legend-columns must be positive")
        if args.contour_label_position_angle is not None:
            if not math.isfinite(args.contour_label_position_angle):
                raise ValueError("Contour label position angle must be finite")
            if not args.label_contours:
                raise ValueError(
                    "--contour-label-position-angle requires --label-contours"
                )
            args.contour_label_position_angle %= 360.0
        if args.figsize is not None:
            width, height = args.figsize
            if (
                not math.isfinite(width)
                or not math.isfinite(height)
                or width <= 0.0
                or height <= 0.0
            ):
                raise ValueError("--figsize dimensions must be finite and positive")
            args.figsize = (width, height)
        args.points = parse_additional_points(args.point)
        if args.rot is not None:
            longitude, latitude = args.rot
            if not math.isfinite(longitude) or not math.isfinite(latitude):
                raise ValueError("Rotation longitude and latitude must be finite")
            if args.projection == "mollweide" and latitude != 0.0:
                raise ValueError(
                    "Only longitude rotations are supported; "
                    "rotation latitude must be 0"
                )
            args.rot = (longitude, latitude)
        if args.projection == "globe":
            if args.quadrants is not None:
                raise ValueError(
                    "--quadrants is only supported by Mollweide projection"
                )
            if args.annotated_longitude_tick is not None:
                raise ValueError(
                    "--annotated-longitude-tick is only supported by Mollweide "
                    "projection"
                )
        else:
            args.annotated_longitude_tick = normalize_annotated_longitude_tick(
                args.annotated_longitude_tick,
                rot=args.rot,
                quadrants=args.quadrants,
                remove_longitude_ticks=args.remove_longitude_ticks,
            )
    except ValueError as exc:
        parser.error(str(exc))
    if args.plot_notemp and (
        args.low3_notemp_artifact is None or args.mid1_notemp_artifact is None
    ):
        parser.error(
            "--plot-notemp requires --low3-notemp-artifact and "
            "--mid1-notemp-artifact"
        )
    return args


def ligo_globe_projection(frame: SkyFrame) -> str:
    """Return the degree-labelled LIGO globe projection for a sky frame."""
    return "galactic degrees globe" if frame == "galactic" else "astro degrees globe"


def register_ligo_projections() -> None:
    """Import LIGO's plotting module for its Matplotlib registration side effect."""
    importlib.import_module("ligo.skymap.plot")


def ligo_globe_center(
    frame: SkyFrame,
    rot: tuple[float, float] | None,
) -> SkyCoord | None:
    """Convert a CLI rotation into the matching LIGO projection centre."""
    if rot is None:
        return None
    longitude, latitude = rot
    if frame == "galactic":
        return SkyCoord(l=longitude * u.deg, b=latitude * u.deg, frame="galactic")
    return SkyCoord(ra=longitude * u.deg, dec=latitude * u.deg, frame="icrs")


def configure_ligo_globe_axes(
    ax,
    frame: SkyFrame,
    *,
    remove_longitude_ticks: frozenset[int],
    remove_latitude_ticks: frozenset[int],
) -> None:
    """Configure a LIGO globe with explicit degree ticks and frame labels."""
    longitude, latitude = ax.coords
    # LIGO's elliptical globe frame places longitude ticks on its horizontal
    # centreline. Make the physical latitude-zero gridline tickable instead,
    # so longitude ticks follow Dec = 0 degrees or b = 0 degrees.
    latitude.add_tickable_gridline("equator", 0 * u.deg)
    longitude.set_ticks_position(("equator",))
    longitude.set_ticklabel_position(("equator",))
    longitude.set_ticks(spacing=30 * u.deg)
    latitude.set_ticks(spacing=30 * u.deg)
    longitude.set_format_unit(u.deg, decimal=True)
    latitude.set_format_unit(u.deg, decimal=True)
    longitude.set_major_formatter(
        degree_tick_formatter(remove_longitude_ticks, wrap_longitude=True)
    )
    latitude.set_major_formatter(
        degree_tick_formatter(remove_latitude_ticks, wrap_longitude=False)
    )
    # Custom formatters may intentionally emit empty labels. Astropy's label
    # simplifier assumes non-empty strings and can fail while comparing them.
    longitude.set_ticklabel(simplify=False)
    latitude.set_ticklabel(simplify=False)
    # WCSAxes only supports one rotation for a complete coordinate's tick
    # labels. Retain its equatorial tick marks, but replace its longitude text
    # with labels whose rotations follow the local projected equator.
    longitude.set_ticklabel_visible(False)
    draw_ligo_globe_longitude_labels(ax, remove_longitude_ticks)
    # Globe axis titles are anchored to frame spines independently of the
    # custom tickable equator, so retain only the unambiguous degree labels.
    longitude.set_axislabel("")
    latitude.set_axislabel("")
    ax.grid(color="0.75", linestyle=":")
    ax._ciri_frame_label = add_mollweide_frame_label(ax, frame, None)


def _upright_rotation_from_local_tangent(tangent: np.ndarray) -> float:
    """Return an upright text rotation parallel to a projected curve."""
    dx, dy = np.asarray(tangent, dtype=float)
    rotation = math.degrees(math.atan2(dy, dx))
    return (rotation + 90.0) % 180.0 - 90.0


def _ligo_globe_longitude_label_geometry(
    ax,
    longitude: float,
) -> tuple[float, tuple[float, float], str] | None:
    """Project one equatorial tick and derive its rotation and inward offset."""
    transform = ax.get_transform("world")
    epsilon = GLOBE_LABEL_LATITUDE_STEP_DEG
    equator, west, east, north, south = transform.transform(
        [
            [longitude, 0.0],
            [longitude - epsilon, 0.0],
            [longitude + epsilon, 0.0],
            [longitude, epsilon],
            [longitude, -epsilon],
        ]
    )
    if not np.all(np.isfinite(equator)):
        return None

    west_is_visible = np.all(np.isfinite(west))
    east_is_visible = np.all(np.isfinite(east))
    if west_is_visible and east_is_visible:
        equator_tangent = east - west
    elif east_is_visible:
        equator_tangent = east - equator
    elif west_is_visible:
        equator_tangent = equator - west
    else:
        return None
    tangent_length = float(np.linalg.norm(equator_tangent))
    if not math.isfinite(tangent_length) or tangent_length == 0.0:
        return None
    tangent_unit = equator_tangent / tangent_length

    north_is_visible = np.all(np.isfinite(north))
    south_is_visible = np.all(np.isfinite(south))

    # Put labels south of the equator in the ordinary case. At the limb, use
    # whichever side of the equator remains inside the visible hemisphere.
    if south_is_visible:
        label_direction = south - equator
    else:
        label_direction = north - equator
    direction_length = float(np.linalg.norm(label_direction))
    if not math.isfinite(direction_length) or direction_length == 0.0:
        return None
    label_unit = label_direction / direction_length
    offset = tuple(GLOBE_LABEL_OFFSET_POINTS * label_unit)

    rotation = _upright_rotation_from_local_tangent(tangent_unit)
    rotation_radians = math.radians(rotation)
    text_up = np.array((-math.sin(rotation_radians), math.cos(rotation_radians)))
    vertical_alignment = "bottom" if np.dot(text_up, label_unit) > 0 else "top"
    return rotation, offset, vertical_alignment


def _update_ligo_globe_longitude_labels(ax) -> None:
    """Update custom longitude labels after projection or layout changes."""
    for longitude, label in ax._ciri_longitude_ticklabels.items():
        geometry = _ligo_globe_longitude_label_geometry(ax, longitude)
        if geometry is None:
            label.set_visible(False)
            continue
        rotation, offset, vertical_alignment = geometry
        label.set_visible(True)
        label.set_position(offset)
        label.set_rotation(rotation)
        label.set_verticalalignment(vertical_alignment)


def draw_ligo_globe_longitude_labels(
    ax,
    hidden_ticks: frozenset[int],
) -> tuple:
    """Draw individually rotated longitude labels on the physical equator."""
    old_callback = getattr(ax, "_ciri_longitude_ticklabel_callback", None)
    if old_callback is not None:
        ax.figure.canvas.mpl_disconnect(old_callback)
    for label in getattr(ax, "_ciri_longitude_ticklabels", {}).values():
        label.remove()

    world_transform = ax.get_transform("world")
    labels = {}
    for longitude in GLOBE_LONGITUDE_TICKS:
        if longitude in hidden_ticks:
            continue
        labels[longitude] = ax.annotate(
            f"{longitude}°",
            xy=(longitude, 0.0),
            xycoords=world_transform,
            xytext=(0.0, -GLOBE_LABEL_OFFSET_POINTS),
            textcoords="offset points",
            horizontalalignment="center",
            verticalalignment="top",
            rotation_mode="anchor",
            annotation_clip=True,
            clip_on=True,
            zorder=10,
        )
    ax._ciri_longitude_ticklabels = labels
    _update_ligo_globe_longitude_labels(ax)

    def update_after_draw(_event) -> None:
        _update_ligo_globe_longitude_labels(ax)

    ax._ciri_longitude_ticklabel_callback = ax.figure.canvas.mpl_connect(
        "draw_event", update_after_draw
    )
    return tuple(labels.values())


def degree_tick_formatter(hidden_ticks: frozenset[int], *, wrap_longitude: bool):
    """Build a WCSAxes formatter for integer-degree globe tick labels."""

    def formatter(values, spacing=None):
        degrees = np.asarray(values.to_value(u.deg))
        if wrap_longitude:
            degrees = np.mod(degrees, 360.0)
        labels = []
        for value in degrees:
            rounded = int(round(float(value)))
            if wrap_longitude:
                rounded %= 360
            labels.append("" if rounded in hidden_ticks else f"{rounded}°")
        return labels

    return formatter


def draw_ligo_globe_posterior(
    ax,
    probability: np.ndarray,
    thresholds: np.ndarray,
    *,
    color: str,
    draw_mesh: bool,
    draw_fills: bool = True,
    contour_linestyle: str = "-",
    contour_alpha: float = 1.0,
):
    """Draw one RING-order HEALPix posterior directly on a LIGO globe."""
    contour_levels = np.unique(np.sort(np.asarray(thresholds, dtype=float)))
    if contour_levels.size < 2:
        raise ValueError("At least two distinct credible thresholds are required")

    mesh = None
    if draw_mesh:
        mesh = ax.imshow_hpx(
            probability,
            nested=False,
            cmap=posterior_density_colormap(color),
            rasterized=True,
            zorder=0,
        )

    fills = None
    if draw_fills:
        fill_levels = np.concatenate(
            (contour_levels, [np.nextafter(float(np.max(probability)), np.inf)])
        )
        fills = ax.contourf_hpx(
            probability,
            nested=False,
            levels=fill_levels,
            cmap=posterior_density_colormap(color),
            zorder=1,
        )
    contours = ax.contour_hpx(
        probability,
        nested=False,
        levels=contour_levels,
        colors=[color],
        linestyles=contour_linestyle,
        alpha=contour_alpha,
        zorder=2,
    )
    return mesh, fills, contours


def label_contours_with_enclosed_probability(
    ax,
    contour_sets,
    thresholds: np.ndarray,
    sigmas: tuple[float, ...] = SIGMAS,
    position_angle: float | None = None,
    center: tuple[float, float] | None = None,
    center_transform=None,
    east_sign: float = 1.0,
    position_step: float = np.deg2rad(0.1),
) -> tuple:
    """Place inline credible-mass labels on one or more contour artists.

    ``thresholds`` are paired with ``sigmas`` before Matplotlib sorts contour
    levels, because a larger enclosed mass corresponds to a lower density
    threshold.  The labels therefore remain correct for either projection.

    When specified, ``position_angle`` selects a label location relative to
    ``center`` in the usual astronomical convention: 0 degrees points north
    and 90 degrees east. The text itself continues to follow the contour.
    """
    threshold_values = np.asarray(thresholds, dtype=float)
    if threshold_values.shape != (len(sigmas),):
        raise ValueError("Contour thresholds must correspond one-to-one with sigmas")
    formats = {
        # The shared plotting style uses LaTeX, where a literal percent sign
        # starts a comment and would disappear from the rendered label.
        float(threshold): f"{gaussian_enclosed_probability(sigma):.0%}".replace(
            "%", r"\%"
        )
        for threshold, sigma in zip(threshold_values, sigmas, strict=True)
    }
    if position_angle is not None:
        if not math.isfinite(position_angle):
            raise ValueError("Contour label position angle must be finite")
        if center is None or center_transform is None:
            raise ValueError("Contour label position requires a posterior peak")
        center_display = np.asarray(center_transform.transform(center), dtype=float)
        step = position_step
        north_display = np.asarray(
            center_transform.transform((center[0], center[1] + step)), dtype=float
        )
        east_display = np.asarray(
            center_transform.transform((center[0] + east_sign * step, center[1])),
            dtype=float,
        )
        north = north_display - center_display
        east = east_display - center_display
        north_norm = float(np.linalg.norm(north))
        east_norm = float(np.linalg.norm(east))
        if (
            not np.isfinite(center_display).all()
            or not math.isfinite(north_norm)
            or not math.isfinite(east_norm)
            or north_norm == 0.0
            or east_norm == 0.0
        ):
            raise ValueError("Unable to determine contour-label position angle")
        north /= north_norm
        east /= east_norm
        angle = np.deg2rad(position_angle)
        target = np.cos(angle) * north + np.sin(angle) * east
        labels = []
        for contours in contour_sets:
            for level, path in zip(
                contours.levels, contours.get_paths(), strict=True
            ):
                display_vertices = contours.get_transform().transform(path.vertices)
                offsets = display_vertices - center_display
                distances = np.linalg.norm(offsets, axis=1)
                valid = np.isfinite(distances) & (distances > 0.0)
                alignment = np.full(distances.shape, -np.inf)
                alignment[valid] = offsets[valid] @ target / distances[valid]
                vertex = display_vertices[int(np.argmax(alignment))]
                location = ax.transData.inverted().transform(vertex)
                existing_labels = tuple(getattr(contours, "labelTexts", ()))
                added_labels = ax.clabel(
                    contours,
                    levels=[level],
                    manual=[location],
                    inline=True,
                    inline_spacing=3,
                    fmt=formats,
                    fontsize="small",
                )
                labels.extend(added_labels[len(existing_labels) :])
        return tuple(labels)

    labels = []
    for contours in contour_sets:
        labels.extend(
            ax.clabel(
                contours,
                inline=True,
                inline_spacing=3,
                fmt=formats,
                fontsize="small",
            )
        )
    return tuple(labels)


def draw_ligo_globe_cmb_marker(ax, frame: SkyFrame):
    """Draw the CMB marker in WCS world coordinates."""
    longitude, latitude = cmb_dipole_direction(frame)
    return ax.scatter(
        longitude,
        latitude,
        transform=ax.get_transform("world"),
        marker="*",
        s=CMB_MARKER_SIZE,
        color="black",
        linewidths=0.0,
        zorder=CMB_MARKER_ZORDER,
        label="CMB",
    )


def draw_ligo_globe_additional_points(ax, points, output_frame: SkyFrame):
    """Draw mixed-frame CLI points in WCS world coordinates."""
    artists = []
    for index, point in enumerate(points):
        longitude, latitude = additional_point_direction(point, output_frame)
        (artist,) = ax.plot(
            longitude,
            latitude,
            transform=ax.get_transform("world"),
            linestyle="None",
            marker=additional_point_marker(index),
            markersize=ADDITIONAL_POINT_MARKER_SIZE,
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=ADDITIONAL_POINT_ZORDER,
            label=point.label,
        )
        artists.append(artist)
    return tuple(artists)


def render_ligo_globe(args, output_frame: SkyFrame) -> None:
    """Render an orthographic LIGO globe and save it to the requested path."""
    register_ligo_projections()
    subplot_kw = {"projection": ligo_globe_projection(output_frame)}
    center = ligo_globe_center(output_frame, args.rot)
    if center is not None:
        subplot_kw["center"] = center
    fig, ax = plt.subplots(
        figsize=args.figsize if args.figsize is not None else (6.0, 6.0),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )
    configure_ligo_globe_axes(
        ax,
        output_frame,
        remove_longitude_ticks=args.remove_longitude_ticks,
        remove_latitude_ticks=args.remove_latitude_ticks,
    )

    enclosed = np.array([gaussian_enclosed_probability(sigma) for sigma in SIGMAS])
    artifact_paths = (args.low3_artifact, args.mid1_artifact)
    for artifact, (label, color) in zip(artifact_paths, POSTERIORS, strict=True):
        probability = posterior_probability_map(
            artifact,
            label=label,
            output_frame=output_frame,
            nside=args.nside,
            smoothing_sigma=args.smoothing_sigma,
        )
        thresholds = credible_density_thresholds(probability, enclosed)
        _mesh, _fills, contours = draw_ligo_globe_posterior(
            ax,
            probability,
            thresholds,
            color=color,
            # Unlike the Mollweide compatibility view, render each globe
            # posterior's continuous density.  Otherwise the second
            # posterior is represented only by discrete HPD fill bands.
            draw_mesh=True,
        )
        if args.label_contours:
            peak_longitude, peak_latitude = hp.pix2ang(
                args.nside,
                int(np.argmax(probability)),
                lonlat=True,
            )
            label_contours_with_enclosed_probability(
                ax,
                (contours,),
                thresholds,
                position_angle=args.contour_label_position_angle,
                center=(float(peak_longitude), float(peak_latitude)),
                center_transform=ax.get_transform("world"),
                position_step=0.1,
            )

    if args.plot_notemp:
        notemp_artifact_paths = (
            args.low3_notemp_artifact,
            args.mid1_notemp_artifact,
        )
        for artifact, (label, color, linestyle) in zip(
            notemp_artifact_paths,
            NOTEMP_POSTERIORS,
            strict=True,
        ):
            probability = posterior_probability_map(
                artifact,
                label=label,
                output_frame=output_frame,
                nside=args.nside,
                smoothing_sigma=args.smoothing_sigma,
            )
            thresholds = credible_density_thresholds(probability, enclosed)
            draw_ligo_globe_posterior(
                ax,
                probability,
                thresholds,
                color=color,
                draw_mesh=False,
                draw_fills=False,
                contour_linestyle=linestyle,
                contour_alpha=NOTEMP_CONTOUR_ALPHA,
            )

    draw_ligo_globe_cmb_marker(ax, output_frame)
    draw_ligo_globe_additional_points(ax, args.points, output_frame)
    legend_handles = [
        posterior_legend_patch(label, color) for label, color in POSTERIORS
    ]
    if args.plot_notemp:
        legend_handles.extend(
            posterior_outline_legend_patch(
                label,
                color,
                linestyle=linestyle,
                alpha=NOTEMP_CONTOUR_ALPHA,
            )
            for label, color, linestyle in NOTEMP_POSTERIORS
        )
    legend_handles += [
        cmb_legend_handle(),
        *additional_point_legend_handles(args.points),
    ]
    place_skymap_legend(ax, legend_handles, columns=args.legend_columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_frame: SkyFrame = args.coordinates
    if args.projection == "globe":
        render_ligo_globe(args, output_frame)
        return
    artifact_paths = (args.low3_artifact, args.mid1_artifact)
    fig, ax = plt.subplots(
        figsize=args.figsize if args.figsize is not None else (8.0, 4.0),
        subplot_kw={"projection": "mollweide"},
        constrained_layout=True,
    )
    configure_mollweide_axes(
        ax,
        output_frame,
        args.quadrants,
        rot=args.rot,
        annotated_longitude_tick=args.annotated_longitude_tick,
        remove_longitude_ticks=args.remove_longitude_ticks,
        remove_latitude_ticks=args.remove_latitude_ticks,
    )

    ax.set_rasterization_zorder(0)
    for posterior_index, (artifact, (label, color)) in enumerate(
        zip(artifact_paths, POSTERIORS, strict=True)
    ):
        probability = posterior_probability_map(
            artifact,
            label=label,
            output_frame=output_frame,
            nside=args.nside,
            smoothing_sigma=args.smoothing_sigma,
        )
        x_grid, y_grid, projected_probability = project_healpix_mollweide(
            probability,
            rot=args.rot,
        )
        thresholds = compute_2d_contours(projected_probability, SIGMAS)

        _mesh, _fills, contours = draw_mollweide_posterior(
            ax,
            x_grid,
            y_grid,
            projected_probability,
            thresholds,
            color=color,
            # Match posterior_cli: only the first overlaid posterior supplies
            # the continuous probability mesh; every run supplies HPD fills.
            draw_mesh=posterior_index == 0,
        )
        if args.label_contours:
            peak_row, peak_column = np.unravel_index(
                int(np.argmax(projected_probability)),
                projected_probability.shape,
            )
            label_contours_with_enclosed_probability(
                ax,
                contours,
                thresholds,
                position_angle=args.contour_label_position_angle,
                center=(float(x_grid[peak_column]), float(y_grid[peak_row])),
                center_transform=ax.transData,
                # Mollweide longitudes increase to the left (east).
                east_sign=-1.0,
            )

    if args.plot_notemp:
        notemp_artifact_paths = (
            args.low3_notemp_artifact,
            args.mid1_notemp_artifact,
        )
        for artifact, (label, color, linestyle) in zip(
            notemp_artifact_paths,
            NOTEMP_POSTERIORS,
            strict=True,
        ):
            probability = posterior_probability_map(
                artifact,
                label=label,
                output_frame=output_frame,
                nside=args.nside,
                smoothing_sigma=args.smoothing_sigma,
            )
            x_grid, y_grid, projected_probability = project_healpix_mollweide(
                probability, rot=args.rot
            )
            thresholds = compute_2d_contours(projected_probability, SIGMAS)
            draw_mollweide_posterior(
                ax,
                x_grid,
                y_grid,
                projected_probability,
                thresholds,
                color=color,
                draw_mesh=False,
                draw_fills=False,
                contour_linestyle=linestyle,
                contour_alpha=NOTEMP_CONTOUR_ALPHA,
            )

    # Draw before quadrant setup so the same compound path clips the star.
    draw_cmb_dipole_marker(ax, output_frame, rot=args.rot)
    draw_additional_points(ax, args.points, output_frame, rot=args.rot)

    clip_patch = None
    selected_axes_bounds = None
    if args.quadrants is not None and len(args.quadrants) < 4:
        clip_patch = configure_mollweide_quadrants(ax, args.quadrants)
        selected_axes_bounds = layout_mollweide_quadrants(ax, args.quadrants)

    legend_handles = [
        posterior_legend_patch(label, color)
        for label, color in POSTERIORS
    ]
    if args.plot_notemp:
        legend_handles.extend(
            posterior_outline_legend_patch(
                label,
                color,
                linestyle=linestyle,
                alpha=NOTEMP_CONTOUR_ALPHA,
            )
            for label, color, linestyle in NOTEMP_POSTERIORS
        )
    legend_handles += [cmb_legend_handle(), *additional_point_legend_handles(args.points)]
    legend = place_skymap_legend(
        ax,
        legend_handles,
        selected_axes_bounds,
        columns=args.legend_columns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_bbox = "tight"
    if clip_patch is not None:
        visible_tick_labels = [
            label
            for label in (*ax.get_xticklabels(), *ax.get_yticklabels())
            if label.get_visible()
        ]
        quadrant_latitude_labels = getattr(
            ax,
            "_ciri_quadrant_latitude_labels",
            (),
        )
        frame_label = ax._ciri_frame_label
        output_bbox = selected_mollweide_bbox(
            fig,
            clip_patch,
            extra_artists=(
                legend,
                *visible_tick_labels,
                *quadrant_latitude_labels,
                frame_label,
            ),
        )
    fig.savefig(args.output, bbox_inches=output_bbox, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
