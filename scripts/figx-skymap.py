"""Plot LOW3 and MID1 directional posteriors on one Mollweide sky map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ciri.palette import MUTED_TEAL_DARK, ROSEWOOD
from ciri.plot_style import configure_plot_style
from ciri.skymap import (
    SkyFrame,
    additional_point_legend_handles,
    cmb_legend_handle,
    compute_2d_contours,
    configure_mollweide_axes,
    configure_mollweide_quadrants,
    draw_mollweide_posterior,
    draw_cmb_dipole_marker,
    draw_additional_points,
    layout_mollweide_quadrants,
    normalize_latitude_tick_removals,
    normalize_longitude_tick_removals,
    normalize_quadrants,
    parse_additional_points,
    place_skymap_legend,
    posterior_outline_legend_patch,
    posterior_legend_patch,
    posterior_probability_map,
    project_healpix_mollweide,
    selected_mollweide_bbox,
)
configure_plot_style()

POSTERIORS = (
    ("RACS-low3", MUTED_TEAL_DARK),
    ("RACS-mid1", ROSEWOOD),
)
NOTEMP_POSTERIORS = (
    ("No temp.", MUTED_TEAL_DARK, "--"),
    ("No temp.", ROSEWOOD, ":"),
)
SIGMAS = (1.0, 2.0)


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
        args.points = parse_additional_points(args.point)
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


def main() -> None:
    args = parse_args()
    output_frame: SkyFrame = args.coordinates
    artifact_paths = (args.low3_artifact, args.mid1_artifact)
    fig, ax = plt.subplots(
        figsize=(8.0, 4.0),
        subplot_kw={"projection": "mollweide"},
        constrained_layout=True,
    )
    configure_mollweide_axes(
        ax,
        output_frame,
        args.quadrants,
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
        x_grid, y_grid, projected_probability = project_healpix_mollweide(probability)
        thresholds = compute_2d_contours(projected_probability, SIGMAS)

        draw_mollweide_posterior(
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
                probability
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
            )

    # Draw before quadrant setup so the same compound path clips the star.
    draw_cmb_dipole_marker(ax, output_frame)
    draw_additional_points(ax, args.points, output_frame)

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
            posterior_outline_legend_patch(label, color, linestyle=linestyle)
            for label, color, linestyle in NOTEMP_POSTERIORS
        )
    legend_handles += [cmb_legend_handle(), *additional_point_legend_handles(args.points)]
    legend = place_skymap_legend(ax, legend_handles, selected_axes_bounds)
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
