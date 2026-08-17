from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ciri.constants import CMB_DIPOLE_EQUATORIAL_DEG, CMB_DIPOLE_GALACTIC_DEG
from ciri.skymap import (
    ADDITIONAL_POINT_MARKERS,
    ADDITIONAL_POINT_MARKER_SIZE,
    ADDITIONAL_POINT_ZORDER,
    CMB_MARKER_SIZE,
    CMB_MARKER_ZORDER,
    FRAME_LABELS,
    SkyPoint,
    SKYMAP_LEGEND_GAP,
    additional_point_direction,
    additional_point_legend_handles,
    additional_point_marker,
    astronomical_mollweide_coordinates,
    bin_weighted_directions,
    cmb_dipole_direction,
    cmb_legend_handle,
    compute_2d_contours,
    configure_mollweide_axes,
    configure_mollweide_quadrants,
    convert_directions,
    convert_galactic_directions,
    credible_density_thresholds,
    draw_mollweide_posterior,
    draw_cmb_dipole_marker,
    draw_additional_points,
    extract_weighted_directions,
    gaussian_enclosed_probability,
    layout_mollweide_quadrants,
    mollweide_quadrant_bounds,
    mollweide_quadrant_path,
    normalize_annotated_longitude_tick,
    normalize_latitude_tick_removals,
    normalize_longitude_tick_removals,
    normalize_quadrants,
    posterior_legend_patch,
    posterior_density_colormap,
    parse_additional_points,
    place_skymap_legend,
    preferred_coordinate_ticks,
    project_healpix_mollweide,
    selected_mollweide_bbox,
    smooth_probability_map,
)


class FakeSamples:
    def __init__(self, longitude, latitude, weights):
        self.columns = {
            "dipole_longitude": np.asarray(longitude),
            "dipole_latitude": np.asarray(latitude),
        }
        self.weights = np.asarray(weights)

    def __getitem__(self, key):
        return self.columns[key]

    def get_weights(self):
        return self.weights


def test_cmb_dipole_constants_match_canonical_dipolesbi_values():
    assert CMB_DIPOLE_GALACTIC_DEG == (264.021, 48.253)
    assert CMB_DIPOLE_EQUATORIAL_DEG == (167.942, -6.94426)
    assert cmb_dipole_direction("galactic") == CMB_DIPOLE_GALACTIC_DEG
    assert cmb_dipole_direction("equatorial") == CMB_DIPOLE_EQUATORIAL_DEG


def test_astronomical_mollweide_coordinates_wrap_and_reverse_longitude():
    x, y = astronomical_mollweide_coordinates(
        [264.021, 167.942, 30.0, 330.0],
        [48.253, -6.94426, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        np.rad2deg(x),
        [95.979, -167.942, -30.0, 30.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.rad2deg(y),
        [48.253, -6.94426, 0.0, 0.0],
        atol=1e-12,
    )


def test_astronomical_mollweide_rotation_places_projection_center_at_origin():
    x, y = astronomical_mollweide_coordinates(
        180.0,
        30.0,
        rot=(180.0, 30.0),
    )

    assert float(x) == pytest.approx(0.0, abs=1e-12)
    assert float(y) == pytest.approx(0.0, abs=1e-12)


def test_rotated_mollweide_axes_label_original_sky_longitudes():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "equatorial", rot=(180.0, 0.0))
        labels = [label.get_text() for label in ax.get_xticklabels()]

        assert labels[0] == r"$\mathrm{RA} = 330^\circ$"
        assert labels[5] == ""
        assert labels[-1] == r"$30^\circ$"
    finally:
        plt.close(fig)


def test_explicit_longitude_annotation_overrides_automatic_placement():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(
            ax,
            "equatorial",
            [2, 4],
            rot=(270.0, 0.0),
            annotated_longitude_tick=240.0,
        )
        labels = [label.get_text() for label in ax.get_xticklabels()]

        assert sum(r"\mathrm{RA} =" in label for label in labels) == 1
        assert r"$\mathrm{RA} = 240^\circ$" in labels
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("value", "kwargs", "message"),
    [
        (240.5, {}, "whole number of degrees"),
        (250, {"rot": (270.0, 0.0)}, "not on the rotated tick grid"),
        (
            60,
            {"rot": (270.0, 0.0), "quadrants": [2, 4]},
            "outside the selected Mollweide quadrants",
        ),
        (
            240,
            {"rot": (270.0, 0.0), "remove_longitude_ticks": [240]},
            "hidden by --remove-longitude-ticks",
        ),
    ],
)
def test_explicit_longitude_annotation_rejects_invalid_ticks(value, kwargs, message):
    with pytest.raises(ValueError, match=message):
        normalize_annotated_longitude_tick(value, **kwargs)


@pytest.mark.parametrize(
    ("frame", "expected_quadrant"),
    [("galactic", 2), ("equatorial", 3)],
)
def test_cmb_projection_has_expected_quadrant_membership(frame, expected_quadrant):
    longitude, latitude = cmb_dipole_direction(frame)
    x, y = astronomical_mollweide_coordinates(longitude, latitude)
    horizontal = 1 if x < 0.0 else 2
    quadrant = horizontal if y > 0.0 else horizontal + 2
    assert quadrant == expected_quadrant


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
def test_cmb_marker_and_legend_proxy_match_reference_style(frame):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        marker = draw_cmb_dipole_marker(ax, frame)
        longitude, latitude = cmb_dipole_direction(frame)
        expected_x, expected_y = astronomical_mollweide_coordinates(
            longitude,
            latitude,
        )
        np.testing.assert_allclose(marker.get_offsets(), [[expected_x, expected_y]])
        np.testing.assert_allclose(marker.get_sizes(), [CMB_MARKER_SIZE])
        np.testing.assert_allclose(marker.get_facecolors(), [[0.0, 0.0, 0.0, 1.0]])
        assert marker.get_zorder() == CMB_MARKER_ZORDER
        assert marker.get_label() == "CMB"
        assert len(marker.get_paths()) == 1

        handle = cmb_legend_handle()
        assert handle.get_label() == "CMB"
        assert handle.get_marker() == "*"
        assert handle.get_linestyle() == "None"
        assert handle.get_markersize() == pytest.approx(np.sqrt(CMB_MARKER_SIZE))
        assert handle.get_markerfacecolor() == "black"
        assert handle.get_markeredgecolor() == "black"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("frame", "quadrant", "included"),
    [
        ("galactic", 1, False),
        ("galactic", 2, True),
        ("galactic", 3, False),
        ("galactic", 4, False),
        ("equatorial", 1, False),
        ("equatorial", 2, False),
        ("equatorial", 3, True),
        ("equatorial", 4, False),
    ],
)
def test_quadrant_clip_includes_cmb_marker_only_at_its_sky_position(
    frame,
    quadrant,
    included,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, frame, [quadrant])
        marker = draw_cmb_dipole_marker(ax, frame)
        clip_patch = configure_mollweide_quadrants(ax, [quadrant])
        assert clip_patch is not None
        assert marker.get_clip_path() is not None
        fig.canvas.draw()
        display_position = ax.transData.transform(marker.get_offsets()[0])
        assert clip_patch.contains_point(display_position) is included

        bbox = selected_mollweide_bbox(fig, clip_patch)
        assert np.all(np.isfinite(bbox.extents))
        assert bbox.width > 0.0
        assert bbox.height > 0.0
    finally:
        plt.close(fig)


def test_full_sky_cmb_marker_is_inside_the_full_axes_clip():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        marker = draw_cmb_dipole_marker(ax, "galactic")
        x, y = marker.get_offsets()[0]
        assert -np.pi < x < np.pi
        assert -np.pi / 2.0 < y < np.pi / 2.0
        fig.canvas.draw()
        display_position = ax.transData.transform((x, y))
        assert ax.patch.contains_point(display_position)
    finally:
        plt.close(fig)


def test_additional_point_parsing_repeats_preserves_labels_and_normalizes_longitude():
    points = parse_additional_points(
        [
            ["galactic", "-30", "12.5", "First point"],
            ["equatorial", "720", "-20", "Label with spaces"],
        ]
    )
    assert points == (
        SkyPoint("galactic", 330.0, 12.5, "First point"),
        SkyPoint("equatorial", 0.0, -20.0, "Label with spaces"),
    )
    assert parse_additional_points(None) == ()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([["ecliptic", "0", "0", "bad"]], "frame must be exactly"),
        ([["Galactic", "0", "0", "bad"]], "frame must be exactly"),
        ([["galactic", "not-a-number", "0", "bad"]], "must be numeric"),
        ([["equatorial", "nan", "0", "bad"]], "must be finite"),
        ([["galactic", "0", "91", "bad"]], "between -90 and 90"),
        ([["equatorial", "0", "0", ""]], "label must not be empty"),
        ([["galactic", "0", "0", "   "]], "label must not be empty"),
        ([["galactic", "0", "0"]], "must contain"),
    ],
)
def test_additional_point_parsing_rejects_invalid_data(values, message):
    with pytest.raises(ValueError, match=message):
        parse_additional_points(values)


def test_additional_point_conversion_reuses_galactic_frame_transform():
    galactic_centre = SkyPoint("galactic", 0.0, 0.0, "Galactic centre")
    assert additional_point_direction(galactic_centre, "galactic") == (0.0, 0.0)
    ra, dec = additional_point_direction(galactic_centre, "equatorial")
    assert ra == pytest.approx(266.40499, abs=2e-4)
    assert dec == pytest.approx(-28.93617, abs=2e-4)


@pytest.mark.parametrize("output_frame", ["galactic", "equatorial"])
def test_equivalent_mixed_frame_points_land_at_same_plotted_location(output_frame):
    points = (
        SkyPoint("galactic", 264.021, 48.253, "Galactic literature value"),
        SkyPoint("equatorial", 167.942, -6.94426, "Equatorial literature value"),
    )
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        artists = draw_additional_points(ax, points, output_frame)
        np.testing.assert_allclose(
            [artists[0].get_xdata()[0], artists[0].get_ydata()[0]],
            [artists[1].get_xdata()[0], artists[1].get_ydata()[0]],
            atol=np.deg2rad(2e-4),
        )
    finally:
        plt.close(fig)


def test_generic_direction_conversion_supports_equatorial_to_galactic_and_same_frame():
    longitude, latitude = convert_directions(
        [167.942],
        [-6.94426],
        "equatorial",
        "galactic",
    )
    np.testing.assert_allclose(longitude, [264.021], atol=2e-4)
    np.testing.assert_allclose(latitude, [48.253], atol=2e-4)

    longitude, latitude = convert_directions(
        [-30.0],
        [12.0],
        "equatorial",
        "equatorial",
    )
    np.testing.assert_array_equal(longitude, [330.0])
    np.testing.assert_array_equal(latitude, [12.0])


def test_additional_point_errors_identify_the_point_index():
    with pytest.raises(ValueError, match=r"Point 2 latitude"):
        parse_additional_points(
            [
                ["galactic", "0", "0", "Valid"],
                ["equatorial", "10", "-91", "Invalid"],
            ]
        )


def test_additional_point_marker_cycle_is_deterministic_and_excludes_star():
    markers = [
        additional_point_marker(index)
        for index in range(len(ADDITIONAL_POINT_MARKERS) + 2)
    ]
    assert markers[: len(ADDITIONAL_POINT_MARKERS)] == list(
        ADDITIONAL_POINT_MARKERS
    )
    assert markers[-2:] == list(ADDITIONAL_POINT_MARKERS[:2])
    assert "*" not in markers
    with pytest.raises(ValueError, match="non-negative"):
        additional_point_marker(-1)


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
def test_additional_point_artists_use_converted_positions_and_black_markers(frame):
    points = (
        SkyPoint("galactic", 30.0, 10.0, "Alpha"),
        SkyPoint("equatorial", 240.0, -20.0, "Beta"),
    )
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        artists = draw_additional_points(ax, points, frame)
        assert len(artists) == 2
        for index, (point, artist) in enumerate(zip(points, artists, strict=True)):
            longitude, latitude = additional_point_direction(point, frame)
            expected_x, expected_y = astronomical_mollweide_coordinates(
                longitude,
                latitude,
            )
            np.testing.assert_allclose(artist.get_xdata(), [expected_x])
            np.testing.assert_allclose(artist.get_ydata(), [expected_y])
            assert artist.get_marker() == additional_point_marker(index)
            assert artist.get_markersize() == ADDITIONAL_POINT_MARKER_SIZE
            assert artist.get_color() == "black"
            assert artist.get_linestyle() == "None"
            assert artist.get_zorder() == ADDITIONAL_POINT_ZORDER
            assert artist.get_label() == point.label
    finally:
        plt.close(fig)


def test_additional_points_share_quadrant_clip_and_keep_legend_entries():
    points = (
        SkyPoint("galactic", 264.0, 48.0, "Included q2"),
        SkyPoint("galactic", 120.0, -20.0, "Excluded q3"),
    )
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        artists = draw_additional_points(ax, points, "galactic")
        clip_patch = configure_mollweide_quadrants(ax, [2])
        assert clip_patch is not None
        fig.canvas.draw()
        included = []
        for artist in artists:
            assert artist.get_clip_path() is not None
            display_position = ax.transData.transform(
                (artist.get_xdata()[0], artist.get_ydata()[0])
            )
            included.append(clip_patch.contains_point(display_position))
        assert included == [True, False]

        handles = additional_point_legend_handles(points)
        assert [handle.get_label() for handle in handles] == [
            "Included q2",
            "Excluded q3",
        ]
        assert [handle.get_marker() for handle in handles] == ["o", "+"]
    finally:
        plt.close(fig)


def test_combined_legend_order_places_points_after_cmb():
    points = parse_additional_points(
        [["galactic", "10", "20", "One"], ["equatorial", "30", "40", "Two"]]
    )
    handles = [
        posterior_legend_patch("RACS-low3", "#000001"),
        posterior_legend_patch("RACS-mid1", "#000002"),
        cmb_legend_handle(),
        *additional_point_legend_handles(points),
    ]
    assert [handle.get_label() for handle in handles] == [
        "RACS-low3",
        "RACS-mid1",
        "CMB",
        "One",
        "Two",
    ]
    assert [handle.get_marker() for handle in handles[3:]] == ["o", "+"]


@pytest.mark.parametrize("point_count", [0, 1, 4, 7])
@pytest.mark.parametrize("quadrants", [None, [2]])
def test_skymap_legend_wraps_below_sky_and_is_included_in_bbox(
    point_count,
    quadrants,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "galactic", quadrants)
        clip_patch = None
        selected_bounds = None
        if quadrants is not None:
            clip_patch = configure_mollweide_quadrants(ax, quadrants)
            selected_bounds = layout_mollweide_quadrants(ax, quadrants)

        points = tuple(
            SkyPoint("galactic", float(index * 30), 20.0, f"Point {index}")
            for index in range(point_count)
        )
        handles = [
            posterior_legend_patch("RACS-low3", "#000001"),
            posterior_legend_patch("RACS-mid1", "#000002"),
            cmb_legend_handle(),
            *additional_point_legend_handles(points),
        ]
        legend = place_skymap_legend(ax, handles, selected_bounds)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer)
        selected_y_min = 0.0 if selected_bounds is None else selected_bounds[2]
        sky_bottom = ax.transAxes.transform((0.5, selected_y_min))[1]
        assert legend_bbox.y1 < sky_bottom
        assert legend._ncols == 3
        expected_x = 0.5 if selected_bounds is None else sum(selected_bounds[:2]) / 2
        expected_anchor = ax.transAxes.transform(
            (expected_x, selected_y_min - SKYMAP_LEGEND_GAP)
        )
        actual_anchor = legend.get_bbox_to_anchor().get_points()[0]
        np.testing.assert_allclose(actual_anchor, expected_anchor)

        if clip_patch is None:
            output_bbox = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
        else:
            output_bbox = selected_mollweide_bbox(
                fig,
                clip_patch,
                extra_artists=(legend,),
            ).transformed(fig.dpi_scale_trans)
        assert output_bbox.contains(legend_bbox.x0, legend_bbox.y0)
        assert output_bbox.contains(legend_bbox.x1, legend_bbox.y1)
    finally:
        plt.close(fig)


def test_cli_parses_repeated_points_and_quoted_label(monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--low3-artifact",
            "low3.zip",
            "--mid1-artifact",
            "mid1.zip",
            "--coordinates",
            "equatorial",
            "--point",
            "galactic",
            "-30",
            "12",
            "Local Group",
            "--point",
            "equatorial",
            "400",
            "-45",
            "Second",
        ],
    )
    args = module["parse_args"]()
    assert args.coordinates == "equatorial"
    assert args.projection == "mollweide"
    assert args.legend_columns == 3
    assert args.figsize is None
    assert args.points == (
        SkyPoint("galactic", 330.0, 12.0, "Local Group"),
        SkyPoint("equatorial", 40.0, -45.0, "Second"),
    )


def test_cli_reports_invalid_point_with_argparse_error():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--low3-artifact",
            "unused-low3.zip",
            "--mid1-artifact",
            "unused-mid1.zip",
            "--point",
            "equatorial",
            "10",
            "91",
            "Invalid latitude",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "latitude must lie between -90 and 90 degrees" in result.stderr


def test_cli_rejects_nonzero_rotation_latitude():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--low3-artifact",
            "unused-low3.zip",
            "--mid1-artifact",
            "unused-mid1.zip",
            "--rot",
            "150",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "rotation latitude must be 0" in result.stderr


def test_cli_globe_accepts_nonzero_rotation_latitude(monkeypatch):
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--low3-artifact",
            "low3.zip",
            "--mid1-artifact",
            "mid1.zip",
            "--projection",
            "globe",
            "--legend-columns",
            "4",
            "--figsize",
            "7",
            "5",
            "--rot",
            "150",
            "-25",
        ],
    )

    args = module["parse_args"]()

    assert args.projection == "globe"
    assert args.legend_columns == 4
    assert args.figsize == (7.0, 5.0)
    assert args.rot == (150.0, -25.0)


def test_cli_rejects_nonpositive_legend_columns():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--low3-artifact",
            "unused-low3.zip",
            "--mid1-artifact",
            "unused-mid1.zip",
            "--legend-columns",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--legend-columns must be positive" in result.stderr


def test_cli_rejects_nonpositive_figsize_dimension():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--low3-artifact",
            "unused-low3.zip",
            "--mid1-artifact",
            "unused-mid1.zip",
            "--figsize",
            "0",
            "6",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--figsize dimensions must be finite and positive" in result.stderr


def test_cli_rejects_contour_label_angle_without_contour_labels():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--low3-artifact",
            "unused-low3.zip",
            "--mid1-artifact",
            "unused-mid1.zip",
            "--contour-label-position-angle",
            "45",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--contour-label-position-angle requires --label-contours" in result.stderr


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--quadrants", "1"], "--quadrants is only supported"),
        (
            ["--annotated-longitude-tick", "30"],
            "--annotated-longitude-tick is only supported",
        ),
    ],
)
def test_cli_globe_rejects_mollweide_only_arguments(extra_args, message):
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--low3-artifact",
            "unused-low3.zip",
            "--mid1-artifact",
            "unused-mid1.zip",
            "--projection",
            "globe",
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        ("galactic", "galactic degrees globe"),
        ("equatorial", "astro degrees globe"),
    ],
)
def test_ligo_globe_projection_uses_degree_longitudes(frame, expected):
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))

    assert module["ligo_globe_projection"](frame) == expected


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
def test_ligo_globe_center_preserves_requested_frame_and_degrees(frame):
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))

    center = module["ligo_globe_center"](frame, (125.0, -35.0))

    if frame == "galactic":
        assert center.frame.name == "galactic"
        longitude = center.l.deg
        latitude = center.b.deg
    else:
        assert center.frame.name == "icrs"
        longitude = center.ra.deg
        latitude = center.dec.deg
    assert longitude == pytest.approx(125.0)
    assert latitude == pytest.approx(-35.0)


def test_ligo_globe_posterior_uses_direct_ring_map_and_expected_layers():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))
    probability = np.linspace(0.0, 1.0, hp.nside2npix(1))

    class FakeAxes:
        def __init__(self):
            self.calls = []

        def imshow_hpx(self, data, **kwargs):
            self.calls.append(("mesh", data, kwargs))
            return "mesh"

        def contourf_hpx(self, data, **kwargs):
            self.calls.append(("fills", data, kwargs))
            return "fills"

        def contour_hpx(self, data, **kwargs):
            self.calls.append(("contours", data, kwargs))
            return "contours"

    ax = FakeAxes()
    result = module["draw_ligo_globe_posterior"](
        ax,
        probability,
        np.array([0.7, 0.3]),
        color="#123456",
        draw_mesh=True,
    )

    assert result == ("mesh", "fills", "contours")
    assert [name for name, _data, _kwargs in ax.calls] == [
        "mesh",
        "fills",
        "contours",
    ]
    for _name, data, kwargs in ax.calls:
        assert data is probability
        assert kwargs["nested"] is False
    np.testing.assert_allclose(ax.calls[-1][2]["levels"], [0.3, 0.7])


def test_contour_labels_use_the_enclosed_probability_for_each_sigma():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))

    class FakeAxes:
        def __init__(self):
            self.calls = []

        def clabel(self, contours, **kwargs):
            self.calls.append((contours, kwargs))
            return [contours]

    ax = FakeAxes()
    labels = module["label_contours_with_enclosed_probability"](
        ax,
        ("western-half", "eastern-half"),
        np.array([0.7, 0.3]),
    )

    assert labels == ("western-half", "eastern-half")
    assert [contours for contours, _kwargs in ax.calls] == [
        "western-half",
        "eastern-half",
    ]
    for _contours, kwargs in ax.calls:
        assert kwargs["fmt"] == {0.7: r"39\%", 0.3: r"86\%"}
        assert kwargs["inline"] is True


def test_contour_label_position_angle_places_labels_relative_to_posterior_peak():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))
    coordinate = np.linspace(-2.0, 2.0, 101)
    x, y = np.meshgrid(coordinate, coordinate)
    density = np.exp(-0.5 * (x**2 + y**2))
    fig, ax = plt.subplots()
    try:
        contours = ax.contour(x, y, density, levels=[0.2, 0.5])
        labels = module["label_contours_with_enclosed_probability"](
            ax,
            (contours,),
            np.array([0.2, 0.5]),
            position_angle=0.0,
            center=(0.0, 0.0),
            center_transform=ax.transData,
        )

        assert len(labels) == 2
        assert all(label.get_position()[1] > 0.0 for label in labels)
    finally:
        plt.close(fig)


def test_ligo_globe_notemp_posterior_is_outline_only_with_requested_style():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))
    probability = np.linspace(0.0, 1.0, hp.nside2npix(1))

    class FakeAxes:
        def __init__(self):
            self.calls = []

        def contour_hpx(self, data, **kwargs):
            self.calls.append((data, kwargs))
            return "contours"

    ax = FakeAxes()
    result = module["draw_ligo_globe_posterior"](
        ax,
        probability,
        np.array([0.7, 0.3]),
        color="#123456",
        draw_mesh=False,
        draw_fills=False,
        contour_linestyle="--",
    )

    assert result == (None, None, "contours")
    assert len(ax.calls) == 1
    assert ax.calls[0][1]["linestyles"] == "--"


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
def test_real_ligo_globe_axes_render_with_degree_ticks(frame, tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))
    module["register_ligo_projections"]()
    fig, ax = plt.subplots(
        figsize=(3.0, 3.0),
        constrained_layout=True,
        subplot_kw={
            "projection": module["ligo_globe_projection"](frame),
            "center": module["ligo_globe_center"](frame, (150.0, -25.0)),
        },
    )
    try:
        module["configure_ligo_globe_axes"](
            ax,
            frame,
            remove_longitude_ticks=frozenset({0, 120}),
            remove_latitude_ticks=frozenset({-30}),
        )
        assert ax.coords[0].get_ticks_position() == ["equator"]
        assert ax.coords[0].get_ticklabel_position() == ["equator"]
        assert not ax.coords[0].get_ticklabel_visible()
        assert ax.coords[0].get_axislabel() == ""
        assert ax.coords[1].get_axislabel() == ""
        frame_label = ax._ciri_frame_label
        assert frame_label.get_text() == (
            "Galactic" if frame == "galactic" else "Equatorial"
        )
        assert frame_label.get_position() == (0.975, 0.925)
        assert frame_label.get_transform() is ax.transAxes
        fig.canvas.draw()

        custom_labels = ax._ciri_longitude_ticklabels
        assert 0 not in custom_labels
        assert 120 not in custom_labels
        visible_longitudes = {
            value for value, label in custom_labels.items() if label.get_visible()
        }
        assert visible_longitudes == {60, 90, 150, 180, 210, 240}
        assert all(label.xy[1] == 0.0 for label in custom_labels.values())

        center_rotation = (custom_labels[150].get_rotation() + 180.0) % 360.0 - 180.0
        assert center_rotation == pytest.approx(0.0, abs=1e-10)
        for value in visible_longitudes:
            rotation = (custom_labels[value].get_rotation() + 180.0) % 360.0 - 180.0
            assert -90.0 <= rotation <= 90.0

        world_transform = ax.get_transform("world")
        west, east = world_transform.transform([[89.0, 0.0], [91.0, 0.0]])
        dx, dy = east - west
        expected_rotation = np.degrees(np.arctan2(dy, dx))
        expected_rotation = (expected_rotation + 90.0) % 180.0 - 90.0
        actual_rotation = (custom_labels[90].get_rotation() + 180.0) % 360.0 - 180.0
        assert actual_rotation == pytest.approx(expected_rotation)

        output = tmp_path / f"{frame}-globe-ticks.png"
        fig.savefig(output, bbox_inches="tight", dpi=100)
        assert output.stat().st_size > 0

        longitude_formatter = module["degree_tick_formatter"](
            frozenset({0, 120}), wrap_longitude=True
        )
        latitude_formatter = module["degree_tick_formatter"](
            frozenset({-30}), wrap_longitude=False
        )
        assert longitude_formatter(np.array([0, 120, 150, 360]) * u.deg) == [
            "",
            "",
            "150°",
            "",
        ]
        assert latitude_formatter(np.array([-30, 0, 30]) * u.deg) == [
            "",
            "0°",
            "30°",
        ]
    finally:
        plt.close(fig)


def test_real_ligo_globe_renders_ring_healpix_contours():
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    module = runpy.run_path(str(script))
    module["register_ligo_projections"]()
    fig, ax = plt.subplots(
        figsize=(3.0, 3.0),
        subplot_kw={
            "projection": "galactic degrees globe",
            "center": module["ligo_globe_center"]("galactic", (150.0, -25.0)),
        },
    )
    try:
        longitude, latitude = hp.pix2ang(
            8,
            np.arange(hp.nside2npix(8)),
            lonlat=True,
        )
        separation = np.deg2rad(
            np.hypot((longitude - 150.0 + 180.0) % 360.0 - 180.0, latitude + 25.0)
        )
        probability = np.exp(-0.5 * np.square(separation / 0.2))
        probability /= probability.sum()
        thresholds = credible_density_thresholds(
            probability,
            [
                gaussian_enclosed_probability(1.0),
                gaussian_enclosed_probability(2.0),
            ],
        )

        mesh, fills, contours = module["draw_ligo_globe_posterior"](
            ax,
            probability,
            thresholds,
            color="#123456",
            draw_mesh=True,
        )
        labels = module["label_contours_with_enclosed_probability"](
            ax,
            (contours,),
            thresholds,
            position_angle=0.0,
            center=(150.0, -25.0),
            center_transform=ax.get_transform("world"),
            position_step=0.1,
        )
        fig.canvas.draw()

        assert mesh is not None
        assert fills is not None
        assert contours is not None
        assert {label.get_text() for label in labels} == {r"39\%", r"86\%"}
    finally:
        plt.close(fig)


def test_galactic_centre_converts_to_known_icrs_position():
    ra, dec = convert_galactic_directions([0.0], [0.0], "equatorial")
    np.testing.assert_allclose(ra, [266.40499], atol=2e-4)
    np.testing.assert_allclose(dec, [-28.93617], atol=2e-4)


def test_galactic_output_wraps_longitude_without_changing_latitude():
    longitude, latitude = convert_galactic_directions([-10.0], [12.0], "galactic")
    np.testing.assert_array_equal(longitude, [350.0])
    np.testing.assert_array_equal(latitude, [12.0])


def test_weighted_samples_accumulate_in_ring_pixels_and_normalize():
    longitude = np.array([10.0, 10.0, 200.0])
    latitude = np.array([20.0, 20.0, -30.0])
    probability = bin_weighted_directions(
        longitude,
        latitude,
        [1.0, 2.0, 1.0],
        nside=8,
    )
    pixels = hp.ang2pix(8, longitude, latitude, lonlat=True, nest=False)

    assert probability.sum() == pytest.approx(1.0)
    assert probability[pixels[0]] == pytest.approx(0.75)
    assert probability[pixels[2]] == pytest.approx(0.25)
    assert np.count_nonzero(probability) == 2


def test_smoothing_is_non_negative_and_renormalized():
    probability = np.zeros(hp.nside2npix(8))
    probability[123] = 1.0
    smoothed = smooth_probability_map(probability, sigma_rad=0.15)

    assert np.all(np.isfinite(smoothed))
    assert np.all(smoothed >= 0.0)
    assert smoothed.sum() == pytest.approx(1.0)
    assert np.count_nonzero(smoothed) > 1


def test_extraction_filters_invalid_rows_without_reweighting():
    samples = FakeSamples(
        [10.0, np.nan, 30.0, 40.0, 50.0],
        [20.0, 30.0, 100.0, 50.0, 60.0],
        [0.25, 0.5, 0.75, 0.0, -1.0],
    )
    longitude, latitude, weights = extract_weighted_directions(samples)

    np.testing.assert_array_equal(longitude, [10.0])
    np.testing.assert_array_equal(latitude, [20.0])
    np.testing.assert_array_equal(weights, [0.25])


def test_extraction_rejects_posterior_with_no_positive_valid_weight():
    samples = FakeSamples([10.0], [20.0], [0.0])
    with pytest.raises(ValueError, match="no valid directions"):
        extract_weighted_directions(samples)


def test_credible_thresholds_use_ranked_equal_area_pixel_mass():
    probability = np.zeros(hp.nside2npix(1))
    probability[:4] = [0.4, 0.3, 0.2, 0.1]
    thresholds = credible_density_thresholds(probability, [0.4, 0.7, 0.9])
    np.testing.assert_allclose(thresholds, [0.4, 0.3, 0.2])


def test_two_dimensional_gaussian_sigma_probabilities():
    assert gaussian_enclosed_probability(1.0) == pytest.approx(0.39346934)
    assert gaussian_enclosed_probability(2.0) == pytest.approx(0.86466472)


def test_frame_labels_match_the_displayed_coordinates():
    assert FRAME_LABELS["galactic"].longitude == r"$\ell$"
    assert FRAME_LABELS["galactic"].latitude == r"$b$"
    assert FRAME_LABELS["equatorial"].longitude == r"$\mathrm{RA}$"
    assert FRAME_LABELS["equatorial"].latitude == r"$\mathrm{Dec}$"


@pytest.mark.parametrize(
    ("frame", "expected_longitude", "expected_latitude"),
    [
        ("galactic", r"$\ell$", r"$b$"),
        ("equatorial", r"$\mathrm{RA}$", r"$\mathrm{Dec}$"),
    ],
)
def test_mollweide_axes_render_frame_specific_symbols(
    frame,
    expected_longitude,
    expected_latitude,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, frame)
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        assert any(
            expected_longitude.strip("$") + " =" in label.get_text()
            for label in ax.get_xticklabels()
        )
        assert any(
            expected_latitude.strip("$") + " =" in label.get_text()
            for label in ax._ciri_quadrant_latitude_labels
        )
    finally:
        plt.close(fig)


def test_healpy_mollweide_projection_has_astronomical_handedness_and_wraps_seam():
    nside = 8
    probability = np.zeros(hp.nside2npix(nside))
    left_pixel = hp.ang2pix(nside, 60.0, 0.0, lonlat=True, nest=False)
    right_pixel = hp.ang2pix(nside, 300.0, 0.0, lonlat=True, nest=False)
    seam_left_pixel = hp.ang2pix(nside, 175.0, 0.0, lonlat=True, nest=False)
    seam_right_pixel = hp.ang2pix(nside, 185.0, 0.0, lonlat=True, nest=False)
    probability[left_pixel] = 0.25
    probability[right_pixel] = 0.50
    probability[seam_left_pixel] = 0.75
    probability[seam_right_pixel] = 1.00

    x_grid, y_grid, projected = project_healpix_mollweide(
        probability,
        xsize=200,
    )
    equator = np.argmin(np.abs(y_grid))
    left = np.argmin(np.abs(x_grid - np.deg2rad(-60.0)))
    right = np.argmin(np.abs(x_grid - np.deg2rad(60.0)))

    assert projected[equator, left] > 0.0
    assert projected[equator, right] > projected[equator, left]
    assert np.any(projected[equator, x_grid < np.deg2rad(-165.0)] > 0.0)
    assert np.any(projected[equator, x_grid > np.deg2rad(165.0)] > 0.0)


def test_rotated_marker_aligns_with_rotated_healpix_projection():
    nside = 32
    center = (180.0, 30.0)
    probability = np.zeros(hp.nside2npix(nside))
    probability[hp.ang2pix(nside, *center, lonlat=True, nest=False)] = 1.0

    x_grid, y_grid, projected = project_healpix_mollweide(
        probability,
        xsize=400,
        rot=center,
    )
    row, column = np.unravel_index(np.argmax(projected), projected.shape)
    marker_x, marker_y = astronomical_mollweide_coordinates(*center, rot=center)

    assert x_grid[column] == pytest.approx(float(marker_x), abs=np.deg2rad(2.0))
    assert y_grid[row] == pytest.approx(float(marker_y), abs=np.deg2rad(2.0))


def test_posterior_colormap_fades_from_transparent_white_to_run_color():
    colormap = posterior_density_colormap("#ca054d", maximum_alpha=0.4)
    np.testing.assert_allclose(colormap(0.0), [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(colormap(1.0), [202 / 255, 5 / 255, 77 / 255, 0.4])


def test_projected_contours_match_dipolesbi_probability_semantics():
    coordinate = np.linspace(-3.0, 3.0, 51)
    x, y = np.meshgrid(coordinate, coordinate)
    density = np.exp(-0.5 * (x**2 + y**2))
    thresholds = compute_2d_contours(density, [1.0, 2.0])

    assert thresholds.shape == (2,)
    assert thresholds[0] < thresholds[1]
    assert 0.0 <= thresholds[0] <= density.max()
    assert 0.0 <= thresholds[1] <= density.max()


def test_draw_posterior_uses_full_unrasterized_mesh_and_solid_contours():
    plot_longitude = np.linspace(-np.pi, np.pi, 21)
    plot_latitude = np.linspace(-np.pi / 2, np.pi / 2, 11)
    x, y = np.meshgrid(plot_longitude, plot_latitude)
    density = np.exp(-5.0 * (x**2 + y**2))

    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        mesh, filled_contours, contours = draw_mollweide_posterior(
            ax,
            plot_longitude,
            plot_latitude,
            density,
            [0.5, 0.2],
            color="#ca054d",
            draw_mesh=True,
        )
        assert mesh is not None
        assert mesh.get_rasterized() is False
        assert not np.ma.is_masked(mesh.get_array())
        assert mesh.cmap(0.0)[-1] == 0.0
        assert mesh.cmap(1.0)[-1] == pytest.approx(0.4)
        assert all(filled.cmap(0.0)[-1] == 0.0 for filled in filled_contours)
        assert all(
            dashes is None
            for contour in contours
            for _, dashes in contour.get_linestyles()
        )
    finally:
        plt.close(fig)


def test_seam_crossing_contours_never_create_full_width_compound_paths():
    plot_longitude = np.linspace(np.pi, -np.pi, 101)
    plot_latitude = np.linspace(-np.pi / 2, np.pi / 2, 51)
    x, y = np.meshgrid(plot_longitude, plot_latitude)
    distance_from_seam = np.pi - np.abs(x)
    density = np.exp(-20.0 * (distance_from_seam**2 + y**2))

    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        _, filled_contours, contours = draw_mollweide_posterior(
            ax,
            plot_longitude,
            plot_latitude,
            density,
            [0.2, 0.5],
            color="#ca054d",
            draw_mesh=False,
        )
        for contour_set in (*filled_contours, *contours):
            for path in contour_set.get_paths():
                if path.vertices.size:
                    assert np.ptp(path.vertices[:, 0]) <= np.pi + 1e-12
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("quadrants", "expected"),
    [
        ([2], (0.0, np.pi, 0.0, np.pi / 2.0)),
        ([1, 2], (-np.pi, np.pi, 0.0, np.pi / 2.0)),
        ([1, 3], (-np.pi, 0.0, -np.pi / 2.0, np.pi / 2.0)),
        ([1, 4], (-np.pi, np.pi, -np.pi / 2.0, np.pi / 2.0)),
        ([1, 2, 3, 4], (-np.pi, np.pi, -np.pi / 2.0, np.pi / 2.0)),
    ],
)
def test_quadrant_bounds_cover_the_smallest_selected_coordinate_extent(
    quadrants,
    expected,
):
    np.testing.assert_allclose(mollweide_quadrant_bounds(quadrants), expected)


def test_quadrants_are_sorted_and_invalid_or_duplicate_values_are_rejected():
    assert normalize_quadrants([4, 1]) == (1, 4)
    assert normalize_quadrants(None) is None
    with pytest.raises(ValueError, match="from 1 to 4"):
        normalize_quadrants([0, 2])
    with pytest.raises(ValueError, match="unique"):
        normalize_quadrants([2, 2])


@pytest.mark.parametrize("quadrant", [1, 2, 3, 4])
def test_single_quadrant_paths_have_curved_exterior_and_expected_bounds(quadrant):
    path = mollweide_quadrant_path(quadrant, points=32)
    assert np.count_nonzero(path.codes == path.MOVETO) == 1
    assert np.count_nonzero(path.codes == path.CLOSEPOLY) == 1
    x_min, x_max, y_min, y_max = mollweide_quadrant_bounds([quadrant])
    np.testing.assert_allclose(
        [path.vertices[:, 0].min(), path.vertices[:, 0].max()],
        [x_min, x_max],
    )
    np.testing.assert_allclose(
        [path.vertices[:, 1].min(), path.vertices[:, 1].max()],
        [y_min, y_max],
    )


@pytest.mark.parametrize(
    ("quadrants", "expected_outline_lines"),
    [
        ([2], 3),
        ([1, 2], 4),
        ([1, 3], 4),
        ([1, 4], 6),
    ],
)
def test_partial_quadrant_unions_clip_artists_without_internal_boundaries(
    quadrants,
    expected_outline_lines,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    marker = ax.scatter([0.5], [0.5])
    try:
        patch = configure_mollweide_quadrants(ax, quadrants)
        assert patch is not None
        assert marker.get_clip_path() is not None
        assert len(ax.lines) == expected_outline_lines
        assert np.count_nonzero(patch.get_path().codes == patch.get_path().MOVETO) == len(
            quadrants
        )
    finally:
        plt.close(fig)


def test_all_four_quadrants_leave_full_sky_axes_unchanged():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        assert configure_mollweide_quadrants(ax, [1, 2, 3, 4]) is None
        assert all(spine.get_visible() for spine in ax.spines.values())
        assert not ax.lines
    finally:
        plt.close(fig)


def test_disjoint_quadrants_place_latitude_labels_on_selected_exteriors():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "galactic", [1, 4])
        configure_mollweide_quadrants(ax, [1, 4])
        layout_mollweide_quadrants(ax, [1, 4])
        labels = ax._ciri_quadrant_latitude_labels

        assert {label.xy[0] for label in labels} == {0.0}
        assert {
            label.xyann[0] for label in labels if label.xy[1] > 0
        } == {8.0}
        assert {
            label.get_rotation() for label in labels if label.xy[1] > 0
        } == {270.0}
        assert {
            label.xyann[0] for label in labels if label.xy[1] < 0
        } == {-8.0}
        assert {
            label.get_rotation() for label in labels if label.xy[1] < 0
        } == {90.0}
        assert all(not label.get_visible() for label in ax.get_yticklabels())
        assert ax.get_ylabel() == ""
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("quadrants", "expected_offset", "expected_rotation"),
    [
        ([1, 2], -8.0, 90.0),
        ([1, 3], 8.0, 270.0),
        ([2, 4], -8.0, 90.0),
    ],
)
def test_adjacent_half_skies_have_one_latitude_label_set(
    quadrants,
    expected_offset,
    expected_rotation,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "equatorial", quadrants)
        configure_mollweide_quadrants(ax, quadrants)
        layout_mollweide_quadrants(ax, quadrants)
        labels = ax._ciri_quadrant_latitude_labels

        assert labels
        assert {label.xy[0] for label in labels} == {0.0}
        assert {label.xyann[0] for label in labels} == {expected_offset}
        assert {label.get_rotation() for label in labels} == {expected_rotation}
        assert len({label.xy[1] for label in labels}) == len(labels)
        assert ax.get_ylabel() == ""
    finally:
        plt.close(fig)


def _visible_coordinate_text(frame, quadrants=None):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    configure_mollweide_axes(ax, frame, quadrants)
    if quadrants is not None and len(quadrants) < 4:
        configure_mollweide_quadrants(ax, quadrants)
        layout_mollweide_quadrants(ax, quadrants)
        latitude_labels = ax._ciri_quadrant_latitude_labels
    else:
        latitude_labels = ax._ciri_quadrant_latitude_labels
    longitude_text = [
        label.get_text() for label in ax.get_xticklabels() if label.get_visible()
    ]
    latitude_text = [label.get_text() for label in latitude_labels]
    return fig, longitude_text, latitude_text


@pytest.mark.parametrize(
    ("frame", "quadrants", "longitude_fragment", "latitude_fragment"),
    [
        ("galactic", None, r"\ell = 300^\circ", r"b = 30^\circ"),
        ("galactic", [2], r"\ell = 300^\circ", r"b = 30^\circ"),
        ("equatorial", [1, 3], r"\mathrm{RA} = 30^\circ", r"\mathrm{Dec} = 30^\circ"),
        ("galactic", [3], r"\ell = 120^\circ", r"b = -30^\circ"),
        ("equatorial", [4], r"\mathrm{RA} = 330^\circ", r"\mathrm{Dec} = -30^\circ"),
        ("equatorial", [1, 4], r"\mathrm{RA} = 330^\circ", r"\mathrm{Dec} = 30^\circ"),
    ],
)
def test_one_visible_tick_names_each_coordinate_for_frame_and_selection(
    frame,
    quadrants,
    longitude_fragment,
    latitude_fragment,
):
    fig, longitude_text, latitude_text = _visible_coordinate_text(frame, quadrants)
    try:
        assert sum(longitude_fragment in text for text in longitude_text) == 1
        assert sum(latitude_fragment in text for text in latitude_text) == 1
        longitude_symbol = r"\ell =" if frame == "galactic" else r"\mathrm{RA} ="
        latitude_symbol = r"b =" if frame == "galactic" else r"\mathrm{Dec} ="
        assert sum(longitude_symbol in text for text in longitude_text) == 1
        assert sum(latitude_symbol in text for text in latitude_text) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("quadrants", "removed", "expected"),
    [
        (None, [], 330),
        ([1], [], 30),
        ([2], [], 330),
        ([3], [], 30),
        ([4], [], 330),
        ([1, 3], [], 30),
        ([2, 4], [], 330),
        ([1, 4], [], 330),
        (None, [330], 30),
        ([1, 4], [330], 30),
        (None, [30], 330),
        ([1], [30], 120),
        ([2], [330], 300),
        (None, [30, 330], 300),
    ],
)
def test_equatorial_longitude_annotation_prefers_visible_30_or_330(
    quadrants,
    removed,
    expected,
):
    longitude, _ = preferred_coordinate_ticks(
        quadrants,
        frame="equatorial",
        remove_longitude_ticks=removed,
    )
    assert longitude == expected

    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(
            ax,
            "equatorial",
            quadrants,
            remove_longitude_ticks=removed,
        )
        annotated = [
            label.get_text()
            for label in ax.get_xticklabels()
            if r"\mathrm{RA} =" in label.get_text()
        ]
        assert len(annotated) == 1
        assert f"{expected}^\\circ" in annotated[0]
    finally:
        plt.close(fig)


def test_posterior_legend_patch_has_translucent_face_and_opaque_run_border():
    patch = posterior_legend_patch("RACS-low3", "#ca054d")
    np.testing.assert_allclose(patch.get_facecolor(), [202 / 255, 5 / 255, 77 / 255, 0.4])
    np.testing.assert_allclose(patch.get_edgecolor(), [202 / 255, 5 / 255, 77 / 255, 1.0])
    assert patch.get_linewidth() == pytest.approx(1.3)
    assert patch.get_label() == "RACS-low3"


def test_manual_graticules_are_dotted_pole_to_pole_and_clipped_to_selection():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "galactic", [2])
        longitude_lines = ax._ciri_longitude_graticules
        latitude_lines = ax._ciri_latitude_graticules
        assert longitude_lines and latitude_lines
        assert all(line.get_linestyle() == ":" for line in (*longitude_lines, *latitude_lines))
        for line in longitude_lines:
            latitude = np.asarray(line.get_ydata())
            assert latitude[0] == pytest.approx(-np.pi / 2.0)
            assert latitude[-1] == pytest.approx(np.pi / 2.0)
        assert not any(line.get_visible() for line in ax.get_xgridlines())
        assert not any(line.get_visible() for line in ax.get_ygridlines())

        configure_mollweide_quadrants(ax, [2])
        assert all(
            line.get_clip_path() is not None
            for line in (*longitude_lines, *latitude_lines)
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("quadrants", "expected_latitudes"),
    [
        (None, [-60, -30, 0, 30, 60]),
        ([1], [0, 30, 60]),
        ([2], [0, 30, 60]),
        ([3], [-60, -30, 0]),
        ([4], [-60, -30, 0]),
        ([1, 2], [0, 30, 60]),
        ([1, 3], [-60, -30, 0, 30, 60]),
        ([1, 4], [-60, -30, 0, 30, 60]),
    ],
)
def test_latitude_ticks_are_30_degree_central_meridian_labels(
    quadrants,
    expected_latitudes,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "galactic", quadrants)
        np.testing.assert_array_equal(
            np.rint(np.rad2deg(ax.get_yticks())).astype(int),
            [-60, -30, 0, 30, 60],
        )
        labels = ax._ciri_quadrant_latitude_labels
        assert {label.xy[0] for label in labels} == {0.0}
        np.testing.assert_array_equal(
            sorted(round(np.rad2deg(label.xy[1])) for label in labels),
            expected_latitudes,
        )
        assert ax.get_title() == ""
    finally:
        plt.close(fig)


def test_cropped_bbox_includes_central_latitude_annotations():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(
            ax,
            "equatorial",
            [2],
            remove_longitude_ticks=[300],
            remove_latitude_ticks=[30],
        )
        patch = configure_mollweide_quadrants(ax, [2])
        assert patch is not None
        layout_mollweide_quadrants(ax, [2])
        labels = ax._ciri_quadrant_latitude_labels
        visible_longitude_labels = tuple(
            label
            for label in ax.get_xticklabels()
            if label.get_visible() and label.get_text()
        )
        frame_label = ax._ciri_frame_label
        bbox = selected_mollweide_bbox(
            fig,
            patch,
            extra_artists=(*labels, *visible_longitude_labels, frame_label),
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_display = bbox.transformed(fig.dpi_scale_trans)
        for label in labels:
            label_bbox = label.get_window_extent(renderer)
            assert bbox_display.contains(label_bbox.x0, label_bbox.y0)
            assert bbox_display.contains(label_bbox.x1, label_bbox.y1)
        for label in visible_longitude_labels:
            label_bbox = label.get_window_extent(renderer)
            assert bbox_display.contains(label_bbox.x0, label_bbox.y0)
            assert bbox_display.contains(label_bbox.x1, label_bbox.y1)
        frame_bbox = frame_label.get_window_extent(renderer)
        assert bbox_display.contains(frame_bbox.x0, frame_bbox.y0)
        assert bbox_display.contains(frame_bbox.x1, frame_bbox.y1)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
@pytest.mark.parametrize(
    ("quadrant", "expected_values"),
    [
        (1, [0, 30, 60]),
        (2, [0, 30, 60]),
        (3, [-60, -30, 0]),
        (4, [-60, -30, 0]),
    ],
)
def test_each_quadrant_binds_latitude_text_to_its_actual_data_coordinate(
    frame,
    quadrant,
    expected_values,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, frame, [quadrant])
        labels = ax._ciri_quadrant_latitude_labels
        actual_values = [round(np.rad2deg(label.xy[1])) for label in labels]
        assert actual_values == expected_values
        for value, label in zip(expected_values, labels, strict=True):
            assert f"{value}^\\circ" in label.get_text()
        coordinate_symbol = "b =" if frame == "galactic" else r"\mathrm{Dec} ="
        expected_annotation = 30 if quadrant in (1, 2) else -30
        annotated = [label for label in labels if coordinate_symbol in label.get_text()]
        assert len(annotated) == 1
        assert round(np.rad2deg(annotated[0].xy[1])) == expected_annotation
        expected_offset = 8.0 if quadrant in (1, 3) else -8.0
        expected_rotation = 270.0 if quadrant in (1, 3) else 90.0
        assert {label.xyann[0] for label in labels} == {expected_offset}
        assert all(label.get_rotation() == expected_rotation for label in labels)
        assert all(label.get_verticalalignment() == "center" for label in labels)
        assert all(label.get_horizontalalignment() == "center" for label in labels)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "quadrants",
    [None, [1], [2], [3], [4], [1, 2], [1, 3], [1, 4]],
)
def test_longitude_zero_is_hidden_but_latitude_zero_is_visible(quadrants):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "galactic", quadrants)
        longitude_by_value = {
            int(value): label.get_text()
            for value, label in zip(
                np.mod(-np.rad2deg(ax.get_xticks()), 360).round().astype(int),
                ax.get_xticklabels(),
                strict=True,
            )
        }
        if quadrants is not None:
            configure_mollweide_quadrants(ax, quadrants)
            layout_mollweide_quadrants(ax, quadrants)
        assert longitude_by_value[0] == ""
        latitude_zero = [
            label
            for label in ax._ciri_quadrant_latitude_labels
            if np.isclose(label.xy[1], 0.0)
        ]
        assert len(latitude_zero) == 1
        assert latitude_zero[0].get_text() == r"$0^\circ$"
        assert "b =" not in latitude_zero[0].get_text()
        central_meridians = [
            line
            for line in ax._ciri_longitude_graticules
            if np.allclose(line.get_xdata(), 0.0)
        ]
        assert len(central_meridians) == 1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
def test_remove_latitude_zero_hides_only_its_text(frame):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(
            ax,
            frame,
            remove_latitude_ticks=[0],
        )
        assert not any(
            np.isclose(label.xy[1], 0.0)
            for label in ax._ciri_quadrant_latitude_labels
        )
        longitude_by_value = {
            int(value): label.get_text()
            for value, label in zip(
                np.mod(-np.rad2deg(ax.get_xticks()), 360).round().astype(int),
                ax.get_xticklabels(),
                strict=True,
            )
        }
        assert longitude_by_value[0] == ""
        equator_graticules = [
            line
            for line in ax._ciri_latitude_graticules
            if np.allclose(line.get_ydata(), 0.0)
        ]
        assert len(equator_graticules) == 1
        latitude_symbol = "b =" if frame == "galactic" else r"\mathrm{Dec} ="
        annotated = [
            label
            for label in ax._ciri_quadrant_latitude_labels
            if latitude_symbol in label.get_text()
        ]
        assert len(annotated) == 1
        assert not np.isclose(annotated[0].xy[1], 0.0)
    finally:
        plt.close(fig)


def test_tick_removal_normalization_is_additive_and_accepts_negative_longitude():
    assert normalize_longitude_tick_removals(None) == frozenset({0})
    assert normalize_longitude_tick_removals([30, -60, 360]) == frozenset(
        {0, 30, 300}
    )
    assert normalize_latitude_tick_removals([-60, 30]) == frozenset({-60, 30})


@pytest.mark.parametrize(
    ("normalizer", "values", "message"),
    [
        (normalize_longitude_tick_removals, [45], "Unsupported longitude tick"),
        (normalize_longitude_tick_removals, [30.5], "whole degrees"),
        (normalize_latitude_tick_removals, [-45], "Unsupported latitude tick"),
        (normalize_latitude_tick_removals, [np.inf], "whole degrees"),
    ],
)
def test_tick_removal_validation_rejects_unsupported_values(
    normalizer,
    values,
    message,
):
    with pytest.raises(ValueError, match=message):
        normalizer(values)


def test_multiple_tick_removals_hide_text_but_preserve_every_graticule():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(
            ax,
            "galactic",
            remove_longitude_ticks=[30, 90, -60],
            remove_latitude_ticks=[-60, 30],
        )
        longitude_by_value = {
            int(value): label.get_text()
            for value, label in zip(
                np.mod(-np.rad2deg(ax.get_xticks()), 360).round().astype(int),
                ax.get_xticklabels(),
                strict=True,
            )
        }
        assert longitude_by_value[0] == ""
        assert longitude_by_value[30] == ""
        assert longitude_by_value[90] == ""
        assert longitude_by_value[300] == ""
        latitude_values = {
            round(np.rad2deg(label.xy[1]))
            for label in ax._ciri_quadrant_latitude_labels
        }
        assert latitude_values == {-30, 0, 60}
        assert len(ax._ciri_longitude_graticules) == 11
        assert len(ax._ciri_latitude_graticules) == 5
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("frame", "quadrants", "remove_longitude", "remove_latitude", "longitude", "latitude"),
    [
        ("galactic", None, [300], [30], 270, 60),
        ("equatorial", [1], [120], [30], 30, 60),
        ("galactic", [2], [300], [30], 270, 60),
        ("equatorial", [3], [120], [-30], 30, -60),
        ("galactic", [4], [300], [-30], 270, -60),
        ("equatorial", [1, 2], [300], [30], 330, 60),
        ("galactic", [1, 3], [120], [30], 90, 60),
        ("equatorial", [2, 4], [300], [30], 330, 60),
        ("galactic", [1, 4], [300], [30], 270, 60),
    ],
)
def test_removed_coordinate_annotations_fall_back_within_selected_regions(
    frame,
    quadrants,
    remove_longitude,
    remove_latitude,
    longitude,
    latitude,
):
    assert preferred_coordinate_ticks(
        quadrants,
        frame=frame,
        remove_longitude_ticks=remove_longitude,
        remove_latitude_ticks=remove_latitude,
    ) == (longitude, latitude)
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(
            ax,
            frame,
            quadrants,
            remove_longitude_ticks=remove_longitude,
            remove_latitude_ticks=remove_latitude,
        )
        longitude_symbol = r"\ell =" if frame == "galactic" else r"\mathrm{RA} ="
        latitude_symbol = r"b =" if frame == "galactic" else r"\mathrm{Dec} ="
        annotated_longitude = [
            label.get_text()
            for label in ax.get_xticklabels()
            if longitude_symbol in label.get_text()
        ]
        annotated_latitude = [
            label.get_text()
            for label in ax._ciri_quadrant_latitude_labels
            if latitude_symbol in label.get_text()
        ]
        assert len(annotated_longitude) == 1
        assert f"{longitude}^\\circ" in annotated_longitude[0]
        assert len(annotated_latitude) == 1
        assert f"{latitude}^\\circ" in annotated_latitude[0]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("quadrants", "longitude_removals", "latitude_removals"),
    [
        ([1], [30, 60, 90, 120, 150], [0, 30, 60]),
        ([2], [210, 240, 270, 300, 330], [0, 30, 60]),
        ([3], [30, 60, 90, 120, 150], [-60, -30, 0]),
        ([4], [210, 240, 270, 300, 330], [-60, -30, 0]),
    ],
)
def test_removing_all_selected_axis_candidates_omits_annotations_cleanly(
    quadrants,
    longitude_removals,
    latitude_removals,
):
    assert preferred_coordinate_ticks(
        quadrants,
        remove_longitude_ticks=longitude_removals,
        remove_latitude_ticks=latitude_removals,
    ) == (None, None)
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(
            ax,
            "galactic",
            quadrants,
            remove_longitude_ticks=longitude_removals,
            remove_latitude_ticks=latitude_removals,
        )
        configure_mollweide_quadrants(ax, quadrants)
        layout_mollweide_quadrants(ax, quadrants)
        assert not any(
            label.get_text()
            for label in ax.get_xticklabels()
            if label.get_visible()
        )
        assert ax._ciri_quadrant_latitude_labels == ()
        assert ax._ciri_frame_label.get_visible()
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--remove-longitude-ticks", "45", "Unsupported longitude tick"),
        ("--remove-latitude-ticks", "-45", "Unsupported latitude tick"),
    ],
)
def test_cli_reports_invalid_tick_removals(option, value, message):
    script = Path(__file__).parents[1] / "scripts" / "figx-skymap.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--low3-artifact",
            "unused-low3.zip",
            "--mid1-artifact",
            "unused-mid1.zip",
            option,
            value,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_central_latitude_labels_mask_graticules_with_minimal_white_background():
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, "galactic", [2])
        graticule_zorders = [
            line.get_zorder()
            for line in (
                *ax._ciri_longitude_graticules,
                *ax._ciri_latitude_graticules,
            )
        ]
        for label in ax._ciri_quadrant_latitude_labels:
            assert label.get_zorder() > max(graticule_zorders)
            background = label.get_bbox_patch()
            np.testing.assert_allclose(background.get_facecolor(), [1.0, 1.0, 1.0, 1.0])
            assert background.get_edgecolor()[-1] == 0.0
            assert background.get_linewidth() == 0.0
            assert background.get_boxstyle().pad == pytest.approx(0.08)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
def test_full_sky_latitude_labels_default_left_and_rotate_vertically(frame):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, frame)
        labels = ax._ciri_quadrant_latitude_labels
        assert labels
        assert {label.xy[0] for label in labels} == {0.0}
        assert {label.xyann[0] for label in labels} == {-8.0}
        assert all(label.get_rotation() == 90.0 for label in labels)
        assert all(label.get_verticalalignment() == "center" for label in labels)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("frame", ["galactic", "equatorial"])
@pytest.mark.parametrize(
    ("quadrants", "expected_position", "expected_alignment"),
    [
        (None, (0.975, 0.925), "right"),
        ([1, 2, 3, 4], (0.975, 0.925), "right"),
        ([1], (0.025, 0.925), "left"),
        ([2], (0.975, 0.925), "right"),
        ([3], (0.025, 0.075), "left"),
        ([4], (0.975, 0.075), "right"),
        ([1, 2], (0.975, 0.925), "right"),
        ([1, 3], (0.025, 0.925), "left"),
        ([2, 4], (0.975, 0.925), "right"),
        ([3, 4], (0.975, 0.075), "right"),
        ([1, 4], (0.025, 0.925), "left"),
    ],
)
def test_frame_label_uses_healpy_corner_priority(
    frame,
    quadrants,
    expected_position,
    expected_alignment,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"})
    try:
        configure_mollweide_axes(ax, frame, quadrants)
        label = ax._ciri_frame_label
        assert label.get_visible()
        assert label.get_text() == (
            "Galactic" if frame == "galactic" else "Equatorial"
        )
        assert label.get_position() == expected_position
        assert label.get_transform() is ax.transAxes
        assert label.get_rotation() == 0.0
        assert label.get_horizontalalignment() == expected_alignment
        assert label.get_verticalalignment() == "center"
        assert not label.get_clip_on()
        if quadrants is not None and len(quadrants) < 4:
            clip_patch = configure_mollweide_quadrants(ax, quadrants)
            assert clip_patch is not None
            layout_mollweide_quadrants(ax, quadrants)
        else:
            clip_patch = None

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_bbox = label.get_window_extent(renderer)
        axes_bbox = ax.get_window_extent(renderer)
        assert axes_bbox.contains(label_bbox.x0, label_bbox.y0)
        assert axes_bbox.contains(label_bbox.x1, label_bbox.y1)
        if clip_patch is not None:
            crop_bbox = selected_mollweide_bbox(
                fig,
                clip_patch,
                extra_artists=(label,),
            ).transformed(fig.dpi_scale_trans)
            assert crop_bbox.contains(label_bbox.x0, label_bbox.y0)
            assert crop_bbox.contains(label_bbox.x1, label_bbox.y1)
    finally:
        plt.close(fig)
