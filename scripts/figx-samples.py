import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from ciri.artifact_io import load_reference_observation
from ciri.palette import RACS_SURVEY_COLOURS
from ciri.plotting import smooth_map
from ciri.plot_style import configure_plot_style, configure_sky_graticule


NATIVE_NSIDE = 64
SMOOTH_SCALE_STERADIANS = 1.


configure_plot_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid1-artifact", type=Path, required=True)
    parser.add_argument("--low3-artifact", type=Path, required=True)
    return parser.parse_args()


def load_smoothed_map(artifact_path: Path) -> np.ndarray:
    reference_observation, _ = load_reference_observation(
        artifact_path=artifact_path,
        load_native=True,
    )
    expected_shape = (hp.nside2npix(NATIVE_NSIDE),)
    if reference_observation.shape != expected_shape:
        raise ValueError(
            f'Native reference observation must have shape {expected_shape}; '
            f'got {reference_observation.shape}'
        )
    return smooth_map(
        reference_observation,
        angle_scale=SMOOTH_SCALE_STERADIANS,
        only_return_data=True
    )


def add_sample_panel(
    sample_map: np.ndarray,
    sub: int,
    cmap,
    norm: Normalize,
    latitude_label_side: str,
):
    hp.projview(
        sample_map,
        cmap=cmap,
        nest=True,
        norm=norm,
        cbar=False,
        sub=sub,
        dpi=300,
    )
    axes = plt.gca()
    configure_sky_graticule(axes, latitude_label_side, colour='light')
    return axes


def add_sample_colorbar(
    figure,
    panel_axes,
    cmap,
    norm: Normalize,
    sample_min: float,
    sample_max: float,
):
    panel_position = panel_axes.get_position()
    colorbar_width = 0.7 * panel_position.width
    colorbar_left = panel_position.x0 + (panel_position.width - colorbar_width) / 2
    colorbar_axes = figure.add_axes(
        [colorbar_left, 0.04, colorbar_width, 0.035]
    )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axes,
        orientation='horizontal',
        ticks=[sample_min, sample_max],
        format='%.3g',
        label='Smoothed count per pixel',
    )
    colorbar.ax.xaxis.set_label_position('bottom')
    colorbar.ax.xaxis.labelpad = -10
    colorbar.ax.set_xlim(sample_min, sample_max)


def main():
    args = parse_args()
    low3_map = load_smoothed_map(args.low3_artifact)
    mid1_map = load_smoothed_map(args.mid1_artifact)

    low3_min = np.nanmin(low3_map)
    low3_max = np.nanmax(low3_map)
    mid1_min = np.nanmin(mid1_map)
    mid1_max = np.nanmax(mid1_map)
    cmap = plt.get_cmap('magma')
    low3_norm = Normalize(vmin=low3_min, vmax=low3_max)
    mid1_norm = Normalize(vmin=mid1_min, vmax=mid1_max)

    left_axes = add_sample_panel(low3_map, 121, cmap, low3_norm, 'right')
    right_axes = add_sample_panel(mid1_map, 122, cmap, mid1_norm, 'left')
    left_axes.text(
        0.02,
        -0.03,
        'RACS-low3',
        color=RACS_SURVEY_COLOURS['low3'],
        transform=left_axes.transAxes,
        ha='left',
        va='bottom',
    )
    right_axes.text(
        0.98,
        -0.03,
        'RACS-mid1',
        color=RACS_SURVEY_COLOURS['mid1'],
        transform=right_axes.transAxes,
        ha='right',
        va='bottom',
    )

    figure = plt.gcf()
    add_sample_colorbar(
        figure,
        left_axes,
        cmap,
        low3_norm,
        low3_min,
        low3_max,
    )
    add_sample_colorbar(
        figure,
        right_axes,
        cmap,
        mid1_norm,
        mid1_min,
        mid1_max,
    )
    plt.savefig('output/figx-samples.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
