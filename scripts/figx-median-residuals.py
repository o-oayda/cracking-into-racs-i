import argparse
from pathlib import Path

from catsim import Racs, prng_key
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from numpy.typing import NDArray

from ciri.artifact_io import (
    load_final_posterior,
    load_model_config,
    load_reference_observation,
)
from ciri.plotting import smooth_map


POSTERIOR_PARAMETERS = (
    'log10_n_initial_samples',
    'observer_speed',
    'dipole_longitude',
    'dipole_latitude',
    'temp_beta',
    'lambda_clus',
)
NATIVE_NSIDE = 64


plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{sfmath}",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low3-artifact', type=Path, required=True)
    parser.add_argument('--mid1-artifact', type=Path, required=True)
    parser.add_argument('--low3-catalogue-path', type=Path, required=True)
    parser.add_argument('--mid1-catalogue-path', type=Path, required=True)
    parser.add_argument('--paf-dir-path', type=Path, required=True)
    parser.add_argument('--noisemap-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output/figx-median-residuals.pdf'),
    )
    return parser.parse_args()


def load_median_parameters(artifact_path: Path) -> dict[str, float]:
    samples = load_final_posterior(artifact_path)
    parameters = {
        name: float(samples[name].quantile(0.5))
        for name in POSTERIOR_PARAMETERS
    }
    invalid = [name for name, value in parameters.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(
            'Posterior median is not finite for parameter(s): '
            + ', '.join(invalid)
        )
    return parameters


def simulate_smoothed_map(
    artifact_path: Path,
    catalogue_path: Path,
    paf_dir_path: Path,
    noisemap_dir: Path,
    expected_product: str,
    rng_key,
) -> NDArray[np.floating]:
    config = load_model_config(artifact_path)
    if config.product.key != expected_product:
        raise ValueError(
            f'Expected a {expected_product} artifact; got {config.product.key}'
        )
    config.catalogue_path = str(catalogue_path.expanduser())
    config.paf_temperature_data_dir = str(paf_dir_path.expanduser())
    config.noisemap_data_dir = noisemap_dir
    config.downscale_nside = None
    _, native_mask = load_reference_observation(
        artifact_path,
        load_native=True,
    )
    expected_shape = (hp.nside2npix(NATIVE_NSIDE),)
    if native_mask.shape != expected_shape:
        raise ValueError(
            f'Native observation mask must have shape {expected_shape}; '
            f'got {native_mask.shape}'
        )
    config.mask_map = native_mask

    model = Racs(config)
    model.initialise_data()
    simulation, _ = model.generate_dipole(
        **load_median_parameters(artifact_path),
        rng_key=rng_key,
    )
    if simulation.shape != expected_shape:
        raise ValueError(
            f'Simulation must have shape {expected_shape}; got {simulation.shape}'
        )
    return smooth_map(simulation, only_return_data=True)


def add_map_panel(
    sample_map: NDArray[np.floating],
    sub: int,
    cmap,
    norm: Normalize,
):
    hp.projview(
        sample_map,
        cmap=cmap,
        nest=True,
        # norm=norm,
        cbar=False,
        sub=sub,
        dpi=300,
    )
    return plt.gca()


def add_map_colorbar(
    figure,
    panel_axes,
    cmap,
    norm: Normalize,
    sample_min: float,
    sample_max: float,
) -> None:
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


def main() -> None:
    args = parse_args()
    low3_key, mid1_key = prng_key(args.seed).split(2)
    low3_map = simulate_smoothed_map(
        args.low3_artifact,
        args.low3_catalogue_path,
        args.paf_dir_path,
        args.noisemap_dir,
        'low3',
        low3_key,
    )
    mid1_map = simulate_smoothed_map(
        args.mid1_artifact,
        args.mid1_catalogue_path,
        args.paf_dir_path,
        args.noisemap_dir,
        'mid1',
        mid1_key,
    )

    # low3_min = float(np.nanmin(low3_map))
    # low3_max = float(np.nanmax(low3_map))
    # mid1_min = float(np.nanmin(mid1_map))
    # mid1_max = float(np.nanmax(mid1_map))
    # from the empirical sample figure
    low3_min = 13.
    low3_max = 14.4
    mid1_min = 10.
    mid1_max = 11.1
    cmap = plt.get_cmap('magma')
    low3_norm = Normalize(vmin=low3_min, vmax=low3_max)
    mid1_norm = Normalize(vmin=mid1_min, vmax=mid1_max)

    left_axes = add_map_panel(low3_map, 121, cmap, low3_norm)
    right_axes = add_map_panel(mid1_map, 122, cmap, mid1_norm)
    left_axes.text(
        0.02,
        -0.03,
        'RACS-low3',
        transform=left_axes.transAxes,
        ha='left',
        va='bottom',
    )
    right_axes.text(
        0.98,
        -0.03,
        'RACS-mid1',
        transform=right_axes.transAxes,
        ha='right',
        va='bottom',
    )

    figure = plt.gcf()
    add_map_colorbar(
        figure,
        left_axes,
        cmap,
        low3_norm,
        low3_min,
        low3_max,
    )
    add_map_colorbar(
        figure,
        right_axes,
        cmap,
        mid1_norm,
        mid1_min,
        mid1_max,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
