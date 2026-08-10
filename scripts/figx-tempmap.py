import argparse
from pathlib import Path
import catsim
from ciri.artifact_io import load_model_config
from ciri.plot_style import configure_plot_style
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import numpy as np


configure_plot_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid1-artifact", type=Path, required=False)
    parser.add_argument("--low3-artifact", type=Path, required=False)
    return parser.parse_args()


def load_racs_object(artifact) -> catsim.Racs:
    cfg = load_model_config(artifact)
    cfg.catalogue_path = '/home/oliver/catalogue_data/racs/mid1/RACS-mid1_noscale_srl.fits'
    cfg.paf_temperature_data_dir = '/home/oliver/Documents/dipole-utils/data/paf_temps'
    racs = catsim.Racs(cfg)
    racs.initialise_data()
    return racs


def add_temp_panel(
    racs: catsim.Racs,
    sub: int,
    cmap,
    norm: CenteredNorm,
):
    hp.projview(
        racs.temperature_map,
        cmap=cmap,
        nest=True,
        norm=norm,
        cbar=False,
        sub=sub,
        dpi=300
    )
    return plt.gca()


def main():
    args = parse_args()
    low3 = load_racs_object(args.low3_artifact)
    mid1 = load_racs_object(args.mid1_artifact)

    temperature_min = min(
        np.nanmin(low3.temperature_map),
        np.nanmin(mid1.temperature_map),
    )
    temperature_max = max(
        np.nanmax(low3.temperature_map),
        np.nanmax(mid1.temperature_map),
    )
    temperature_span = max(
        25.0 - temperature_min,
        temperature_max - 25.0,
    )
    cmap = plt.get_cmap('coolwarm')
    norm = CenteredNorm(vcenter=25.0, halfrange=temperature_span)

    left_axes = add_temp_panel(low3, 121, cmap, norm)
    right_axes = add_temp_panel(mid1, 122, cmap, norm)
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
    colorbar_axes = figure.add_axes([0.30, 0.06, 0.40, 0.035])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axes,
        orientation='horizontal',
        ticks=[temperature_min, 25.0, temperature_max],
        format='%.3g',
        label=r'PAF Temperature ($^\circ$C)',
    )
    colorbar.ax.xaxis.set_label_position('top')
    colorbar.ax.set_xlim(temperature_min, temperature_max)
    plt.savefig('output/figx-tempmap.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
