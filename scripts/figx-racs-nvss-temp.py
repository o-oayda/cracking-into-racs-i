from astropy.io.registry import IORegistryError
from catsim.racs_products import RACS_LOW3, RACS_MID1
from dipoleutils import DataLoader, CatalogueToMap
import argparse
from pathlib import Path
from astropy.table import Table
from dipoleutils.utils.crossmatch import CrossMatch
from dipoleutils.utils.plotting import plot_binned_mean, plot_binned_quantile
import numpy as np
from catsim.racs import Racs, RacsConfig
import matplotlib.pyplot as plt
from palette import CELADON, MUTED_TEAL, MUTED_TEAL_DARK, ROSEWOOD
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{sfmath}",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    }
)



FLUX_MIN = 15
FLUX_MAX = None
CROSSMATCH_RADIUS = 5
SPECTRAL_INDEX = 0.8 # S ~ \nu ^{-\alpha}
SURVEY_FREQUENCY_MHZ = {
    'low3': 943.5,
    'mid1': 1367.5,
    'nvss': 1400.
}
survey_to_product = {
    'low3': RACS_LOW3,
    'mid1': RACS_MID1
}


def parse_args():
    argparser = argparse.ArgumentParser()
    # argparser.add_argument(
    #     '--racs-sample',
    #     choices=['low3', 'mid1'],
    #     type=str,
    #     help='RACS epoch to use for comparison with NVSS.'
    # )
    argparser.add_argument(
        '--low3-path',
        type=str,
        help='Path of low3 catalogue.'
    )
    argparser.add_argument(
        '--mid1-path',
        type=str,
        help='Path of low3 catalogue.'
    )
    argparser.add_argument(
        '--nvss-path',
        type=str,
        help='Path of nvss catalogue.'
    )
    argparser.add_argument(
        '--paf-dir-path',
        type=str,
        help='Path to dir containing PAF temps.'
    )
    args = argparser.parse_args()
    return args


def load_catalogue(catalogue_path: str | Path) -> Table:
    path = Path(catalogue_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Catalogue does not exist: {path}")
    try:
        return Table.read(path, unit_parse_strict="silent")
    except IORegistryError:
        return Table.read(path, format='ascii')


def get_common_sources(racs: CatalogueToMap, nvss: CatalogueToMap) -> Table:
    xmatch = CrossMatch(
        racs.get_catalogue(),
        nvss.get_catalogue(),
        coordinate_system='equatorial'
    )
    xmatch.cross_match(
        radius=CROSSMATCH_RADIUS,
        source_name_A_column='Source_ID',
        source_name_B_column='source_name',
    )
    common_table = xmatch.get_common_sources()
    return common_table


def compute_flux_ratio(common_sources: Table, survey: str):
    scaled_nvss_flux = np.asarray(common_sources['B_integrated_flux'], dtype=float) * (
        SURVEY_FREQUENCY_MHZ[survey] / SURVEY_FREQUENCY_MHZ['nvss']
    ) ** -SPECTRAL_INDEX
    racs_flux = np.asarray(common_sources['A_Total_flux'], dtype=float)
    flux_ratio = racs_flux / scaled_nvss_flux
    return flux_ratio


def get_lookups_from_model(racs_sample, paf_dir_path):
    # initialise a blank model to load catsim lookups
    model = Racs(
        RacsConfig(
            FLUX_MIN,
            product=survey_to_product[racs_sample],
            paf_temperature_data_dir=paf_dir_path
        )
    )
    model.initialise_data()
    temp_by_idx = model.tile_temperature_by_index
    idx_by_sbid = model._tile_index_from_sbid
    return temp_by_idx, idx_by_sbid


def sbid_to_tile_idx(sbid, idx_by_sbid):
    tile_indices = np.array(
        [idx_by_sbid.get(int(source_sbid), -1) for source_sbid in sbid],
        dtype=np.int32,
    )
    return tile_indices

def idx_to_temp(temp_by_idx, idxs):
    temperature = temp_by_idx[idxs]
    return temperature


def draw_ratio_plot(x, y, y_err, **kwargs):
    print(x)
    plt.errorbar(x, y, yerr=y_err, **kwargs)


def get_ratio_per_temp_bin(
        racs_sample: str, racs_path: str, nvss_path: str, paf_dir_path: str
    ):
    racs = CatalogueToMap(load_catalogue(racs_path))
    nvss = CatalogueToMap(load_catalogue(nvss_path))
    racs.make_cut('Total_flux', minimum=FLUX_MIN, maximum=FLUX_MAX)
    nvss.make_cut('integrated_flux', minimum=FLUX_MIN, maximum=FLUX_MAX)

    if racs_sample == 'mid1':
        racs.make_cut('Tile_centre_sep', minimum=None, maximum=2)
        print(np.max(racs.catalogue['Tile_centre_sep']))
    elif racs_sample == 'low3':
        racs.make_cut('Tile_sep', minimum=None, maximum=2)
        print(np.max(racs.catalogue['Tile_sep']))

    temp_by_idx, idx_by_sbid = get_lookups_from_model(racs_sample, paf_dir_path)
    tile_indices = sbid_to_tile_idx(racs.get_catalogue()['SBID'], idx_by_sbid)
    racs_temperature = idx_to_temp(temp_by_idx, tile_indices)
    cat = racs.get_catalogue()
    cat['temperature_C'] = racs_temperature

    common_sources = get_common_sources(CatalogueToMap(cat), nvss)
    racs_temperature = np.asarray(common_sources['A_temperature_C'], dtype=float)
    flux_ratio = compute_flux_ratio(common_sources, racs_sample)

    # going to redraw this because the function is ass
    edges = np.arange(22.5, 35.5, 1)
    ratio_bin_temperatures, flux_ratio, ratio_errs, _ = plot_binned_mean(
        racs_temperature,
        flux_ratio,
        bins=edges,
        # n_bootstrap=500,
        bootstrap_resamples=500,
        color='tab:blue'
    )
    plt.close()
    return ratio_bin_temperatures, flux_ratio, ratio_errs


def main():
    args = parse_args()
    low3_bins, low3_ratios, low3_errs = get_ratio_per_temp_bin(
        'low3', racs_path=args.low3_path, nvss_path=args.nvss_path,
        paf_dir_path=args.paf_dir_path
    )
    mid1_bins, mid1_ratios, mid1_errs = get_ratio_per_temp_bin(
        'mid1', racs_path=args.mid1_path, nvss_path=args.nvss_path,
        paf_dir_path=args.paf_dir_path
    )

    plt.figure(figsize=(4,3))
    draw_ratio_plot(
        low3_bins, low3_ratios, low3_errs, label='RACS-low3', color=MUTED_TEAL_DARK,
        marker='o'
    )
    draw_ratio_plot(
        mid1_bins, mid1_ratios, mid1_errs, label='RACS-mid1', color=ROSEWOOD,
        marker='s'
    )
    ax = plt.gca()
    ax.axvspan(22, 25, color='saddlebrown', alpha=0.2)
    plt.xlabel(r'PAF temperature ($^\circ$C)')
    plt.ylabel(r'Mean flux ratio (RACS / NVSS)')
    plt.xlim(22.5, None)
    plt.legend()
    plt.savefig('output/figx-racs-nvss-temp.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
