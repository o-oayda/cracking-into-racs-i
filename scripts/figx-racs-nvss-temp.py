from astropy.io.registry import IORegistryError
from catsim.racs_products import RACS_LOW3, RACS_MID1
from catsim.racs_temperature import evaluate_temperature_response
from dipoleutils import DataLoader, CatalogueToMap
import argparse
from pathlib import Path
from astropy.table import Table
from dipoleutils.utils.crossmatch import CrossMatch
from dipoleutils.utils.plotting import plot_binned_mean, plot_binned_quantile
import numpy as np
from catsim.racs import Racs, RacsConfig
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from ciri.palette import CELADON, DRY_SAGE, MUTED_TEAL, RACS_SURVEY_COLOURS
from ciri.artifact_io import load_final_posterior
from ciri.plot_style import configure_plot_style


configure_plot_style()



FLUX_MIN = 15
FLUX_MAX = None
CROSSMATCH_RADIUS = 5
SPECTRAL_INDEX = {  # S ~ \nu ^{-\alpha}
    'low3': 1.0,
    'mid1': 0.8,
}
ONE_SIGMA_QUANTILES = (0.158655254, 0.5, 0.841344746)
MODEL_MEDIAN_LINE_WIDTH = 1.2
MODEL_INTERVAL_LINE_WIDTH = 0.
MODEL_INTERVAL_LINE_ALPHA = 0.6
MODEL_INTERVAL_FILL_ALPHA = 0.2
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
    argparser.add_argument(
        '--noisemap-dir',
        type=str,
        help='Path to dor containing noisemaps.'
    )
    argparser.add_argument(
        '--low3-artifact',
        type=Path,
        required=True,
        help='LOW3 analysis artifact containing posterior samples.'
    )
    argparser.add_argument(
        '--mid1-artifact',
        type=Path,
        required=True,
        help='MID1 analysis artifact containing posterior samples.'
    )
    argparser.add_argument(
        '--tile-separation-cut',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Limit RACS sources to within 2 degrees of their tile centre '
            '(default: enabled).'
        ),
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
    ) ** -SPECTRAL_INDEX[survey]
    racs_flux = np.asarray(common_sources['A_Total_flux'], dtype=float)
    flux_ratio = racs_flux / scaled_nvss_flux
    return flux_ratio


def get_lookups_from_model(racs_sample, paf_dir_path, noisemap_dir):
    # initialise a blank model to load catsim lookups
    model = Racs(
        RacsConfig(
            FLUX_MIN,
            product=survey_to_product[racs_sample],
            paf_temperature_data_dir=paf_dir_path,
            noisemap_data_dir=noisemap_dir
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
    racs_sample: str, racs_path: str, nvss_path: str, paf_dir_path: str,
    noisemap_dir: str, tile_separation_cut: bool,
    ):
    racs = CatalogueToMap(load_catalogue(racs_path))
    nvss = CatalogueToMap(load_catalogue(nvss_path))
    racs.make_cut('Total_flux', minimum=FLUX_MIN, maximum=FLUX_MAX)
    nvss.make_cut('integrated_flux', minimum=FLUX_MIN, maximum=FLUX_MAX)

    if racs_sample == 'mid1' and tile_separation_cut:
        racs.make_cut('Tile_centre_sep', minimum=None, maximum=2)
        print(np.max(racs.catalogue['Tile_centre_sep']))
    elif racs_sample == 'low3' and tile_separation_cut:
        racs.make_cut('Tile_sep', minimum=None, maximum=2)
        print(np.max(racs.catalogue['Tile_sep']))

    temp_by_idx, idx_by_sbid = get_lookups_from_model(racs_sample, paf_dir_path, noisemap_dir)
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


def flux_ratio_at_temperature(
    bin_temperatures,
    flux_ratios,
    temperature: float = 25.0,
) -> float:
    bin_temperatures = np.asarray(bin_temperatures, dtype=float)
    flux_ratios = np.asarray(flux_ratios, dtype=float)
    matches = np.flatnonzero(np.isclose(bin_temperatures, temperature))
    if matches.size != 1:
        raise ValueError(
            f'Expected exactly one empirical flux-ratio bin at {temperature:g} C; '
            f'found {matches.size}'
        )
    reference_ratio = float(flux_ratios[matches[0]])
    if not np.isfinite(reference_ratio):
        raise ValueError(
            f'Empirical flux ratio at {temperature:g} C is not finite'
        )
    return reference_ratio


def plot_model_fit(
    path: str | Path,
    color: str,
    reference_flux_ratio: float,
) -> None:
    samples = load_final_posterior(path)
    beta_quantiles = np.asarray(
        samples['temp_beta'].quantile(ONE_SIGMA_QUANTILES),
        dtype=float,
    )
    temps = np.linspace(23, 35, 1000)
    temperature_response = evaluate_temperature_response(
        temps[None, :],
        beta_quantiles[:, None],
        25,
        model="hot_linear",
        xp=np,
    )
    model_curves = reference_flux_ratio * temperature_response
    lower_curve = np.minimum(model_curves[0], model_curves[2])
    upper_curve = np.maximum(model_curves[0], model_curves[2])
    axis = plt.gca()
    axis.plot(
        temps,
        model_curves[1],
        color=color,
        linestyle='dotted',
        linewidth=MODEL_MEDIAN_LINE_WIDTH,
        zorder=0,
    )
    axis.plot(
        temps,
        lower_curve,
        color=color,
        alpha=MODEL_INTERVAL_LINE_ALPHA,
        linewidth=MODEL_INTERVAL_LINE_WIDTH,
        zorder=0,
    )
    axis.plot(
        temps,
        upper_curve,
        color=color,
        alpha=MODEL_INTERVAL_LINE_ALPHA,
        linewidth=MODEL_INTERVAL_LINE_WIDTH,
        zorder=0,
    )
    axis.fill_between(
        temps,
        lower_curve,
        upper_curve,
        color=color,
        alpha=MODEL_INTERVAL_FILL_ALPHA,
        linewidth=0,
        zorder=-1,
    )


def main():
    args = parse_args()
    low3_bins, low3_ratios, low3_errs = get_ratio_per_temp_bin(
        'low3', racs_path=args.low3_path, nvss_path=args.nvss_path,
        paf_dir_path=args.paf_dir_path, noisemap_dir=args.noisemap_dir,
        tile_separation_cut=args.tile_separation_cut,
    )
    mid1_bins, mid1_ratios, mid1_errs = get_ratio_per_temp_bin(
        'mid1', racs_path=args.mid1_path, nvss_path=args.nvss_path,
        paf_dir_path=args.paf_dir_path, noisemap_dir=args.noisemap_dir,
        tile_separation_cut=args.tile_separation_cut,
    )

    plt.figure(figsize=(4,3))
    draw_ratio_plot(
        low3_bins, low3_ratios, low3_errs, label='RACS-low3', color=RACS_SURVEY_COLOURS['low3'],
        marker='o'
    )
    plot_model_fit(
        args.low3_artifact,
        RACS_SURVEY_COLOURS['low3'],
        flux_ratio_at_temperature(low3_bins, low3_ratios),
    )
    draw_ratio_plot(
        mid1_bins, mid1_ratios, mid1_errs, label='RACS-mid1', color=RACS_SURVEY_COLOURS['mid1'],
        marker='s'
    )
    plot_model_fit(
        args.mid1_artifact,
        RACS_SURVEY_COLOURS['mid1'],
        flux_ratio_at_temperature(mid1_bins, mid1_ratios),
    )
    ax = plt.gca()
    ax.axvspan(22, 25, color=DRY_SAGE, alpha=0.3)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xlabel(r'PAF temperature ($^\circ$C)')
    plt.ylabel(r'Mean flux ratio (RACS / NVSS)')
    plt.xlim(22.5, 35)
    plt.legend()
    plt.savefig('output/figx-racs-nvss-temp.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
