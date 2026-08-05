import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from astropy.io import fits
from astropy.time import Time
from catsim.utils.weather import (
    DEFAULT_MAX_INTERPOLATION_GAP_MINUTES,
    get_mean_paf_temperatures_for_observations,
)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ciri.palette import DARK_AMETHYST, DRY_SAGE, HOT_FUCHSIA, ROSY_TAUPE, ROYAL_PLUM, STEEL_BLUE, TANGERINE_DREAM
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{sfmath}",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    }
)


EXAMPLE_SAMPLE = 'low3'
ANTENNAS = tuple(f'ak{number:02d}' for number in range(1, 37))
TEMPERATURE_COLUMN = 'Temperature'
AWST_UTC_OFFSET = timedelta(hours=8)
AWST_TIMEZONE = ZoneInfo('Australia/Perth')
MEAN_PAF_CADENCE = '1min'
OUTPUT_PATH = Path('output/figx-paf-temp.pdf')


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from error


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--paf-dir-path',
        type=str,
        required=True,
        help='Path to dir containing PAF temps.'
    )
    argparser.add_argument(
        '--observation-day',
        type=parse_date,
        required=True,
        metavar='YYYY-MM-DD',
        help='Only plot temperature reports from this day.'
    )
    argparser.add_argument(
        '--catalogue-path',
        type=str,
        required=True,
        help='Path to a catalogue containing the Scan_start_MJD column.'
    )
    args = argparser.parse_args()
    return args


def load_paf_temps(path: str, observation_day: date):
    sample_path = Path(path).expanduser() / EXAMPLE_SAMPLE
    files_by_antenna = {
        antenna: sorted(
            sample_path.glob(f'{antenna} ctrl_adc1_pafAvTemp-data-*.csv')
        )
        for antenna in ANTENNAS
    }
    missing_antennas = [
        antenna for antenna, paths in files_by_antenna.items() if not paths
    ]
    if missing_antennas:
        raise FileNotFoundError(
            'No PAF-temperature CSV found for: ' + ', '.join(missing_antennas)
        )

    temps_by_antenna = {}
    for antenna, paths in files_by_antenna.items():
        temperature_column = f'{antenna} ctrl_adc1_pafAvTemp (via MoniCA)'
        antenna_temps = []
        for csv_path in paths:
            temps = pd.read_csv(
                csv_path,
                usecols=['Time', temperature_column],
                parse_dates=['Time'],
            )
            temps = select_observation_day(temps, observation_day)
            temps = temps.rename(columns={temperature_column: TEMPERATURE_COLUMN})
            antenna_temps.append(temps)
        temps_by_antenna[antenna] = pd.concat(
            antenna_temps, ignore_index=True
        ).drop_duplicates(subset='Time')

    return temps_by_antenna


def load_ambient_temperature(path: str, observation_day: date):
    sample_path = Path(path).expanduser() / EXAMPLE_SAMPLE
    ambient_paths = sorted(
        sample_path.glob('Temperature & Humidity-data-*.csv')
    )


    if not ambient_paths:
        raise FileNotFoundError(
            f'No ambient-temperature CSV found in {sample_path}'
        )

    ambient_temps = []
    for csv_path in ambient_paths:
        temps = pd.read_csv(
            csv_path,
            usecols=['Time', 'temperature'],
            parse_dates=['Time'],
        )
        # The weather export uses UTC, while the antenna timestamps use local
        # ASKAP time (AWST, UTC+8). Convert before selecting the local day.
        temps['Time'] = temps['Time'] + AWST_UTC_OFFSET
        temps = select_observation_day(temps, observation_day)
        temps = temps.rename(columns={'temperature': TEMPERATURE_COLUMN})
        ambient_temps.append(temps)

    return (
        pd.concat(ambient_temps, ignore_index=True)
        .drop_duplicates(subset='Time')
        .dropna(subset=[TEMPERATURE_COLUMN]),
        ambient_paths,
    )


def compute_catsim_mean_paf_temperature(
    path: str, observation_day: date
) -> pd.DataFrame:
    sample_path = Path(path).expanduser() / EXAMPLE_SAMPLE
    day_start = datetime.combine(
        observation_day, time.min, tzinfo=AWST_TIMEZONE
    )
    day_end = day_start + timedelta(days=1)
    awst_times = pd.date_range(
        day_start,
        day_end,
        freq=MEAN_PAF_CADENCE,
        inclusive='left',
    )
    observation_mjd = Time(awst_times.to_pydatetime()).mjd
    mean_temperatures = get_mean_paf_temperatures_for_observations(
        observation_mjd,
        data_dir=sample_path,
        max_interpolation_gap_minutes=DEFAULT_MAX_INTERPOLATION_GAP_MINUTES,
    )
    return pd.DataFrame(
        {
            'Time': awst_times.tz_localize(None),
            TEMPERATURE_COLUMN: mean_temperatures,
        }
    )


def select_observation_day(temps: pd.DataFrame, observation_day: date):
    day_start = datetime.combine(observation_day, time.min)
    day_end = day_start + timedelta(days=1)
    return temps.loc[
        (temps['Time'] >= day_start) & (temps['Time'] < day_end)
    ]


def load_scan_start_times(
    catalogue_path: str, observation_day: date
) -> list[datetime]:
    path = Path(catalogue_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Catalogue does not exist: {path}')

    with fits.open(path, memmap=True) as hdus:
        catalogue_hdu = next(
            (
                hdu for hdu in hdus
                if getattr(hdu, 'columns', None) is not None
                and 'Scan_start_MJD' in hdu.columns.names
            ),
            None,
        )
        if catalogue_hdu is None:
            raise ValueError(
                f'Catalogue does not contain Scan_start_MJD: {path}'
            )
        scan_start_mjd = np.asarray(
            catalogue_hdu.data['Scan_start_MJD'], dtype=float
        )
        scan_start_mjd = np.unique(scan_start_mjd[np.isfinite(scan_start_mjd)])

    scan_starts_utc = pd.to_datetime(
        Time(scan_start_mjd, format='mjd', scale='utc').to_datetime(), utc=True
    )
    scan_starts_awst = scan_starts_utc.tz_convert(AWST_TIMEZONE)
    on_observation_day = scan_starts_awst.date == observation_day
    return list(
        scan_starts_awst[on_observation_day]
        .tz_localize(None)
        .to_pydatetime()
    )


def plot_paf_temperature(
    temps_by_antenna,
    ambient_temps,
    mean_paf_temps,
    scan_start_times,
    observation_day: date,
):
    reported_temps = {
        antenna: temps.dropna(subset=[TEMPERATURE_COLUMN])
        for antenna, temps in temps_by_antenna.items()
    }
    reported_temps = {
        antenna: temps for antenna, temps in reported_temps.items()
        if not temps.empty
    }
    if not reported_temps:
        raise ValueError(
            f'No reported PAF temperatures found on {observation_day.isoformat()}.'
        )
    if ambient_temps.empty:
        raise ValueError(
            f'No ambient temperatures found on {observation_day.isoformat()}.'
        )

    day_start = datetime.combine(observation_day, time.min)
    day_end = day_start + timedelta(days=1)

    fig, ax = plt.subplots(figsize=(7, 3))
    for index, temps in enumerate(reported_temps.values()):
        ax.plot(
            temps['Time'],
            temps[TEMPERATURE_COLUMN],
            linewidth=0.8,
            color=DRY_SAGE,
            alpha=0.5,
            label='PAF temperatures' if index == 0 else None,
        )
    ax.plot(
        mean_paf_temps['Time'],
        mean_paf_temps[TEMPERATURE_COLUMN],
        color=ROYAL_PLUM,
        linewidth=1.8,
        label='Mean PAF temperature',
        zorder=3,
    )
    ax.plot(
        ambient_temps['Time'],
        ambient_temps[TEMPERATURE_COLUMN],
        color=TANGERINE_DREAM,
        linewidth=1.5,
        label='Ambient temperature',
    )
    for index, scan_start in enumerate(scan_start_times):
        ax.axvline(
            scan_start,
            color=STEEL_BLUE,
            linestyle='dotted',
            linewidth=1,
            zorder=-2,
            alpha=0.8,
            label='Scan start' if index == 0 else None,
        )
    ax.set_xlim(day_start, day_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel(f'Time on {observation_day.isoformat()}')
    ax.set_ylabel(r'Temperature ($^\circ$C)')
    ax.legend()
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    temps_by_antenna = load_paf_temps(
        args.paf_dir_path, args.observation_day
    )
    ambient_temps, ambient_paths = load_ambient_temperature(
        args.paf_dir_path, args.observation_day
    )
    mean_paf_temps = compute_catsim_mean_paf_temperature(
        args.paf_dir_path, args.observation_day
    )
    scan_start_times = load_scan_start_times(
        args.catalogue_path, args.observation_day
    )
    for ambient_path in ambient_paths:
        print(f'Using ambient temperature file: {ambient_path}')
    plot_paf_temperature(
        temps_by_antenna,
        ambient_temps,
        mean_paf_temps,
        scan_start_times,
        args.observation_day,
    )


if __name__ == '__main__':
    main()
