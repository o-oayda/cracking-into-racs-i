"""Shared Matplotlib style configuration for figure scripts."""

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


FontFamily = Literal["serif", "sans-serif"]

FONT_FAMILY: FontFamily = "sans-serif"
SKY_GRATICULE_COLOURS = {'dark': '0.3', 'light': '0.7'}


def plot_rc_params(font_family: FontFamily = FONT_FAMILY) -> dict[str, object]:
    """Return the shared Matplotlib settings for ``font_family``."""
    if font_family == "sans-serif":
        return {
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage{amsmath}\usepackage{mathtools}"
                r"\usepackage{sfmath}"
            ),
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans"],
        }
    if font_family == "serif":
        return {
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage{amsmath}\usepackage{mathtools}"
            ),
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
        }
    raise ValueError(
        f"Unsupported font family {font_family!r}; expected 'serif' or "
        "'sans-serif'."
    )


def configure_plot_style(font_family: FontFamily = FONT_FAMILY) -> None:
    """Apply the shared Matplotlib style for all subsequent figures."""
    plt.rcParams.update(plot_rc_params(font_family))


def configure_sky_graticule(
    axes,
    latitude_label_side: Literal['left', 'right'] = 'left',
    *,
    colour: Literal['dark', 'light'] = 'light',
) -> None:
    """Style a Mollweide sky grid with astronomical longitude labels.

    Use 90-degree longitude and 45-degree latitude spacing, with longitude
    labels at latitude +60 degrees and tick text sized like colorbar ticks.
    Call after creating the projection with healpy.projview.
    """
    axes.set_longitude_grid(90)
    axes.set_latitude_grid(45)
    axes.set_longitude_grid_ends(90)
    axes.grid(True, color=SKY_GRATICULE_COLOURS[colour], linestyle=':')
    axes.yaxis.set_major_formatter(FuncFormatter(
        lambda latitude, _: rf'${np.rad2deg(latitude):g}^{{\circ}}$'
    ))
    axes.xaxis.set_major_formatter(FuncFormatter(
        lambda longitude, _: rf'${(-np.rad2deg(longitude)) % 360:g}^{{\circ}}$'
    ))
    axes.tick_params(axis='x', labelbottom=False, labeltop=False, length=0)
    for longitude in axes.get_xticks():
        axes.text(
            longitude,
            np.deg2rad(60),
            axes.xaxis.get_major_formatter()(longitude),
            transform=axes.transData,
            ha='center',
            va='center',
            fontsize=plt.rcParams['xtick.labelsize'],
        )
    axes.tick_params(
        axis='y',
        labelleft=latitude_label_side == 'left',
        labelright=latitude_label_side == 'right',
        labelsize=plt.rcParams['xtick.labelsize'],
        length=0,
    )
