"""Shared Matplotlib style configuration for figure scripts."""

from typing import Literal

import matplotlib.pyplot as plt


FontFamily = Literal["serif", "sans-serif"]

FONT_FAMILY: FontFamily = "sans-serif"


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
