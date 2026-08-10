"""Custom GetDist plotting style copied from ``dipolesbi.style.paperplot``."""

from __future__ import annotations

import math
import warnings

from getdist import plots
import numpy as np

from ciri.plot_style import plot_rc_params


class PaperPlotter(plots.GetDistPlotter):
    """GetDist style for publication plots."""

    marginal_title_x = 0.06

    _style_rc = plot_rc_params()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings.linewidth = 3.0
        self.settings.linewidth_contour = 3.0
        self.settings.title_limit = 1
        self.settings.title_limit_labels = True
        self.settings.title_limit_fontsize = 1.03 * self.settings.axes_labelsize

    @staticmethod
    def _weighted_quantiles(
        values: np.ndarray,
        weights: np.ndarray | None,
        quantiles: list[float],
    ) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        qs = np.asarray(quantiles, dtype=np.float64)
        if data.size == 0:
            return np.full_like(qs, np.nan, dtype=np.float64)
        if weights is None:
            return np.quantile(data, qs)

        sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if sample_weights.size != data.size:
            return np.quantile(data, qs)
        mask = np.isfinite(data) & np.isfinite(sample_weights) & (sample_weights > 0)
        if not np.any(mask):
            return np.quantile(data, qs)

        data = data[mask]
        sample_weights = sample_weights[mask]
        sorter = np.argsort(data)
        data = data[sorter]
        sample_weights = sample_weights[sorter]
        cumulative = np.cumsum(sample_weights) - 0.5 * sample_weights
        total = cumulative[-1] + 0.5 * sample_weights[-1]
        if total <= 0:
            return np.quantile(data, qs)
        cumulative /= total
        return np.interp(qs, cumulative, data, left=data[0], right=data[-1])

    @staticmethod
    def _round_to_sig(value: float, sig: int) -> tuple[float, int]:
        if value == 0:
            return 0.0, 0
        exponent = int(math.floor(math.log10(abs(value))))
        round_digits = sig - 1 - exponent
        return round(value, round_digits), max(round_digits, 0)

    @staticmethod
    def _format_fixed(value: float, decimals: int) -> str:
        if decimals <= 0:
            return f"{int(round(value))}"
        return f"{value:.{decimals}f}"

    def _format_interval(self, median: float, lower: float, upper: float) -> str:
        plus_val, plus_dec = self._round_to_sig(abs(upper - median), 2)
        minus_val, minus_dec = self._round_to_sig(abs(median - lower), 2)
        common_dec = min(plus_dec, minus_dec)

        plus_str = self._format_fixed(round(plus_val, common_dec), common_dec)
        minus_str = self._format_fixed(round(minus_val, common_dec), common_dec)
        median_str = self._format_fixed(round(median, common_dec), common_dec)
        return rf"{median_str}^{{+{plus_str}}}_{{-{minus_str}}}"

    def add_1d(
        self,
        root,
        param,
        plotno=0,
        normalized=None,
        ax=None,
        title_limit=None,
        **kwargs,
    ):
        param_info = self._check_param(root, param)
        result = super().add_1d(
            root,
            param_info,
            plotno=plotno,
            normalized=normalized,
            ax=ax,
            # The weighted title below replaces GetDist's built-in title.
            title_limit=0,
            **kwargs,
        )

        try:
            samples = self.sample_analyser.samples_for_root(root)
        except Exception:
            return result
        if samples is None or not hasattr(samples, "getMargeStats"):
            return result

        stats = samples.getMargeStats()
        if stats is None or stats.parWithName(param_info.name) is None:
            return result
        try:
            param_index = samples.paramNames.numberOfName(param_info.name)
        except Exception:
            return result
        if param_index is None or param_index < 0:
            return result

        sample_array = getattr(samples, "samples", None)
        if sample_array is None:
            return result
        try:
            param_samples = np.asarray(
                sample_array[:, param_index], dtype=np.float64
            )
        except Exception:
            return result

        weights = getattr(samples, "weights", None)
        q_lower, q_median, q_upper = self._weighted_quantiles(
            param_samples,
            weights,
            [0.158655254, 0.5, 0.841344746],
        )
        if not all(np.isfinite((q_lower, q_median, q_upper))):
            warnings.warn(
                "paperplot: unable to determine median/credible interval for "
                f"{param_info.name!r}; skipping markers."
            )
            return result

        axis = self.get_axes(ax, pars=(param_info,))
        line_kwargs = (
            self.lines_added.get(plotno, {})
            if hasattr(self, "lines_added")
            else {}
        )
        # color = line_kwargs.get("color") or self.settings.axis_marker_color
        # width = self._scaled_linewidth(self.settings.linewidth_contour)
        # axis.axvline(q_median, color=color, ls="--", lw=width, alpha=0.8)
        # axis.axvline(q_lower, color=color, ls="--", lw=width, alpha=0.6)
        # axis.axvline(q_upper, color=color, ls="--", lw=width, alpha=0.6)

        label = param_info.label or param_info.name
        title = axis.set_title(
            rf"${label} = {self._format_interval(q_median, q_lower, q_upper)}$",
            loc="left",
            fontsize=self._scaled_fontsize(
                self.settings.title_limit_fontsize,
                self.settings.axes_fontsize,
            ),
        )
        title.set_x(self.marginal_title_x)
        title.set_horizontalalignment("left")
        return result


style_name = "paperplot"
plots.add_plotter_style(style_name, PaperPlotter)
