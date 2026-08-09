import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
from getdist import MCSamples, plots
import numpy as np
from ciri.artifact_io import load_final_posterior
from ciri.corner import corner_parameters, prepare_corner_samples
from ciri.skymap import SkyFrame
from paperplot import style_name
from ciri.palette import MUTED_TEAL, ROSEWOOD, MUTED_TEAL_DARK, TANGERINE_DREAM_DARK


plots.set_active_style(style_name)
import matplotlib.pyplot as plt
plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)


SAMPLE_COLORS = (ROSEWOOD, MUTED_TEAL_DARK) #, TANGERINE_DREAM_DARK)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid1-artifact", type=Path, required=True)
    parser.add_argument("--low3-artifact", type=Path, required=True)
    parser.add_argument("--low2-artifact", type=Path, required=False)
    parser.add_argument(
        "--coordinates",
        choices=("galactic", "equatorial"),
        default="galactic",
        help="Coordinates shown on the output map (default: galactic).",
    )
    parser.add_argument(
        "--exclude-log10-n-initial-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude log10_n_initial_samples from the triangle plot.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def getdist_samples(
    artifact: Path,
    label: str,
    output_frame: SkyFrame,
) -> MCSamples:
    nested = load_final_posterior(artifact, label=label)
    parameters = corner_parameters(output_frame)
    names = [name for name, _ in parameters]
    labels = [parameter_label for _, parameter_label in parameters]
    matrix, weights = prepare_corner_samples(nested, output_frame)

    # Match dipolesbi.tools.posterior_cli: preserve the nested-sampling weights
    # when converting the anesthetic samples into a GetDist MCSamples object.
    with redirect_stdout(io.StringIO()):
        return MCSamples(
            samples=matrix,
            weights=weights,
            sampler="nested",
            names=names,
            labels=labels,
            label=label,
            settings={"ignore_rows": 0.0},
        )


def stack_marginal_titles(
    plotter: plots.GetDistPlotter,
    samples: list[MCSamples],
    parameters: tuple[tuple[str, str], ...],
) -> None:
    """Show one color-matched credible-interval title per sample set."""
    for plot_index, (param_name, param_label) in enumerate(parameters):
        axis = plotter.subplots[plot_index, plot_index]
        for location in ("left", "center", "right"):
            axis.set_title("", loc=location)

        fontsize = plotter._scaled_fontsize(
            plotter.settings.title_limit_fontsize,
            plotter.settings.axes_fontsize,
        )
        for sample_index, (sample, color) in enumerate(
            zip(samples, SAMPLE_COLORS, strict=True)
        ):
            param_index = sample.paramNames.numberOfName(param_name)
            values = np.asarray(sample.samples[:, param_index], dtype=np.float64)
            weights = np.asarray(sample.weights, dtype=np.float64)
            lower, median, upper = plotter._weighted_quantiles(
                values,
                weights,
                [0.158655254, 0.5, 0.841344746],
            )
            interval = plotter._format_interval(median, lower, upper)
            axis.text(
                plotter.marginal_title_x,
                1.03 + sample_index * 0.19,
                rf"${param_label} = {interval}$",
                color=color,
                fontsize=fontsize,
                ha="left",
                va="bottom",
                transform=axis.transAxes,
                clip_on=False,
            )


def main() -> None:
    args = parse_args()
    output_frame: SkyFrame = args.coordinates
    samples = [
        getdist_samples(args.mid1_artifact, "RACS-mid1", output_frame),
        getdist_samples(args.low3_artifact, "RACS-low3", output_frame),
    ]
    if args.low2_artifact is not None:
        samples.append(
            getdist_samples(args.low2_artifact, "RACS-low2", output_frame)
        )

    all_parameters = corner_parameters(output_frame)
    parameters = (
        all_parameters[1:]
        if args.exclude_log10_n_initial_samples
        else all_parameters
    )
    names = [name for name, _ in parameters]

    plotter = plots.get_subplot_plotter(width_inch=8)
    plotter.settings.progress = False
    plotter.triangle_plot(
        samples,
        params=names,
        filled=True,
        contour_colors=SAMPLE_COLORS,
        legend_labels=[],
    )
    stack_marginal_titles(plotter, samples, parameters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotter.fig.savefig(str(args.output), bbox_inches="tight")


if __name__ == "__main__":
    main()
