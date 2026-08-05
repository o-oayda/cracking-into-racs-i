import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import gamma
from ciri.artifact_io import load_reference_observation
import numpy as np

from ciri.palette import DRY_SAGE, MUTED_TEAL_DARK, ROSEWOOD, TANGERINE_DREAM_DARK
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
    parser.add_argument("--mid1-artifact", type=Path, required=False)
    parser.add_argument("--low3-artifact", type=Path, required=True)
    return parser.parse_args()


def get_counts_residuals(path):
    x0, _ = load_reference_observation(
        artifact_path=path,
        load_native=True
    )
    finite_x0 = np.asarray(x0)[np.isfinite(x0)]
    x = np.arange(int(finite_x0.min()), int(finite_x0.max()) + 1)
    bin_edges = np.arange(x[0] - 0.5, x[-1] + 1.5)
    observed_counts, _ = np.histogram(
        finite_x0,
        bins=bin_edges,
    )
    lamb = finite_x0.mean()
    observed_probability = observed_counts / observed_counts.sum()
    poisson_probability = poisson.pmf(x, mu=lamb)
    expected_counts = finite_x0.size * poisson_probability
    poisson_residual = np.divide(
        observed_counts - expected_counts,
        np.sqrt(expected_counts),
        out=np.full_like(expected_counts, np.nan),
        where=expected_counts > 0,
    )
    more_x = np.linspace(int(finite_x0.min()), int(finite_x0.max()), 200)
    continuous_extension = np.exp(-lamb) * (lamb ** more_x) / gamma(more_x+1)

    return x, finite_x0, bin_edges, poisson_probability, poisson_residual, (more_x, continuous_extension)


def draw_plots(data_tuple, ax, colors, marker):
    x, finite_x0, bin_edges, poisson_probability, poisson_residual, (more_x, continuous_extension) = data_tuple
    ax[0].hist(
        finite_x0,
        bins=bin_edges,
        alpha=0.2,
        color=colors[0],
        density=True,
    )
    ax[0].hist(
        finite_x0,
        bins=bin_edges,
        histtype="step",
        color=colors[0],
        lw=2,
        density=True,
    )
    # ax[0].scatter(
    #     x,
    #     poisson_probability,
    #     c=colors[1],
    #     marker="o",
    #     s=15,
    #     zorder=10,
    # )
    ax[0].scatter(
        x,
        poisson_probability,
        c=colors[0],
        marker=marker,
        s=10,
        zorder=10
        # linestyle='dotted'
    )
    ax[0].plot(
        more_x,
        continuous_extension,
        linestyle='dotted',
        color=colors[0]
    )

    ax[1].axhline(0, color=DRY_SAGE, lw=1, alpha=0.5, zorder=-10)
    ax[1].plot(
        x,
        poisson_residual,
        color=colors[0],
        marker=marker,
        markersize=3,
        lw=1.5,
    )
    ax[1].set_ylabel(r"Residual ($\sigma$)")
    ax[0].set_ylabel("Probability")
    ax[1].set_xlabel("Pixel count")
    # ax[0].set_yscale("log")
    ax[1].set_ylim(-8, 8)
    ax[0].set_xlim(None, 29)
    ax[1].set_yticks([-5,-2,2,5])
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

def main():
    args = parse_args()
    data_tuple_low3 = get_counts_residuals(args.low3_artifact)
    data_tuple_mid1 = get_counts_residuals(args.mid1_artifact)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.5, 5.25),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1), "hspace": 0.0},
    )

    draw_plots(data_tuple_low3, axes, (MUTED_TEAL_DARK, TANGERINE_DREAM_DARK), marker='o')
    draw_plots(data_tuple_mid1, axes, (ROSEWOOD, TANGERINE_DREAM_DARK), marker='s')
    plt.savefig("output/figx-clustering.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()
