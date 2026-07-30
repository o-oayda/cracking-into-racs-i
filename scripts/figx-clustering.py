import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import poisson
from artifact_io import load_reference_observation
import numpy as np

from palette import MUTED_TEAL_DARK, TANGERINE_DREAM_DARK
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


args = parse_args()
x0, _ = load_reference_observation(
    artifact_path=args.low3_artifact,
    load_native=True
)

finite_x0 = np.asarray(x0)[np.isfinite(x0)]
x = np.arange(int(finite_x0.min()), int(finite_x0.max()) + 1)
bin_edges = np.arange(x[0] - 0.5, x[-1] + 1.5)
observed_counts, _ = np.histogram(
    finite_x0,
    bins=bin_edges,
)
observed_probability = observed_counts / observed_counts.sum()
poisson_probability = poisson.pmf(x, mu=finite_x0.mean())
expected_counts = finite_x0.size * poisson_probability
poisson_residual = np.divide(
    observed_counts - expected_counts,
    np.sqrt(expected_counts),
    out=np.full_like(expected_counts, np.nan),
    where=expected_counts > 0,
)

fig, (ax, deviation_ax) = plt.subplots(
    2,
    1,
    figsize=(5.25, 5.75),
    sharex=True,
    gridspec_kw={"height_ratios": (2, 1), "hspace": 0.0},
)
ax.hist(
    finite_x0,
    bins=bin_edges,
    alpha=0.3,
    color=MUTED_TEAL_DARK,
    density=True,
)
ax.hist(
    finite_x0,
    bins=bin_edges,
    histtype="step",
    color=MUTED_TEAL_DARK,
    lw=2,
    density=True,
)
ax.scatter(
    x,
    poisson_probability,
    c=TANGERINE_DREAM_DARK,
    marker="o",
    s=15,
    zorder=10,
)

deviation_ax.axhline(0, color=TANGERINE_DREAM_DARK, lw=1)
deviation_ax.plot(
    x,
    poisson_residual,
    color=MUTED_TEAL_DARK,
    marker="D",
    markersize=3,
    lw=1.5,
)
# deviation_ax.set_ylim(-50,50)
# deviation_ax.set_yscale('symlog')
deviation_ax.set_ylabel(r"Poisson residual ($\sigma$)")
ax.set_ylabel("Probability density")
deviation_ax.set_xlabel("Pixel count")

fig.savefig("output/figx-clustering.pdf", bbox_inches="tight")
