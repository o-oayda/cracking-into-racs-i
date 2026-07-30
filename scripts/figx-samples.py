from pathlib import Path
from artifact_io import load_reference_observation
import argparse
import healpy as hp
import matplotlib.pyplot as plt
from ciri.plotting import smooth_map
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
x0, mask = load_reference_observation(
    artifact_path=args.low3_artifact,
    load_native=True
)
smooth_map(x0, cmap='magma', unit='Smoothed count per pixel')
fig = plt.gcf()
fig.set_size_inches((4,3))
plt.show()

