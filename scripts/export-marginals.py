"""Export one-sigma weighted posterior marginal summaries as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ciri.artifact_io import load_final_posterior
from ciri.corner import prepare_marginal_samples, weighted_quantiles
from ciri.skymap import SkyFrame


ONE_SIGMA_QUANTILES = (0.158655254, 0.5, 0.841344746)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        nargs=3,
        metavar=("DATASET", "MODEL", "PATH"),
        help="Repeat for every dataset/model posterior to export.",
    )
    parser.add_argument(
        "--coordinates", choices=("galactic", "equatorial"), default="equatorial"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen = set()
    for dataset, model, _ in args.artifact:
        if (dataset, model) in seen:
            parser.error(f"Duplicate dataset/model pair: {dataset}/{model}")
        seen.add((dataset, model))
    return args


def main() -> None:
    args = parse_args()
    output_frame: SkyFrame = args.coordinates
    payload: dict[str, object] = {
        "coordinates": output_frame,
        "quantiles": {"lower": ONE_SIGMA_QUANTILES[0], "median": 0.5, "upper": ONE_SIGMA_QUANTILES[2]},
        "datasets": {},
    }
    datasets: dict[str, dict[str, object]] = payload["datasets"]  # type: ignore[assignment]
    for dataset, model, raw_path in args.artifact:
        posterior = load_final_posterior(Path(raw_path), label=f"{dataset}/{model}")
        parameters, matrix, weights = prepare_marginal_samples(posterior, output_frame)
        summaries = {}
        for index, (name, label) in enumerate(parameters):
            lower, median, upper = weighted_quantiles(
                matrix[:, index], weights, ONE_SIGMA_QUANTILES
            )
            summaries[name] = {
                "label": label,
                "median": float(median),
                "lower_uncertainty": float(median - lower),
                "upper_uncertainty": float(upper - median),
            }
        datasets.setdefault(dataset, {})[model] = summaries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
