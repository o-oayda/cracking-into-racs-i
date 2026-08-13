"""Write MID1 and LOW3 log Bayes-factor booktabs table fragments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ciri.artifact_io import load_per_round_log_evidence
from ciri.bayes_factors import (
    BayesFactor,
    ComparisonArtifacts,
    comparison_log_bayes_factor,
    render_log_bayes_table,
    summarize_log_evidence,
    write_log_bayes_json,
)


def nonnegative_integer(value: str) -> int:
    """Argparse converter for positional round counts."""
    try:
        converted = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if converted < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return converted


def validate_comparisons(
    values: list[list[str]] | None,
) -> tuple[ComparisonArtifacts, ...]:
    """Validate CLI comparison triples or flagged quadruples in input order."""
    comparisons = []
    labels = set()
    for index, value in enumerate(values or (), start=1):
        if len(value) not in (3, 4):
            raise ValueError(
                f"Comparison {index} must have LABEL, MID1_ARTIFACT, "
                "LOW3_ARTIFACT, and optionally FLAG"
            )
        label, mid1_artifact, low3_artifact, *optional_flag = value
        flag = optional_flag[0].strip() if optional_flag else ""
        normalized_label = label.strip()
        if not normalized_label:
            raise ValueError(f"Comparison {index} label must not be blank")
        if normalized_label in labels:
            raise ValueError(f"Duplicate comparison label: {label!r}")
        labels.add(normalized_label)
        comparisons.append(
            ComparisonArtifacts(
                label,
                Path(mid1_artifact),
                Path(low3_artifact),
                flag,
            )
        )
    return tuple(comparisons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mid1-fiducial-artifact", type=Path, required=True)
    parser.add_argument("--low3-fiducial-artifact", type=Path, required=True)
    parser.add_argument("--discard-rounds", type=nonnegative_integer, default=4)
    parser.add_argument(
        "--mid1-output",
        type=Path,
        default=Path("output/tablex-log-bayes-mid1.tex"),
    )
    parser.add_argument(
        "--low3-output",
        type=Path,
        default=Path("output/tablex-log-bayes-low3.tex"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--comparison-key",
        action="append",
        metavar="KEY",
        help="Stable JSON key for each --comparison, in the same order.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        nargs="+",
        metavar="VALUE",
        help=(
            "Add LABEL, MID1_ARTIFACT, LOW3_ARTIFACT, and an optional "
            "trusted-LaTeX FLAG."
        ),
    )
    args = parser.parse_args()
    try:
        args.comparisons = validate_comparisons(args.comparison)
        keys = tuple(args.comparison_key or ())
        if keys and len(keys) != len(args.comparisons):
            raise ValueError("Supply exactly one --comparison-key per --comparison")
        if len(set(keys)) != len(keys) or any(not key.strip() for key in keys):
            raise ValueError("Comparison keys must be unique and nonblank")
        args.comparison_keys = keys
    except ValueError as exc:
        parser.error(str(exc))
    return args


def summarized_artifact(path: Path, discard_rounds: int):
    """Load and summarize one artifact's per-round log evidence."""
    return summarize_log_evidence(
        load_per_round_log_evidence(path),
        discard_rounds,
    )


def main() -> None:
    args = parse_args()
    mid1_fiducial = summarized_artifact(
        args.mid1_fiducial_artifact, args.discard_rounds
    )
    low3_fiducial = summarized_artifact(
        args.low3_fiducial_artifact, args.discard_rounds
    )

    mid1_rows: list[BayesFactor] = []
    low3_rows: list[BayesFactor] = []
    for comparison in args.comparisons:
        mid1_rows.append(
            comparison_log_bayes_factor(
                comparison.label,
                summarized_artifact(comparison.mid1_artifact, args.discard_rounds),
                mid1_fiducial,
                comparison.flag,
            )
        )
        low3_rows.append(
            comparison_log_bayes_factor(
                comparison.label,
                summarized_artifact(comparison.low3_artifact, args.discard_rounds),
                low3_fiducial,
                comparison.flag,
            )
        )

    for output, rows in (
        (args.mid1_output, mid1_rows),
        (args.low3_output, low3_rows),
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_log_bayes_table(rows), encoding="utf-8")

    if args.json_output is not None:
        if not args.comparison_keys:
            raise ValueError("--json-output requires one --comparison-key per --comparison")
        write_log_bayes_json(
            args.json_output,
            mid1=dict(zip(args.comparison_keys, mid1_rows, strict=True)),
            low3=dict(zip(args.comparison_keys, low3_rows, strict=True)),
        )


if __name__ == "__main__":
    main()
