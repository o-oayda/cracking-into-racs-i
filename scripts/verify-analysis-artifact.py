"""Verify a pinned RACS analysis artifact and record its provenance."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import zipfile

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.artifact.expanduser().resolve(strict=True)
    actual_digest = sha256(artifact)
    expected_digest = args.expected_sha256.lower()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
        )

    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        if names.count("artifact-manifest.yaml") != 1:
            raise ValueError(
                "Artifact must contain exactly one artifact-manifest.yaml"
            )
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt ZIP member: {corrupt_member}")
        manifest = yaml.safe_load(archive.read("artifact-manifest.yaml"))

    if not isinstance(manifest, dict):
        raise ValueError("Artifact manifest must be a YAML mapping")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Artifact manifest has no source provenance mapping")

    expected = {
        "format_version": args.expected_format_version,
        "experiment_id": args.expected_experiment_id,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Unexpected {field}: expected {value!r}, got {manifest.get(field)!r}"
            )

    source_expected = {
        "repository": args.expected_source_repository,
        "commit": args.expected_source_commit,
    }
    for field, value in source_expected.items():
        if source.get(field) != value:
            raise ValueError(
                f"Unexpected source {field}: expected {value!r}, "
                f"got {source.get(field)!r}"
            )

    if args.require_clean and (
        source.get("dirty") is not False or source.get("reproducible") is not True
    ):
        raise ValueError(
            "Artifact was not produced from a clean, reproducible source"
        )

    return {
        "verification_format_version": 1,
        "verified": True,
        "artifact": artifact.name,
        "size": artifact.stat().st_size,
        "sha256": actual_digest,
        "artifact_format_version": manifest["format_version"],
        "experiment_id": manifest["experiment_id"],
        "source": source,
        "implementation_fingerprint": manifest.get("implementation_fingerprint"),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-experiment-id", required=True)
    parser.add_argument("--expected-format-version", type=int, required=True)
    parser.add_argument("--expected-source-repository", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = verify(args)
        write_report(args.report, report)
    except (OSError, ValueError, zipfile.BadZipFile, yaml.YAMLError) as error:
        raise SystemExit(f"error: {error}") from None
    print(f"Verified artifact: {args.artifact}")
    print(f"Verification report: {args.report}")


if __name__ == "__main__":
    main()
