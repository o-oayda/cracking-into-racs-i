"""Pin a cached analysis artifact in workflow/config/artifacts.yaml."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "workflow/config/artifacts.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths(path: Path, parser: argparse.ArgumentParser) -> list[Path]:
    """Return one artifact path, or all ZIP files immediately in a directory."""
    path = path.expanduser().resolve(strict=True)
    if path.is_dir():
        artifacts = sorted(
            candidate for candidate in path.glob("*.zip") if candidate.is_file()
        )
        if not artifacts:
            parser.error(f"no ZIP files found in {path}")
        return artifacts
    return [path]


def artifact_entry(
    artifact: Path, parser: argparse.ArgumentParser
) -> tuple[str, dict[str, object]]:
    try:
        stored_path = artifact.relative_to(ROOT).as_posix()
    except ValueError:
        parser.error(f"artifact must be inside {ROOT}")

    try:
        with zipfile.ZipFile(artifact) as archive:
            if archive.namelist().count("artifact-manifest.yaml") != 1:
                raise ValueError("artifact must contain one artifact-manifest.yaml")
            manifest = yaml.safe_load(archive.read("artifact-manifest.yaml"))
        source = manifest["source"]
        entry = {
            "path": stored_path,
            "sha256": sha256(artifact),
            "experiment_id": manifest["experiment_id"],
            "format_version": manifest["format_version"],
            "source_repository": source["repository"],
            "source_commit": source["commit"],
            "require_clean": False,
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        yaml.YAMLError,
    ) as error:
        raise SystemExit(
            f"error: invalid artifact manifest in {artifact}: {error}"
        ) from None
    return entry["experiment_id"], entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        type=Path,
        help="cached .analysis.zip to pin, or a directory containing ZIP files",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="remove all current artifact entries before pinning the supplied artifact(s)",
    )
    args = parser.parse_args()

    artifacts = artifact_paths(args.artifact, parser)
    config_path = args.config.expanduser().resolve(strict=True)
    entries = [artifact_entry(artifact, parser) for artifact in artifacts]

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "artifacts" not in config:
        raise SystemExit("error: config must contain an artifacts mapping")
    if args.clear:
        config["artifacts"] = {}
    elif not isinstance(config["artifacts"], dict):
        raise SystemExit("error: config must contain an artifacts mapping")
    for name, entry in entries:
        config["artifacts"][name] = entry

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config_path.parent, delete=False
    ) as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
        temporary = Path(stream.name)
    temporary.chmod(config_path.stat().st_mode)
    temporary.replace(config_path)
    for name, entry in entries:
        print(f"Pinned {name} to {entry['path']}")


if __name__ == "__main__":
    main()
