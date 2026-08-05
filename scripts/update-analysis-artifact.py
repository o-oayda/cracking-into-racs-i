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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="cached .analysis.zip to pin")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    artifact = args.artifact.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
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
            "require_clean": True,
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
        yaml.YAMLError,
    ) as error:
        raise SystemExit(f"error: invalid artifact manifest: {error}") from None

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("artifacts"), dict):
        raise SystemExit("error: config must contain an artifacts mapping")
    name = entry["experiment_id"]
    config["artifacts"][name] = entry

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config_path.parent, delete=False
    ) as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
        temporary = Path(stream.name)
    temporary.chmod(config_path.stat().st_mode)
    temporary.replace(config_path)
    print(f"Pinned {name} to {stored_path}")


if __name__ == "__main__":
    main()
