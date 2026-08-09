"""Load plotting inputs directly from a verified RACS analysis artifact."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from anesthetic import Samples, read_csv
from catsim import RacsConfig
import numpy as np
from numpy.typing import NDArray


FINAL_POSTERIOR_PATH = "posterior/final-posterior.csv"
MODEL_CONFIG_PATH = "provenance/model_config.json"
REFERENCE_OBSERVATION_PATH = "observation/reference-observation.npz"
NATIVE_OBSERVATION_PATH = "observation/reference-observation-native.npz"
PER_ROUND_LOG_EVIDENCE_PATH = "posterior/per-round-log-evidence.npy"


def list_artifact_files(artifact_path: str | Path) -> tuple[str, ...]:
    """Return the archive paths contained in an analysis artifact."""
    with ZipFile(Path(artifact_path).expanduser()) as archive:
        return tuple(info.filename for info in archive.infolist())


def load_final_posterior(
    artifact_path: str | Path,
    *,
    label: str | None = None,
) -> Samples:
    """Load the final-round weighted posterior as anesthetic samples."""
    path = Path(artifact_path).expanduser()
    sample_label = label if label is not None else path.name.removesuffix(".analysis.zip")
    with ZipFile(path) as archive:
        with archive.open(FINAL_POSTERIOR_PATH) as stream:
            return read_csv(stream, label=sample_label)


def load_per_round_log_evidence(
    artifact_path: str | Path,
) -> NDArray[np.float64]:
    """Load and validate per-round log evidence and stored estimator errors.

    The artifact member has shape ``(2, n_rounds)``: row zero contains log
    evidences and row one contains the estimator errors stored by the analysis.
    """
    path = Path(artifact_path).expanduser()
    with ZipFile(path) as archive:
        try:
            payload = archive.read(PER_ROUND_LOG_EVIDENCE_PATH)
        except KeyError as exc:
            raise ValueError(
                f"Artifact is missing {PER_ROUND_LOG_EVIDENCE_PATH}"
            ) from exc

    try:
        evidence = np.load(BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Artifact member {PER_ROUND_LOG_EVIDENCE_PATH} is not a valid NumPy array"
        ) from exc

    if (
        not np.issubdtype(evidence.dtype, np.number)
        or np.issubdtype(evidence.dtype, np.complexfloating)
    ):
        raise ValueError("Per-round log evidence must be a real numeric array")
    if evidence.ndim != 2 or evidence.shape[0] != 2:
        raise ValueError(
            "Per-round log evidence must have shape (2, n_rounds); "
            f"got {evidence.shape}"
        )

    result = np.asarray(evidence, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("Per-round log evidence and stored errors must be finite")
    if np.any(result[1] < 0.0):
        raise ValueError("Stored per-round log-evidence errors must be nonnegative")
    return result


def load_reference_observation(
    artifact_path: str | Path,
    load_native: bool = False
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Load copies of the reference data vector and its mask."""
    if load_native:
        ref_path = NATIVE_OBSERVATION_PATH
    else:
        ref_path = REFERENCE_OBSERVATION_PATH

    with ZipFile(Path(artifact_path).expanduser()) as archive:
        payload = archive.read(ref_path)

    with np.load(BytesIO(payload), allow_pickle=False) as observation:
        missing = {"x0", "mask"} - set(observation.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Reference observation is missing array(s): {names}")
        x0 = np.asarray(observation["x0"], dtype=np.float32).copy()
        mask = np.asarray(observation["mask"], dtype=np.bool_).copy()

    if x0.ndim != 1 or mask.ndim != 1 or x0.shape != mask.shape:
        raise ValueError(
            "Reference observation x0 and mask must be equally sized vectors; "
            f"got {x0.shape} and {mask.shape}"
        )
    return x0, mask


def load_model_config(
    artifact_path: str | Path
) -> RacsConfig:
    """Load the serialized simulator configuration from an artifact."""
    path = Path(artifact_path).expanduser()
    with ZipFile(path) as archive:
        with archive.open(MODEL_CONFIG_PATH) as stream:
            serialized_config = json.load(stream)

    return RacsConfig(**serialized_config["fields"])
