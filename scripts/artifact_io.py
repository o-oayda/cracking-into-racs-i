"""Load plotting inputs directly from a verified RACS analysis artifact."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from anesthetic import Samples, read_csv
import numpy as np
from numpy.typing import NDArray


FINAL_POSTERIOR_PATH = "posterior/final-posterior.csv"
REFERENCE_OBSERVATION_PATH = "observation/reference-observation.npz"
NATIVE_OBSERVATION_PATH = "observation/reference-observation-native.npz"


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
        payload = archive.read(NATIVE_OBSERVATION_PATH)

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
