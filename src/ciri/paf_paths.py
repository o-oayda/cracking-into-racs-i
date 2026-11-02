"""Paths to the product-specific PAF temperature series."""
from pathlib import Path


def paf_product_dir(root: str | Path, product: str) -> Path:
    return Path(root).expanduser() / f"RACS-{product}_PAF_temperature_series"
