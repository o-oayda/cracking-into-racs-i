"""Verify the downloaded mid1 catalogue, then convert its fluxes to mJy."""

import argparse
import hashlib
import json
from pathlib import Path

from astropy import units as u
from astropy.table import Table
import yaml


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(path, manifest_path):
    product = yaml.safe_load(manifest_path.read_text())['racs_mid1_catalogue']
    actual = sha256(path)
    if actual == product['sha256']:
        table = Table.read(path, format='fits')
        columns = ('Total_flux', 'E_Total_flux')
        for name in columns:
            if table[name].unit != u.Jy:
                raise ValueError(f'{name}: expected Jy, found {table[name].unit}')
        for name in columns:
            table[name].convert_unit_to(u.mJy)
        table.write(path, format='fits', overwrite=True)
        actual = sha256(path)
    if actual != product['modified_sha256']:
        raise ValueError(
            f'{path}: expected converted SHA256 {product["modified_sha256"]}, '
            f'got {actual}; restore the original download and use uv sync --locked'
        )
    print(f'Verified mJy catalogue: {path}')
    return {"sha256": actual}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('required-data.yaml'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.catalogue, args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
