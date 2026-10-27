"""Verify the data files listed in required-data.yaml (excluding artifacts)."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def data_files(manifest, root):
    for key, product in manifest.items():
        if 'files' in product:
            sample = 'RACS-mid1' if key == 'racs_mid1_paf_temperatures' else 'RACS-low3'
            directory = root / 'paf_temps' / f'{sample}_PAF_temperature_series'
            for entry in product['files']:
                yield directory / entry['filename'], entry['sha256']
        else:
            directory = root / 'noisemaps' if key.endswith('_noisemap') else root
            yield directory / product['filename'], product.get('modified_sha256', product['sha256'])


def verify(manifest_path, root):
    manifest = yaml.safe_load(manifest_path.read_text())
    verified = {}
    for path, expected in data_files(manifest, root):
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'{path}: expected SHA256 {expected}, got {actual}')
        print(f'Verified: {path}')
        verified[str(path)] = actual
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('required-data.yaml'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    verified = verify(args.manifest, args.data_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(verified, indent=2) + '\n')
