"""Generate fixed evaluation specifications without decoding test images.

From the repository root: python -m scripts.generate_manifests
"""
import argparse
from pathlib import Path
from src.data.manifests import generate_manifest, save_manifest
from src.data.pets_dataset import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits-dir', type=Path, default=REPO_ROOT / 'data/splits')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'data/manifests')
    args = parser.parse_args()
    for split in ('val', 'test'):
        manifest = generate_manifest(args.splits_dir / f'pets_{split}.json', split=split)
        destination = args.output_dir / f'pets_{split}_corruptions.json'
        save_manifest(manifest, destination)
        print(f"{split}: {manifest['image_count']} images, {len(manifest['records'])} cases → {destination}", flush=True)
    print('Only path lists were read. No test image pixels were loaded.')


if __name__ == '__main__':
    main()
