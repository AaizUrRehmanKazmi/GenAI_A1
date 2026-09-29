"""Decode train/validation images, check the tensor contract and save a sample grid.

Run from the repository root: python -m scripts.check_pets_dataset
The official test split is deliberately not inspected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.data.pets_dataset import PetsDataset, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts/data_checks')
    args = parser.parse_args()
    # Small CPU tensors do not benefit from a large thread pool here.
    torch.set_num_threads(1)
    results, thumbnails = {}, []
    for split in ['train', 'val']:
        dataset = PetsDataset(REPO_ROOT / f'data/splits/pets_{split}.json')
        failures = []
        for index in range(len(dataset)):
            try:
                sample = dataset[index]
                image = sample['image']
                if (image.shape != (3, 128, 128) or image.dtype != torch.float32
                        or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1):
                    raise ValueError('Invalid tensor shape, dtype or pixel values')
                if split == 'train' and index < 16:
                    pixels = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    thumbnails.append((Image.fromarray(pixels), sample['image_id']))
            except (RuntimeError, ValueError) as exc:
                failures.append({'index': index, 'path': dataset.paths[index], 'error': str(exc)})
        results[split] = {'total': len(dataset), 'passed': len(dataset) - len(failures), 'failures': failures}
        print(f"{split}: {results[split]['passed']}/{len(dataset)} passed", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
    canvas = Image.new('RGB', (4 * 192, 4 * 156), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(thumbnails):
        x, y = (i % 4) * 192, (i // 4) * 156
        canvas.paste(im, (x + 32, y))
        draw.text((x + 4, y + 132), label, fill='black')
    canvas.save(args.output_dir / 'train_grid.png')
    if any(result['failures'] for result in results.values()):
        raise SystemExit('Image failures found; inspect report.json. No split entries were removed.')


if __name__ == '__main__':
    main()
