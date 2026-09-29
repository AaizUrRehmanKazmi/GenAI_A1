"""Preview fixed severities on four training images; no test images are read.

Run: python -m scripts.check_corruptions
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from src.data.pets_dataset import PetsDataset, REPO_ROOT
from src.data.corruptions import corrupt


def main():
    torch.set_num_threads(1)
    dataset = PetsDataset()
    settings = [('Clean', 'clean', {})]
    settings += [(f'Salt p={p}', 'salt', {'probability': p}) for p in [.03, .08, .15]]
    settings += [(f'Blur {k}/{s}', 'blur', {'kernel_size': k, 'sigma': s})
                 for k, s in [(3, .7), (5, 1.5), (7, 2.5)]]
    settings += [(f'Masks {n}/{int(a*100)}%', 'occlusion', {'rectangles': n, 'area_fraction': a})
                 for n, a in [(1, .10), (2, .20), (3, .35)]]
    canvas = Image.new('RGB', (140 * len(settings), 170 * 4), 'white')
    draw, records = ImageDraw.Draw(canvas), []
    for row in range(4):
        sample = dataset[row]
        for column, (title, condition, options) in enumerate(settings):
            output, spec = corrupt(sample['image'], seed=42 + row * 10 + column,
                                   condition=condition, **options)
            pixels = (output.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            x, y = column * 140, row * 170
            draw.text((x, y), title, fill='black')
            canvas.paste(Image.fromarray(pixels), (x, y + 18))
            if column == 0:
                draw.text((x, y + 149), sample['image_id'][:23], fill='black')
            if condition == 'occlusion':
                draw.text((x, y + 149), f"Actual: {spec['actual_area_fraction']:.1%}", fill='black')
            records.append({'image_id': sample['image_id'], 'path': sample['path'], 'spec': spec})
    folder = REPO_ROOT / 'artifacts/data_checks'
    folder.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / 'corruptions_grid.png')
    (folder / 'corruptions_preview_specs.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'Saved {folder / "corruptions_grid.png"}')
    print('Preview specs use training images only; these are not evaluation manifests.')


if __name__ == '__main__':
    main()
