# Clean pet image preprocessing

Status: provisional implementation choices; no model-quality comparison yet.

The assignment requires RGB images resized to 128 × 128. The clean loader returns a dictionary with `image` (CPU float32 tensor, channels/height/width, [0,1]), `image_id`, and the relative `path`. It reads our fixed split lists without resplitting. Later, a corruption wrapper can create an input from this clean tensor while preserving the target. No breed label is needed for restoration.

## Choices and alternatives

- PyTorch map-style Dataset: integrates with DataLoader batching and loads samples on demand. A preloaded tensor dataset uses more RAM; it has not been benchmarked. The implementation follows the proposed PyTorch stack; no claim of superiority over TensorFlow is made.
- Bilinear resize: explicit, simple baseline. Bicubic is supported via `interpolation='bicubic'` for later comparison; it is not rejected. Nearest-neighbor is typically relevant to categorical masks, which this loader does not resize. Compare visual and validation effects before claiming one interpolation is best.
- Direct square resizing: retains the whole image but distorts aspect ratio. Cropping can discard content; padding introduces borders. Neither alternative has been evaluated here. The assignment specifies output size without mandating an aspect-ratio strategy.
- Apply EXIF orientation before RGB conversion and resizing, to respect stored display orientation. Match this at deployment.
- Scale uint8 RGB values by 255 to [0,1], without ImageNet mean/std normalization. This is an explicit restoration target convention, not a learned choice. Future output activation, SSIM data range, corruption values and inference preprocessing must agree with it.
- Invalid images raise an error with the filename. They are never silently replaced or removed from the fixed split.

## Sources

- PyTorch Dataset/DataLoader tutorial: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial
- Pillow resize API: https://pillow.readthedocs.io/en/stable/reference/Image.html
- Pillow EXIF transpose: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html

## Usage

Run from the repository root in a Python environment with torch, numpy and Pillow:

```python
from torch.utils.data import DataLoader
from src.data.pets_dataset import PetsDataset

train = PetsDataset('data/splits/pets_train.json')
sample = train[0]
print(sample['image'].shape)  # torch.Size([3, 128, 128])
print(sample['image_id'])
loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(batch['image'].shape)  # torch.Size([16, 3, 128, 128])
```

Batch size 16 is illustrative, not a selected hyperparameter. Start with num_workers=0 for debugging; measure worker counts on the training machine later.

```sh
python -m unittest discover -s tests -p test_pets_dataset.py -v
python -m scripts.check_pets_dataset
```

The check scans only training and validation, writes an explicit failure report, and makes a 16-image grid under artifacts/data_checks/. Review this before implementing corruptions. It never edits the original images or split lists.
