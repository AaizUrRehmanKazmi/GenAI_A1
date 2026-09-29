# Dynamic training and fixed evaluation data

TrainingPetsDataset wraps the clean loader and samples a fresh 63-bit seed from the local PyTorch RNG at each access. `corrupt` then selects one of four conditions with equal probability and samples its settings. Seed torch in the training script; PyTorch DataLoader workers have their own torch RNG states. Initially use num_workers=0 for debugging and checkpoint continuation, recording both global torch and DataLoader-generator state. Multi-worker prefetch makes exact mid-epoch continuation more involved and is not promised here.

A sample contains input and target tensors, numeric label, image_id, relative path, and spec_json. Specs are JSON strings so default DataLoader collation works across conditions with different parameter keys and numbers of rectangles. Decode the string with json.loads only when metadata is needed. Losses use input/target; classifier training uses input/label.

EvaluationPetsDataset reads fixed manifests and never samples new corruption settings. Validation uses clean plus all three specified severities of each corruption, providing comparable coverage across tasks. This is more expensive than a single fixed corruption per image; that alternative would reduce validation cost but provide less per-image severity coverage. No model experiment has compared these validation designs. Validation/test seeds are separated. Test manifests are generated from filenames without inspecting test pixels.

Training conditional selection is probabilistically balanced, not exactly balanced per batch. Task 2 still needs a balanced batch design. Passing condition='salt', 'blur' or 'occlusion' restricts the wrapper to a specialist's corruption type. The base PetsDataset remains clean-only.

## Run checks
```sh
python -m unittest discover -s tests -p test_restoration_data.py -v
python -m scripts.generate_manifests
python -m scripts.check_restoration_data
```

## Training batch example
```python
import torch
from torch.utils.data import DataLoader
from src.data.pets_dataset import TrainingPetsDataset

torch.manual_seed(42)
loader = DataLoader(TrainingPetsDataset(), batch_size=16, shuffle=True,
                    num_workers=0, generator=torch.Generator().manual_seed(42))
batch = next(iter(loader))
inputs, targets, labels = batch['input'], batch['target'], batch['label']
```

The next component is Task 1's autoencoder, loss and training loop. No model or training loop is implemented by this milestone.
