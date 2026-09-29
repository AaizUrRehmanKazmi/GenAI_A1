"""Clean Oxford-IIIT Pet targets for Tasks 1–3.

Loads our saved JSON path lists without downloading, splitting or corrupting data.
Each sample is {image: float32 CHW tensor in [0, 1], image_id: str, path: str}.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
RESAMPLING = {
    'bilinear': Image.Resampling.BILINEAR,
    'bicubic': Image.Resampling.BICUBIC,
}


class PetsDataset(Dataset):
    """Read one clean image at a time; defaults use the repository's training split.

    ``raw_dir`` contains images/; entries in ``split_file`` are relative to it.
    Image orientation follows EXIF metadata. Direct square resizing preserves
    the whole frame but can change aspect ratio. Original files are never saved.
    """

    def __init__(self, split_file=None, raw_dir=None, *, interpolation='bilinear'):
        self.raw_dir = Path(raw_dir or REPO_ROOT / 'data/raw/oxford_pets').resolve()
        self.split_file = Path(split_file or REPO_ROOT / 'data/splits/pets_train.json')
        if interpolation not in RESAMPLING:
            raise ValueError(f'interpolation must be one of {tuple(RESAMPLING)}')
        self.interpolation = interpolation
        entries = json.loads(self.split_file.read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError('Split must be a non-empty JSON list of relative image paths.')
        self.paths = []
        resolved_seen = set()
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError('Each split entry must be a non-empty path string.')
            relative = Path(entry)
            absolute = (self.raw_dir / relative).resolve()
            if relative.is_absolute() or not absolute.is_relative_to(self.raw_dir):
                raise ValueError(f'Image path escapes dataset root: {entry}')
            if absolute in resolved_seen:
                raise ValueError(f'Duplicate image path: {entry}')
            if not absolute.is_file():
                raise FileNotFoundError(f'Image listed in {self.split_file} is missing: {absolute}')
            resolved_seen.add(absolute)
            self.paths.append(relative.as_posix())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        relative = self.paths[index]
        path = self.raw_dir / relative
        try:
            with Image.open(path) as source:
                rgb = ImageOps.exif_transpose(source).convert('RGB')
                resized = rgb.resize((128, 128), resample=RESAMPLING[self.interpolation])
                pixels = np.array(resized, dtype=np.float32, copy=True) / 255.0
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise RuntimeError(f'Cannot decode/preprocess image {path}: {exc}') from exc
        image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        return {'image': image, 'image_id': Path(relative).stem, 'path': relative}


class TrainingPetsDataset(Dataset):
    """Fresh corruption per access, with clean target and class label.

    Uses the worker-local PyTorch RNG. Seed torch and the DataLoader generator in
    the training script. Exact checkpoint continuation initially uses num_workers=0.
    A fixed condition supports independent specialist training later.
    """

    def __init__(self, split_file=None, raw_dir=None, *, condition=None):
        from .corruptions import CLASSES
        if condition is not None and condition not in CLASSES:
            raise ValueError(f'condition must be None or one of {CLASSES}')
        self.clean_dataset = PetsDataset(split_file, raw_dir)
        self.condition = condition

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, index):
        from .corruptions import corrupt
        sample = self.clean_dataset[index]
        seed = torch.randint(0, 2**63 - 1, ()).item()
        image, spec = corrupt(sample['image'], seed=seed, condition=self.condition)
        return {'input': image, 'target': sample['image'], 'label': spec['label'],
                'image_id': sample['image_id'], 'path': sample['path'],
                'spec_json': json.dumps(spec)}


class EvaluationPetsDataset(Dataset):
    """Replay a verified fixed manifest. Each image has ten evaluation cases."""

    def __init__(self, split_file, manifest_file, raw_dir=None):
        from .manifests import load_manifest
        self.manifest = load_manifest(manifest_file, split_file)
        self.clean_dataset = PetsDataset(split_file, raw_dir)
        self.indices = {path: index for index, path in enumerate(self.clean_dataset.paths)}

    def __len__(self):
        return len(self.manifest['records'])

    def __getitem__(self, index):
        from .corruptions import apply_corruption
        record = self.manifest['records'][index]
        sample = self.clean_dataset[self.indices[record['path']]]
        spec = record['spec']
        return {'input': apply_corruption(sample['image'], spec), 'target': sample['image'],
                'label': spec['label'], 'image_id': sample['image_id'], 'path': sample['path'],
                'severity': record['severity'], 'spec_json': json.dumps(spec)}
