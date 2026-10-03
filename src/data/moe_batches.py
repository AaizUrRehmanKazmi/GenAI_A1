"""Exactly balanced Task 3 batches with aligned clean targets."""
import torch
from .corruptions import CLASSES, corrupt

def balanced_restoration_batch(dataset, indices):
    inputs, targets, labels = [], [], []
    for index in indices:
        clean = dataset[index]['image']
        for label, condition in enumerate(CLASSES):
            seed = torch.randint(0, 2**63 - 1, ()).item()
            image, _ = corrupt(clean, seed=seed, condition=condition)
            inputs.append(image);targets.append(clean);labels.append(label)
    if not inputs:
        raise ValueError('At least one source image is required')
    order = torch.randperm(len(inputs))
    return torch.stack(inputs)[order], torch.stack(targets)[order], torch.tensor(labels)[order]
