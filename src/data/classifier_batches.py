"""Exactly balanced batches: each selected clean image produces all four classes."""
import torch
from .corruptions import CLASSES, corrupt

def balanced_batch(dataset, indices):
    images=[]; labels=[]
    for index in indices:
        clean=dataset[index]['image']
        for label,condition in enumerate(CLASSES):
            seed=torch.randint(0,2**63-1,()).item()
            image,_=corrupt(clean,seed=seed,condition=condition)
            images.append(image); labels.append(label)
    order=torch.randperm(len(images))
    return torch.stack(images)[order],torch.tensor(labels)[order]
