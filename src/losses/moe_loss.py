"""Assignment Task 3 four-term loss. Balance is meaningful on balanced batches."""
import math
import torch
from torch import nn
from src.losses.reconstruction import ReconstructionLoss

class MoELoss(nn.Module):
    def __init__(self, l1_weight=.8, ssim_weight=.2, classification_weight=.1, balance_weight=.01):
        super().__init__()
        self.coefficients = (l1_weight, ssim_weight, classification_weight, balance_weight)
        if any(not math.isfinite(v) or v < 0 for v in self.coefficients) or l1_weight + ssim_weight <= 0:
            raise ValueError('Loss weights must be finite, nonnegative, with reconstruction enabled')
        self.reconstruction = ReconstructionLoss()

    def forward(self, result, target, labels):
        metrics = self.reconstruction(result['output'], target)
        l1 = metrics['l1'].mean()
        structural = (1 - metrics['ssim']).mean()
        # Raw classifier logits supervise corruption; temperature controls mixture only.
        ce = torch.nn.functional.cross_entropy(result['logits'], labels)
        balance = ((result['weights'].mean(0) - .25) ** 2).sum()
        loss = sum(w * term for w, term in zip(self.coefficients, (l1, structural, ce, balance)))
        return {'loss': loss, 'l1': l1, 'ssim': metrics['ssim'].mean(), 'cross_entropy': ce, 'balance': balance}
