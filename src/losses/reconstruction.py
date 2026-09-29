"""Per-image L1 and differentiable SSIM, with explicit [0,1] data range."""
import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure


class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1].')
        self.alpha = float(alpha)

    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError('Expected matching NCHW image batches.')
        l1 = (prediction - target).abs().mean(dim=(1, 2, 3))
        ssim = structural_similarity_index_measure(
            prediction, target, data_range=1.0, gaussian_kernel=True,
            sigma=1.5, kernel_size=11, reduction='none')
        loss = self.alpha * l1 + (1 - self.alpha) * (1 - ssim)
        return {'loss': loss, 'l1': l1, 'ssim': ssim}
