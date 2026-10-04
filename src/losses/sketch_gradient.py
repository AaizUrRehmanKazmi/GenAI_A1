"""Signed adjacent-pixel gradient matching against the target sketch."""
import torch

def sketch_gradient_loss(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4 or min(prediction.shape[-2:]) < 2:
        raise ValueError('Expected matching NCHW tensors with spatial size >=2')
    dx = (prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    return .5 * (dx.abs().mean() + dy.abs().mean())
