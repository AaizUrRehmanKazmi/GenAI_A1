"""Task 3 differentiable identity + three-expert mixture initialized from Task 2."""
import math
from pathlib import Path
import torch
from torch import nn
from src.models.corruption_classifier import CorruptionClassifier
from src.models.specialist_ae import SpecialistAutoencoder
from exports.task2_bundle import SELECTED, digest

class SoftMoE(nn.Module):
    def __init__(self, gate, salt, blur, occlusion, temperature=1.0):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        self.gate = gate
        self.experts = nn.ModuleList([salt, blur, occlusion])
        parameters = [id(p) for m in [gate, salt, blur, occlusion] for p in m.parameters()]
        if len(parameters) != len(set(parameters)):
            raise ValueError('Gate and experts must not share parameters')
        self.temperature = float(temperature)
        self.warmup = False

    def set_stage(self, stage):
        if stage not in ('warmup', 'joint'):
            raise ValueError('Stage must be warmup or joint')
        self.warmup = stage == 'warmup'
        for expert in self.experts:
            expert.requires_grad_(not self.warmup)
            for parameter in expert.parameters():
                parameter.grad = None
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.warmup:
            for expert in self.experts:
                expert.eval()
        return self

    def forward(self, image):
        logits = self.gate(image)
        weights = (logits / self.temperature).softmax(dim=1)
        branches = torch.stack([image] + [expert(image) for expert in self.experts], dim=1)
        restored = (branches * weights[:, :, None, None, None]).sum(dim=1)
        return {'output': restored, 'logits': logits, 'weights': weights}


def from_task2_bundle(folder, temperature=1.0):
    """Load exact selected PyTorch checkpoints, never ONNX or random initialization."""
    folder = Path(folder)
    models = {}
    for name, (_, _, expected, epoch) in SELECTED.items():
        path = folder / name / 'best.pt'
        if digest(path) != expected:
            raise ValueError(f'{name}: selected checkpoint hash mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['progress']['epoch'] != epoch:
            raise ValueError('Wrong selected epoch')
        if name != 'classifier' and saved['config']['condition'] != name:
            raise ValueError('Wrong expert condition')
        cls = CorruptionClassifier if name == 'classifier' else SpecialistAutoencoder
        model = cls(**saved['config']['model'])
        model.load_state_dict(saved['model'], strict=True)
        models[name] = model
    return SoftMoE(models['classifier'], models['salt'], models['blur'], models['occlusion'], temperature)
