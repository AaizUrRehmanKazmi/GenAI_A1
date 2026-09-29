"""CPU corruptions for RGB float32 tensors in [0,1], with replayable metadata.

No operation changes its input. Rectangle coordinates are [x0,y0,x1,y1],
with exclusive right/bottom edges. Save returned specs in future manifests.
"""
import math
import random

import torch
import torch.nn.functional as F

CLASSES = ('clean', 'salt', 'blur', 'occlusion')


def _check_image(image):
    if (not isinstance(image, torch.Tensor) or image.device.type != 'cpu'
            or image.dtype != torch.float32 or tuple(image.shape) != (3, 128, 128)):
        raise ValueError('Expected a CPU float32 image with shape [3,128,128].')
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Image values must be finite and in [0,1].')


def _rectangles(rng, count, fraction):
    """Place non-overlapping rectangles with varied aspect ratios by rejection.

    Pixel rounding permits <=0.5 percentage-point error in requested coverage.
    All accepted masks still stay inside the required 10–35% range.
    """
    desired = fraction * 128 * 128 / count
    for _ in range(200):
        boxes = []
        for _ in range(count):
            for _ in range(500):
                aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
                width = min(128, max(1, round(math.sqrt(desired * aspect))))
                height = min(128, max(1, round(desired / width)))
                x, y = rng.randint(0, 128 - width), rng.randint(0, 128 - height)
                box = [x, y, x + width, y + height]
                if all(box[2] <= b[0] or b[2] <= box[0] or
                       box[3] <= b[1] or b[3] <= box[1] for b in boxes):
                    boxes.append(box)
                    break
            else:
                break
        actual = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / 16384
        if len(boxes) == count and 0.10 <= actual <= 0.35 and abs(actual - fraction) <= 0.005:
            return boxes, actual
    raise RuntimeError('Unable to place occlusion rectangles within the coverage tolerance.')


def make_spec(condition, *, seed, probability=None, kernel_size=None, sigma=None,
              rectangles=None, area_fraction=None):
    """Sample training settings or supply fixed evaluation settings explicitly.

    Omit condition (pass None) for equal-probability selection of all four classes.
    Always supply a seed; use a fresh seed per load during future training.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63).')
    rng = random.Random(seed)
    condition = rng.choice(CLASSES) if condition is None else condition
    if condition not in CLASSES:
        raise ValueError(f'condition must be one of {CLASSES}')
    spec = {'version': 1, 'condition': condition, 'label': CLASSES.index(condition), 'seed': seed}
    if condition == 'salt':
        p = rng.uniform(0.02, 0.15) if probability is None else probability
        if not 0.02 <= p <= 0.15:
            raise ValueError('Salt probability must be in [0.02,0.15].')
        spec['probability'] = p
    elif condition == 'blur':
        k = rng.choice([3, 5, 7]) if kernel_size is None else kernel_size
        s = rng.uniform(0.5, 2.5) if sigma is None else sigma
        if type(k) is not int or k not in [3, 5, 7] or not 0.5 <= s <= 2.5:
            raise ValueError('Blur needs kernel 3/5/7 and sigma in [0.5,2.5].')
        spec.update(kernel_size=k, sigma=s)
    elif condition == 'occlusion':
        n = rng.randint(1, 3) if rectangles is None else rectangles
        fraction = rng.uniform(0.10, 0.35) if area_fraction is None else area_fraction
        if type(n) is not int or n not in [1, 2, 3] or not 0.10 <= fraction <= 0.35:
            raise ValueError('Occlusion needs 1–3 rectangles and coverage in [0.10,0.35].')
        boxes, actual = _rectangles(rng, n, fraction)
        spec.update(rectangles=boxes, requested_area_fraction=fraction, actual_area_fraction=actual)
    return spec


def apply_corruption(image, spec):
    """Apply a saved spec deterministically, without relying on global RNG state."""
    _check_image(image)
    condition = spec.get('condition')
    # Validate shared fields and scalar parameters through the same public constructor.
    if spec.get('version') != 1 or condition not in CLASSES or spec.get('label') != CLASSES.index(condition):
        raise ValueError('Invalid spec version, condition or class label.')
    seed = spec.get('seed')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Invalid spec seed.')
    output = image.clone()
    if condition == 'salt':
        p = spec['probability']
        make_spec('salt', seed=seed, probability=p)
        generator = torch.Generator().manual_seed(seed)
        draws = torch.rand((128, 128), generator=generator)
        # A single draw per spatial pixel: all RGB channels become black or white.
        output[:, draws < p / 2] = 0
        output[:, (draws >= p / 2) & (draws < p)] = 1
    elif condition == 'blur':
        k, sigma = spec['kernel_size'], spec['sigma']
        make_spec('blur', seed=seed, kernel_size=k, sigma=sigma)
        axis = torch.arange(k, dtype=image.dtype) - k // 2
        kernel = torch.exp(-axis.square() / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        weights = torch.outer(kernel, kernel).expand(3, 1, k, k).contiguous()
        padded = F.pad(image.unsqueeze(0), (k // 2,) * 4, mode='reflect')
        output = F.conv2d(padded, weights, groups=3).squeeze(0).clamp(0, 1)
    elif condition == 'occlusion':
        boxes = spec['rectangles']
        if not isinstance(boxes, list) or not 1 <= len(boxes) <= 3:
            raise ValueError('Expected 1–3 rectangles.')
        mask = torch.zeros((128, 128), dtype=torch.bool)
        for box in boxes:
            if (len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] <= 128 or not 0 <= box[1] < box[3] <= 128):
                raise ValueError('Invalid rectangle coordinates.')
            x0, y0, x1, y1 = box
            mask[y0:y1, x0:x1] = True
        actual = mask.sum().item() / 16384
        requested = spec['requested_area_fraction']
        if (not 0.10 <= actual <= 0.35 or not 0.10 <= requested <= 0.35
                or abs(actual - requested) > 0.005
                or abs(actual - spec['actual_area_fraction']) > 1e-9):
            raise ValueError('Rectangle union area does not match the spec or allowed range.')
        output[:, mask] = 0
    return output


def corrupt(image, *, seed, condition=None, **settings):
    """Return (corrupted image, replayable spec); the clean input is unchanged."""
    spec = make_spec(condition, seed=seed, **settings)
    return apply_corruption(image, spec), spec
