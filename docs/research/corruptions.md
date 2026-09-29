# Image corruption implementation

Status: implemented and tested; dynamic dataset wrapper and validation/test manifests remain pending.

## Contract

`corrupt(clean, seed=42, condition=None)` returns `(input_image, spec)`.
The clean tensor must be CPU float32, shape [3,128,128], values [0,1]. The output is separate storage. The spec is JSON serializable and records the corruption class, numeric label, seed and all relevant settings. Replay with `apply_corruption(clean, spec)`; it does not use the process's global random state.

Class order is fixed: clean=0, salt=1, blur=2, occlusion=3. If condition is None, selection has equal probability. This is not an exactly balanced batch sampler; the Task 2 classifier still needs balanced batching later. A fixed seed repeats a corruption. Future training integration must supply a fresh seed on every load rather than hardcoding 42 for every sample.

## Required settings

Training: salt probability uniform 0.02–0.15; blur kernel selected from 3/5/7 with sigma uniform 0.5–2.5; occlusion uses 1–3 rectangles and requested area uniform 10–35%. Explicit settings support the prescribed three evaluation severities. Clean returns an unchanged copy.

Salt-and-pepper uses one random draw per spatial pixel. Draws below p/2 become black; draws between p/2 and p become white across all RGB channels. The realised proportion varies statistically around p; it is not forced to equal p exactly. Channel-independent noise was not selected because the requirement describes black/white pixels.

Blur uses a normalized discrete 2D Gaussian kernel, a separate convolution per channel and reflection padding. This makes finite kernel size and sigma explicit without adding a torchvision dependency. Reflection padding is a provisional boundary convention: zero padding would introduce dark borders. Constant-field and impulse tests check the implementation; no downstream comparison of padding methods has been made.

Occlusion sampling divides the desired area approximately equally across the sampled number of rectangles, samples width/height aspect ratios from 0.5–2 on a log scale and samples locations uniformly within valid bounds. Rejection sampling avoids overlap. Actual union coverage is measured on a boolean mask at replay time, not inferred by counting black pixels in a potentially dark image. Stored coordinates use [x0,y0,x1,y1], with exclusive right/bottom edges. Quantized area differs by at most 0.005 (half a percentage point) from the requested fraction and must remain in [0.10,0.35]. Placement is bounded and fails explicitly if no valid layout is found.

Non-overlapping masks simplify coverage control. Overlapping masks with union-area correction are a valid alternative, not an experimentally rejected one. Equal approximate mask areas and limited aspect ratios are implementation choices that restrict the shape distribution; revisit them if research supports a broader distribution. Endpoint acceptance can bias rounding slightly inward to preserve the required area bounds.

## Usage

From the repository root in the activated environment:
```python
from src.data.pets_dataset import PetsDataset
from src.data.corruptions import corrupt, apply_corruption

clean = PetsDataset()[0]['image']
noisy, spec = corrupt(clean, seed=42, condition='salt', probability=0.08)
replayed = apply_corruption(clean, spec)
```

```sh
python -m unittest discover -s tests -p test_corruptions.py -v
python -m scripts.check_corruptions
```

The preview uses four training images, each shown clean and at three severities of each corruption. Artifacts are saved under artifacts/data_checks/. Preview specs are demonstration records, not final evaluation manifests. Reproducibility is checked within the installed CPU environment; record library versions with future manifests and experiments.

Requirement source: supplied assignment PDF, corruption definitions. Constants match configs/corruptions.yaml. This module deliberately keeps these required ranges explicit; it does not yet parse YAML. Model training, validation/test manifest generation and runtime dataset corruption are separate future steps.
