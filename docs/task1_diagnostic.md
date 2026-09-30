# Fixed clean-image reconstruction diagnostic

Purpose: investigate whether the current vector-bottleneck autoencoder can fit a tiny clean-image problem. This is a training memorization diagnostic, not validation, final evaluation or proof of generalization. Do not compare its scores directly with the full-corruption baseline scores.

The script loads the first 16 entries of the existing seeded training list once, converts them with the established clean preprocessing, and uses the same tensors as input and target for every update. No corruptions, augmentation, validation images or test images are used. It initializes the current architecture from scratch using seed 42 and forces dropout to zero. L1+SSIM weighting remains alpha=0.8 and Adam uses 0.001 by default. It does not change baseline source/config files or load baseline model weights.

One step is one full-batch optimizer update on all 16 images. The default 1,000 updates is a diagnostic budget, not a recommended full-dataset epoch count or a guarantee of sharp reconstruction. The default 60-minute soft limit checks between updates; allow extra time for reporting and saving. Ctrl+C/SIGTERM requests graceful stopping at the next safe boundary. Hard runtime termination loses progress since the most recent checkpoint.

## Run in the existing Colab runtime

Push the new files from your laptop before running this cell. Mount Drive and retain/restore the downloaded Pets dataset as before.

```python
%cd /content/GenAI_A1
!git pull --ff-only
!python -m training.diagnose_task1 --device cuda --steps 1000 --max-minutes 60 --output-dir /content/drive/MyDrive/GenAI_A1/task1-diagnostic
```

This uses the same requirements-training.txt dependencies already installed for the baseline. If the runtime was reset, restore dependencies and data first.

## Outputs

- selected_images.json: the exact 16 relative image paths.
- config.json: architecture with dropout zero, alpha, learning rate and seed.
- provenance.json: source hashes, split hash and hash of the actual clean tensors.
- environment.json: device, PyTorch version, threads and parameter count.
- history.json: post-update loss, L1 and SSIM on those same fixed training images, including step 0.
- step_000000.png, step_000050.png, etc.: target/output/absolute-error grids. Errors use a fixed [0,1] scale, without contrast amplification.
- last.pt: latest resumable state; saved every 25 updates and on graceful exit.
- best.pt: best reconstruction loss among measurement checkpoints, not necessarily every optimizer update.
- mlruns/: a separate local MLflow experiment with parameters, metrics, grids and last checkpoint.

The baseline folder stays unchanged. Fresh runs refuse a non-empty output directory. To compare settings, use separate named directories and preserve selected images and update budgets.

## Resume after an interruption

```python
!python -m training.diagnose_task1 --device cuda --steps 1000 --max-minutes 60 --output-dir /content/drive/MyDrive/GenAI_A1/task1-diagnostic --resume /content/drive/MyDrive/GenAI_A1/task1-diagnostic/last.pt
```

--steps is the total target, not additional steps. You can increase it deliberately when extending this diagnostic, but keep learning rate, selected images, seed, source and model/loss settings unchanged. Saved optimizer and RNG state are restored. Exact CPU same-environment continuation is checked; cross-device numerical equivalence is not guaranteed. There is no LR scheduler, mixed precision or random data augmentation.

## View results

```python
from pathlib import Path
from IPython.display import Image, display
folder = Path('/content/drive/MyDrive/GenAI_A1/task1-diagnostic')
grids = sorted(folder.glob('step_*.png'))
display(Image(filename=str(grids[-1])))
```

If a tiny clean set reconstructs clearly, investigate the larger data/corruption problem next. If it remains blurry, optimization and architecture are both candidates; this test does not prove that the bottleneck alone is responsible. Preserve this diagnostic before implementing a controlled spatial-bottleneck comparison. Do not add unrestricted skip connections just to achieve a high reconstruction score.
