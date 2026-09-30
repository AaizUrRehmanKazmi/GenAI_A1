# Task 1 spatial bottleneck experiment

## Motivation and limits
The vector baseline produced smooth validation reconstructions despite fitting 16 familiar training images. The dropout-zero full run improved best validation loss to approximately 0.16782 and SSIM to 0.48461. Tiny-set memorization does not demonstrate generalization. This experiment tests a convolutional spatial representation; it is a hypothesis, not an established fix or a promise of loss below 0.05.

## Architecture
RGB 3×128×128 → four stride-2 convolution/ReLU blocks (16,32,64,128 channels) → 128×8×8 → linear 1×1 convolution → 32×8×8 latent → 1×1 convolution/ReLU → four nearest-upsample/convolution blocks → sigmoid RGB 3×128×128. There are no skip connections, input residuals, or clean-target inputs at inference. Every output depends on the compressed latent.

The model has 203,107 trainable parameters versus 4,397,507 in the vector model. The latent contains 2,048 values versus 49,152 input values (24:1 scalar-count compression). The old vector latent contained 256 values (192:1). This changes latent capacity and parameter count as well as spatial structure, so it is not a pure layout ablation. No pretrained weights are used.

## Controlled settings
Compare against the dropout-zero vector run: same 128px images, seed42, split, corruption sampler, fixed validation cases, Adam LR0.001, batch16, alpha0.8 L1+SSIM, dropout0, and 20 epochs. Equal epochs give the same number of image exposures and updates, not equal wall time. New architecture initialization consumes RNG differently, so individual corruption draws need not match between architectures; validation cases do match. Optuna selection remains future work.

Choose best.pt by condition-balanced validation loss. Compare L1 and SSIM gains against corrupted input for each condition and severity, including clean-image degradation; inspect representative images. Do not select by training loss alone or compare numerical losses with a friend's run without matching definitions and settings. Do not use the official test set for architecture selection.

## Run
Use notebooks/task1_spatial_colab.ipynb after committing and pushing these files. It includes environment setup, a one-epoch smoke check, full training, optional resume, and best-checkpoint validation analysis. Commands from repository root:

```bash
python -m training.train_task1_spatial --device cuda --max-hours 4.5 --output-dir /content/drive/MyDrive/GenAI_A1/task1-spatial
python -m evaluation.evaluate_task1_spatial --device cuda --checkpoint /content/drive/MyDrive/GenAI_A1/task1-spatial/best.pt --output-dir /content/drive/MyDrive/GenAI_A1/task1-spatial/validation_analysis
```

To resume, add `--resume /content/drive/MyDrive/GenAI_A1/task1-spatial/last.pt` to the training command. Keep the config and source revision unchanged. Never reuse vector-run output folders or weights. Preserve the entire new folder including config, signatures, environment, MLflow logs, checkpoints and analysis.

## Implementation isolation
The original model, training loop, loss, data pipeline and evaluator are unchanged so existing checkpoint fingerprints remain valid. A separate spatial run loop shares the original checkpoint, preview and validation helpers, recording both trainer source hashes. The spatial evaluator shares report helpers and verifies spatial-model/data/loss provenance. A later shared-trainer refactor should explicitly version checkpoint compatibility.
