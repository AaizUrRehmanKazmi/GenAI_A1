# Alternative Task 4 implementation review

Source: user-provided task4_face2sketch.zip. Inspected all training/data/model/evaluation/export/search scripts; archive contains code only, no trained checkpoints. Original files isolated under artifacts/task4-alternative-review/source; existing implementation unchanged.

## Executed checks

Generator/discriminator forward and backward passed at width 32, style dimension 8. Both learned style embeddings receive gradients. Generator parameters: existing 1,175,451; alternative width32 7,321,723 (roughly 6.2x).

Executed 100 Adam updates on the same fixed 12 training pairs as the previous L1-only diagnostic, same seed, no augmentation/dropout, LR 2e-4, betas (0.5,0.999), full batch12. Alternative receives [-1,1] inputs; outputs mapped to [0,1]; loss 100*L1 on [0,1], matching previous diagnostic. Uses existing bilinear dataset/preprocessing and existing torchmetrics SSIM. Alternative normal initialization and architecture retained. This tests the architecture/initialization package, not one isolated architectural feature. Base32/style8 differ from archive defaults64/16 to bound CPU cost.

| Generator | Training L1 | Existing SSIM |
|---|---:|---:|
| Existing, 100 updates | 0.143747 | 0.345388 |
| Alternative, 100 updates | 0.073354 | 0.579403 |

Alternative fits this small set substantially better, but is larger; equal steps are not equal compute. Outputs remain soft and patterned. No validation improvement or full GAN improvement has been demonstrated. Official test untouched. Results in comparison.json, preview.png; exact harness saved as run_comparison.py in artifact folder.

## Integration caveats

- Architecture: six downsampling levels, BatchNorm, transposed convolutions, bottleneck style projection, tanh. Several simultaneous changes; causal attribution requires ablation.
- Archive SSIM gives 0.632275 on identical final outputs where existing SSIM gives 0.579403. Its zero-padding/boundary convention differs. Retain existing evaluator for comparison.
- Archive uses sklearn stratified split rather than saved manifests; it also uses bicubic resize and crop augmentation. Reuse existing manifests/preprocessing for a fair experiment. No split-overlap count was measured: sklearn unavailable locally.
- Archive training L1 uses [-1,1], twice [0,1] L1. lambda100 therefore doubles reconstruction contribution versus existing lambda100; explicit scale conversion required.
- No optimizer/RNG/epoch resume state or resume CLI; last.pt written only at end. Interrupted cloud runs cannot resume exactly. Preserve existing resumable infrastructure when integrating.
- Discriminator gradients are unnecessarily computed during generator updates, and its BatchNorm buffers update on those passes; review intended policy.
- Full supplied smoke suite not run: torchvision and sklearn unavailable in local environment. Model test did not require either. ONNX export not executed.

Recommendation: integrate alternative as a separate candidate into existing resumable pipeline, preserve old architecture/checkpoints, then compare full validation with same metrics/data. Do not replace production model based on this training-only diagnostic.
