# Task 4 baseline diagnostic

Inspected task4-baseline_continued.zip; selected checkpoint is epoch 14 by validation L1. Reproduced its validation metrics locally on CPU. Full unaugmented training set: 898 pairs; validation: 160 pairs. No official test images read.

| Split | Equal-style L1 | Equal-style SSIM |
|---|---:|---:|
| Train | 0.110477 | 0.499833 |
| Validation | 0.108415 | 0.504349 |

There is no large train/validation performance gap in this diagnostic. This supports investigating fitting/optimization limitations rather than attributing the visible poor output solely to overfitting; it does not establish a root cause. Same-photo, three-style previews show conditioning affects output, notably intensity, but do not prove faithful style transfer. Style 2 has the largest validation error (L1 0.147102, SSIM 0.417887). All styles exhibit artifacts and weak detail.

Code inspection: generator uses nearest-neighbor resize plus convolution, not transposed convolution. BCE operates on raw discriminator logits. Fake images are detached for discriminator updates and discriminator parameters frozen for generator updates. Existing model/gradient-isolation tests pass. No source used in checkpoint fingerprints was changed.

Reproduce from repository root:

```sh
python -m scripts.diagnose_task4 --checkpoint artifacts/task4-baseline-diagnostic/best.pt --output-dir artifacts/task4-baseline-diagnostic
```

Outputs: report.json, train_styles.png, val_styles.png. Preview targets are valid only for each image's annotated style; alternative-style outputs are qualitative probes.

Next: preserve baseline, then run controlled tuning/diagnostics in separate output folders. Do not overwrite or resume this checkpoint with changed architecture/configuration. More baseline epochs alone are not justified by the observed plateau. Final model selection and official test evaluation remain outstanding.
