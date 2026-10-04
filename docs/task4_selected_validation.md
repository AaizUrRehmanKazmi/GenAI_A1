# Task 4 selected model validation

Selected checkpoint: epoch 19 (minimum validation L1 in the 30-epoch fresh run).
SHA-256: `94762299fea12f22abb1dbf943f9c8eaebe0178b767a037153fea03440c9d9ba`.

CPU re-evaluation used all 898 unaugmented training pairs and all 160 validation pairs, with equal weighting across three styles. Official test images were not evaluated.

| Split | L1 | SSIM |
|---|---:|---:|
| train | 0.10456845 | 0.50626836 |
| val | 0.10518875 | 0.50645363 |

The CPU metrics reproduce the saved GPU validation metrics within floating-point differences. Similar training/validation scores do not indicate a large generalization gap. The same-photo three-style previews show style-dependent intensity, but weak detail and grid-like artifacts remain. These observations do not establish their cause or prove faithful style transfer. No paired reference exists for the alternative styles of each preview image.

Local evidence is in artifacts/task4-selected-diagnostic/report.json, train_styles.png and val_styles.png. The complete source backup remains in Downloads/task4-selected-full_backup.zip. This is a selected baseline with documented visual limitations, not a claim of satisfactory sketch synthesis.

Next delivery step: export the generator with image and style inputs; verify PyTorch/ONNX parity for all three styles, then integrate the style-conditioned API. Official held-out testing remains pending.
