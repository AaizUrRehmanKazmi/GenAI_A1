# Full reconstruction fitting review

Evaluated selected reconstruction epoch14 using existing diagnose_task4.py on all898 train and160 validation pairs, no augmentation, generator eval mode, equal-style metrics. No official test used.

| Split | L1 | SSIM |
|---|---:|---:|
| Train | 0.08013179 | 0.56682378 |
| Validation | 0.08853802 | 0.55807106 |

Training previews remain blurred on the same example identities that the fixed12 diagnostic reproduced sharply. Full-run validation is slightly worse, but low training quality shows this is not solely a held-out generalization failure. Small-set fitting does not prove sufficient capacity for all898 pairs, nor identify dropout as cause. Dataset size, exposures, dropout and augmentation differ between these experiments.

Next controlled proposal: compare full-dataset reconstruction with dropout0 versus0.2, same initialization, fixed budget, data augmentation and optimizer; assess both train and validation to distinguish improved fitting from overfitting. Do not change losses, augmentation, and capacity simultaneously. Existing best models remain reference checkpoints.

Artifacts: artifacts/task4-reconstruction-fit-review/report.json, train_styles.png, val_styles.png. These results motivate an experiment, not a proven fix.
