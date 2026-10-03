# Task 4 provisional GAN training

Trainer training/train_task4_gan.py uses paired random horizontal flips, RGB
128x128 tensors, learned styles in both models, Adam betas(.5,.999), separate
D real/fake and G adversarial/L1 logging. Initial LR .0002 for both, base32,
embedding8, dropout.2, L1 weight100, batch8, total30 epochs. These are provisional
settings pending inspection and assignment-required Optuna tuning.

Validation uses original unaugmented pairs and equal style weighting. Best.pt
minimizes style-balanced validation L1; SSIM and per-style values are reported.
Preview uses the first fixed validation pair per style. MLflow archives each
preview by epoch. GAN loss alone is not a model-quality ranking metric.

Both networks and Adam states, CPU/CUDA RNG, source shuffle/cursor and accumulated
losses are saved at epoch boundaries, each50steps and graceful stop. Training
updates are resumed at complete D+G batch boundaries. Hard kills may lose steps
since last checkpoint. Resuming on another device type is rejected. Source,
config, split/metadata and train/val image byte hashes must match. Official test
pixels are not read. Original data and other task weights are not modified.

Kaggle: push source, import notebooks/task4_gan_kaggle.ipynb, attach the FS2K
archive as private dataset, enable GPU/Internet. Run setup, dataset preparation,
smoke, then five-epoch baseline. Download task4-baseline_backup.zip and inspect
history/preview before continuing. After session reset attach FULL run backup,
set restore_folder to extracted root containing last.pt; do not point it at FS2K.
Keep one visible GPU. Two-hour budget is soft, not a runtime guarantee.

CPU smoke/resume diagnostics use only two training pairs and one validation pair
per style; their scores are not quality evidence. Full Kaggle execution pending.
