# Task 4 dropout ablation

New config task4_reconstruction_nodropout.yaml changes only generator.dropout from0.2 to0.0. Existing trainer/source fingerprints are unchanged. Start fresh, never resume previous dropout0.2 checkpoints. Same initialization seed, widths, style embeddings, bilinear data processing, split manifests, random-flip policy, loss, optimizer and epoch budget. Dropout consumes RNG, so the precise subsequent random augmentations/order need not match despite the shared seed; this is not a common-random-number experiment. A single-run comparison is not statistical significance evidence.

Notebook task4_nodropout_kaggle.ipynb uses the explicit new config for BOTH smoke and full run. Attach FS2K only, GPU enabled, first restore_folder=None. Compare five epochs first against the existing five-epoch reconstruction history (L1 0.09527756, SSIM0.52597130). Preserve complete task4-reconstruction-nodropout_backup.zip. Share only history.json and preview.png initially. Rerunning resumes this new run only.

If validation improves, examine full train/validation metrics at a matched later budget before attributing improved generalization. If training improves without validation gains, investigate overfitting. Do not infer benefit from training loss alone. This ablation is not the final required conditional GAN.
