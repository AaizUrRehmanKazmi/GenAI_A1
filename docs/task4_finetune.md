# Task 4 matched fine-tuning comparison

Two NEW five-epoch runs initialize only the selected reconstruction generator (epoch14). Both reset Adam state and retain identical seed/model construction, dropout, augmentation, saved splits and metrics. The discriminator starts fresh, never from the unused reconstruction optimizer. Generator LR5e-5; discriminator LR2e-5. Control objective100*L1; GAN objective100*L1+0.1*BCE. These are provisional settings, not established improvements. Candidate BatchNorm buffer-preserving policy is retained. Original source/checkpoints unchanged.

Use notebooks/task4_finetune_kaggle.ipynb. Attach FS2K and the complete reconstruction backup. Set initial to the exact non-MLflow reconstruction best.pt path. First restore_folder=None. Runs control then GAN, up to one hour each invocation; rerun to finish five total epochs per arm. Download task4-finetune-comparison_backup.zip before ending a session. Restore full comparison folder after reset and reattach the identical initialization checkpoint.

New initialization validates reconstruction objective, generator configuration and image/split fingerprints. Subsequent exact resume requires matching source/config/data and initialization SHA256, loads both optimizer/model states and RNG, and does not reinitialize generator from the source. --resume refers ONLY to a checkpoint made by this fine-tuning trainer. Stored discriminator/optimizer in control are unused, preserving initialization sequence/checkpoint format.

Compare the two arms at matched epochs and against untouched reconstruction epoch14. Best.pt is minimum validation L1 among fine-tuning epochs; it is not guaranteed better than initialization. Inspect same-photo styles and noise as well as L1/SSIM. All validation is on existing split; official test untouched. Reconstruction control is an ablation, not the final cGAN deliverable.

Validation: real selected-checkpoint CPU smoke runs completed for both arms on debug subsets; notebook cells compile. Full GPU quality assessment pending.
