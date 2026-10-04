# Task 4 sketch-gradient ablation

Both arms initialize the reconstruction epoch14 generator and reset Adam, using the same generator LR5e-5, five epochs, seed, architecture, dropout, dataset and preprocessing. Control100*L1; experimental100*L1+100*gradient_error. The latter is the average absolute difference between signed horizontal/vertical adjacent-pixel differences of prediction and target sketch, averaged over both axes/channels/batch. No image-border padding. Constant intensity offsets are invisible to this term, which is why pixel L1 remains. This targets local line differences, not photo edges. Weight100 is provisional, not tuned or guaranteed to improve quality; alignment differences can also affect it.

No discriminator updates. This is not a replacement for the final cGAN requirement. Compare unchanged validation L1/SSIM plus line quality/noise to control at matched epochs and the initialization. Do not compare total training losses between arms. Preserve all prior trainers/checkpoints; use new output directory and architecture-compatible reconstruction initialization only.

Use notebooks/task4_edges_kaggle.ipynb; attach FS2K and reconstruction_continued. Set exact epoch14 best.pt path. First restore_folder=None. Each arm gets one hour per invocation; rerun to finish five epochs. Download task4-edge-comparison_backup.zip. After a reset restore complete output folder and reattach identical initialization; signatures check checkpoint/source/config/data.

Validation: three gradient loss tests (identity/constant offset, missing-line gradients, invalid shape) pass; real-checkpoint CPU debug training completes; notebook compiles. No GPU quality claim. Logs include g_reconstruction and g_gradient separately.
