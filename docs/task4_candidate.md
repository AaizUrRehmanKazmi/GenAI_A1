# Task 4 deep U-Net candidate

Adapted from user-provided task4_face2sketch.zip models.py, preserving attribution here. Adds six encoder levels, BatchNorm, learned bottleneck style projection and transposed-convolution decoding. Candidate width32/style8/dropout0.2 keeps a bounded GPU footprint; this is distinct from the submitted default width64/style16. Parameter/initialization changes are evaluated as a package, not attributed to one feature.

Separate trainer training/train_task4_candidate.py and config configs/task4_candidate.yaml preserve original baseline source fingerprints and checkpoints. Architecture identifier deep_unet_v1 prevents accidental baseline-config use. Both model adapters expose [0,1] inputs/output: internally scale to [-1,1] and map tanh output back. L1 weight100 therefore retains our baseline loss scale, not the submitted trainer's doubled effective weight. Existing bilinear preprocessing, official saved splits, flip augmentation, equal-style validation L1/SSIM and best-L1 checkpoint selection are unchanged. Test split untouched.

Discriminator runs in train mode for real/detached-fake updates. During the generator step discriminator parameters are frozen and eval mode freezes BatchNorm running statistics; gradients still propagate through its input into the generator. This is an explicit policy change from the submitted trainer, and should be included in reporting.

Resumption retains generator/discriminator parameters and buffers, both Adam states, shuffle/cursor state and CPU/CUDA RNG states. Config/source/data fingerprint checks remain strict. Use a new output folder; do not resume any original baseline checkpoint. Source/config changes after starting require a new run. Diagnostic script recognizes both architectures.

Kaggle: import notebooks/task4_candidate_kaggle.ipynb, attach FS2K, enable GPU/Internet, run setup, smoke check and five-epoch candidate run, then download task4-candidate_backup.zip. This first run is a validation comparison, not proof of improvement. Full default schedule is30; inspect five epochs before continuing. Same-photo style diagnostic remains available after training.
