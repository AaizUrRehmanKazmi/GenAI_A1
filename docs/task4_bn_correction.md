# Task 4 discriminator BatchNorm correction

Use training.train_task4_candidate_bn and configs/task4_candidate_bn.yaml. Original baseline and candidate trainers remain unchanged, preserving checkpoint fingerprints. Start a fresh run: previous checkpoints used different optimization semantics and cannot be resumed as this experiment.

During generator optimization, generator_discriminator_phase sets discriminator train mode (batch statistics) and freezes parameter gradients. Backward runs inside the context. All running buffers are restored only after backward to avoid modifying tensors saved for autograd. Previous module modes and requires_grad flags are restored, including on exceptions. Discriminator optimizer updates remain ordinary train-mode updates.

This corrects the observed train/eval mismatch; it does not guarantee improved image quality. Keep all other configuration, splits, loss scale and metrics unchanged for a controlled comparison. Run five epochs and inspect validation and previews before further training.

Kaggle: notebooks/task4_candidate_bn_kaggle.ipynb. Attach FS2K, enable GPU/Internet, run setup and smoke before the full-split five-epoch run. First run restore_folder=None. Output task4-candidate-bn; download task4-candidate-bn_backup.zip. Later resume only a full backup of this corrected run with identical source/config/data and visible GPU count.

Focused test verifies identical logits to train-mode discriminator, nonzero input gradients, absent discriminator parameter gradients and exact parameter/buffer preservation. CPU two-epoch debug resume is checked against uninterrupted execution. These checks are not quality evidence; GPU validation remains pending.
