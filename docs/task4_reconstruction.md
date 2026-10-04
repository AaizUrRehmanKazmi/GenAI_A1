# Task 4 reconstruction-only ablation

Use notebooks/task4_reconstruction_kaggle.ipynb. This is an experiment to isolate the adversarial contribution, not a replacement for the assignment's cGAN. Start fresh with restore_folder=None and keep prior GAN runs intact. Run smoke then five epochs; download task4-reconstruction_backup.zip.

Uses CandidateGenerator (width32/style8/dropout0.2), identical seed42, full saved train/validation splits, bilinear resize, shared horizontal flips, Adam2e-4 betas(0.5,0.999), batch8 and coefficient100 on [0,1] L1. Validation retains equal-style L1/SSIM and best-L1 selection. No official test images read. No adversarial forward/backward or discriminator updates occur.

For matched initialization/RNG consumption and checkpoint format, the discriminator and its optimizer are constructed and stored but remain unused. This small storage overhead keeps the existing resume format and initialization sequence; discriminator optimizer state should remain empty. No policy switch or warm-start migration is supported. Checkpoints include objective=reconstruction_only and reject GAN configs.

The separate trainer preserves earlier source fingerprints. Resume retains generator/optimizer/RNG/order/cursor and strict source/config/data checks. Full target remains30 epochs but notebook initially stops at5 for comparison. Judge validation metrics alongside images; successful reconstruction does not prove faithful alternative-style synthesis or that GAN warmup will help.
