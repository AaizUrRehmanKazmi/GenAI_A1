# Task 4 resumable screening

Use notebooks/task4_optuna_kaggle.ipynb after committing and pushing the implementation. Attach FS2K, select GPU, and run cells in order. Install requirements-optimization.txt. No baseline checkpoint is required: every new trial starts from scratch with seed 42. Baseline training source is unchanged.

Eight trials, five complete epochs each, are an initial screen. Trial 0 repeats the baseline. Search generator LR 5e-5–4e-4, discriminator LR 2e-5–2e-4 (log scales), batch 4/8, shared generator/discriminator width 32/48, generator dropout 0/.2/.4, style embedding dimension 8/16, L1 weight 50/100/150. TPE uses four startup observations. This is a small exploratory budget, not exhaustive optimization or a demonstrated remedy for grid artifacts.

Rank the final screening epoch by 0.8*equal-style validation L1 + 0.2*(1-equal-style validation SSIM). Do not compare weighted GAN training loss across trials. Full train/validation splits are used by default; no official test images are read. Model checkpoint best.pt within a trial is selected by L1; screening ranks final-epoch metrics, so these selections can differ.

study.json atomically stores sampled parameters before training and records completed scores. A pending trial resumes last.pt, including both optimizer and RNG states; completed trials are not retrained. Source, config, split, image-byte and Optuna-version fingerprints must match. --trials is the total count and can be increased. --max-hours is per invocation and allows cooperative stopping, not a runtime guarantee. An abrupt interruption can lose work since the latest trainer checkpoint. Back up the ENTIRE folder, including trial YAMLs, checkpoints and study.json. After a session reset attach the backup and set resume_search_folder to its extracted root. Keep the same visible GPU count.

After screening, inspect the winner's previews and compare against the baseline. Retrain selected_config.yaml from scratch for the full 30-epoch schedule in a new folder, without --resume or --stop-after-epoch. Select/evaluate the resulting checkpoint on validation before any official test use. Early five-epoch ranking may miss slower-learning configurations. All style-transfer visual judgments require same-photo style probes.

Validation: config/objective unit tests; real CPU two-image training / one validation pair per style smoke search. Debug results are not assignment evidence.
