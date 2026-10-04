# Validation of this scaffold

Executed on 2026-09-28:
- Five Python unittest checks passed: all four inference routes return 503 for valid images; health explicitly reports inference unavailable; malformed images are rejected; styles outside 1–3 are rejected; the validation helper rejects oversized payloads.
- All Python files parsed successfully; task YAML and Compose YAML parsed successfully.
- Docker smoke script passed shell syntax checking.
- Frontend dependency installation succeeded with pnpm 11.19.0 and a frozen lockfile; Vite production build succeeded (30 modules).
- CLI help works for the Task 1 training placeholder. The execution path intentionally exits unsuccessfully until implemented.
- Git ignore checks confirmed raw images, ONNX models and .env are excluded.

Not verified:
- Docker/Compose schema validation, image build, container health, Nginx proxy and clean-clone startup: Docker is not installed in the execution environment. YAML parsing alone does not establish Compose validity. Run `sh scripts/docker_smoke.sh` on a Docker host.
- Browser visual/interaction inspection and final Google Stitch design.
- Model training, inference, evaluation, ONNX parity, datasets and research findings: these are intentionally unimplemented.
- IEEE report compilation: outline only; requires an installed IEEEtran LaTeX distribution.

Tests used the available local Python environment rather than the backend image. The HTTP tests use an in-process ASGI client; the oversized-file check targets the validator directly, not the multipart server path. These do not replace container integration tests.

Git is initialized on main, with no remote and no initial commit. Author identity is not configured. Set your own repository-local name/email before committing; do not use a fabricated identity.

## Dataset split milestone — 2026-09-29

Implemented `scripts/prepare_pets.py` and ran it on the downloaded dataset. Generated 2,944 training, 736 validation and 3,669 official test paths. Excluded 41 unlisted JPEG files. A rerun accepted identical outputs. Six isolated split tests passed, covering reproducibility, partitions, official test order and invalid-input/conflicting-output handling. Image decoding and preprocessing are not yet verified.

## Clean-image loader milestone — 2026-09-29

Implemented PetsDataset and the development-image scanner. Five loader tests passed: RGB channel/range correctness, deterministic retrieval and source preservation; grayscale/RGBA conversion and batching; EXIF rotation; explicit corrupt-image failure; malformed lists, duplicate/escaping/missing paths and invalid interpolation rejection.

Scanned every training and validation image: 2,944/2,944 training and 736/736 validation passed decoding, shape, dtype, finite-value and range checks. Reviewed the 16-image grid: colours and image content were readable. Some source poses/orientations are naturally unusual; the loader follows EXIF metadata without guessing semantic orientation. No official test image was decoded. Outputs are artifacts/data_checks/report.json and train_grid.png (ignored by Git).

Verification environment: Python 3.12, torch 2.14.0+cpu, NumPy 2.5.3, Pillow 11.2.1, installed in the workspace's work/pets-venv. GPU behavior and multi-worker data loading have not been benchmarked. This is a CPU development check, not the final lab training environment.

## Corruption milestone — 2026-09-29

Six corruption tests passed in the CPU development environment. Verified deterministic JSON-spec replay and clean-input preservation for all four classes; approximate salt/pepper frequencies and RGB pixel consistency; Gaussian constant-image preservation, impulse symmetry and no channel mixing; 300 occlusion cases across required severity levels; training-parameter ranges and class counts over 2,000 seeds; and invalid input/spec rejection.

Generated and visually inspected artifacts/data_checks/corruptions_grid.png: four training images each show clean plus low/medium/high salt, blur and occlusion (40 panels). Noise density and blur severity increase visibly, and mask coverage is labelled using actual union area. No test images were loaded. Dynamic dataset integration, balanced classifier batching and fixed validation/test manifests remain unimplemented.

## Restoration dataset integration — 2026-09-29

Four new integration tests passed: fresh training corruption with seeded replay; fixed evaluation and mixed-condition batching; manifest generation without images and identical reruns; rejection of modified records, conflicting files and wrong splits.

Generated 7,360 validation cases and 36,690 test specifications, each covering clean plus nine corruption/severity cases per image. No official test pixels were read. Real-data smoke check verified a 16-image input/target batch with shape [16,3,128,128], fresh per-access training specs, unchanged targets and identical replay for the first ten validation cases. It did not decode every validation corruption case. Multi-worker throughput and exact multi-worker checkpoint continuation remain untested.

## Task 1 baseline — 2026-09-29

Four tests passed for the model bottleneck/output shape and gradients, SSIM/L1 identity and weighting endpoints, tiny learning, and optimizer/RNG/checkpoint continuation. The initial 20-update tiny learning test did not reach its improvement threshold. Investigated convergence and extended the synthetic constant-image diagnostic to 120 updates at its diagnostic learning rate 0.01 (not the baseline training rate); loss decreased from about 0.321 to 0.051. This does not establish learning of pet structure or restoration quality.

Ran one real-data CPU smoke epoch with the default 4,397,507-parameter model on 16 training images and all ten fixed cases of one validation image. Training loss was about 0.31159; condition-balanced validation loss about 0.37408 and SSIM about 0.2499. These DEBUG SUBSET values are not assignment results. Confirmed last.pt, best.pt, JSON metrics, config/provenance, preview and MLflow records. Visually inspected the preview: columns were correct; generated output remained almost uniform after a single training batch, as expected for this smoke check. Resuming a completed smoke run performed no additional epochs.

Also exercised actual process termination and resume using a smaller two-epoch diagnostic model on 32 real training images: sent SIGTERM after a progress update, saved at epoch 0/cursor 8, resumed and compared against uninterrupted training. Every final model tensor and the complete epoch metric history matched exactly on CPU. The process test used a temporary workspace folder; it is not a trained-model deliverable.

Colab notebook structure/JSON checked, but GPU, Google Drive integration and Colab execution were not run here. No full-split training, Optuna search, official test evaluation or ONNX export was performed. Dependency environment includes torchmetrics 1.7.1, PyYAML 6.0.2 and mlflow-skinny 2.22.0 alongside the existing CPU PyTorch.

## Task 1 validation comparison tool — 2026-09-29

Implemented validation-only evaluator with checkpoint/source/split provenance checks, input-versus-restored L1/SSIM/combined loss, equal-condition aggregation, CSV/JSON outputs and example selection. Three tests passed for aggregation weighting, representative/regression selection and absence of invented failures. End-to-end CPU smoke run analyzed 40 fixed cases from four validation images using the local one-batch smoke checkpoint. Twelve representative panels and four unique-image SSIM regressions were generated and visually inspected. The near-uniform predictions belong to the smoke model, not the user's trained Colab checkpoint. The trained epoch-13 checkpoint and full validation run remain to be analyzed in Colab. No official test image was evaluated.

## Clean-image diagnostic — 2026-09-30

Implemented a separate fixed-clean-image diagnostic without modifying baseline model, loss, training source or configuration. A two-update real-data CPU smoke run on two selected training images saved checkpoints, MLflow records and step-0/1/2 grids; inspected the final grid for correct target/output/error layout. This short check is not the proposed 16-image capacity experiment.

An integration test passed for fixed-image loading, grayscale conversion, forced zero dropout, source-image preservation, rejecting overwrite and incompatible resume settings, and exact optimizer/model/history continuation. It compared four uninterrupted updates with two updates plus two resumed updates on a tiny synthetic fixture. GPU execution and the full 1,000-update diagnostic remain for Colab. Baseline files and checkpoints are unchanged.

## Tiny-set corruption diagnostic — 2026-09-30

Added a separate corrupted-input diagnostic while keeping baseline and clean-diagnostic source unchanged. Two tests passed: fixed-case reproducibility without advancing the training RNG, fresh training inputs with preserved targets, and end-to-end resumed/uninterrupted equivalence, dropout configuration and overwrite protection. A two-update CPU smoke run on two real training images produced fixed-case metrics, grids and checkpoints. Visually inspected the ten-case grid. No full diagnostic learning or GPU run was performed locally; these scores are not restoration-quality evidence.

## Spatial Task 1 experiment (2026-09-30)
- Three spatial-model tests passed on local CPU: compressed latent/output shape, no forward bypass, finite gradients and tiny-target loss improvement; stochastic next-update checkpoint equivalence; invalid shape/compression rejection.
- Existing three validation-report tests passed. New notebook code cells parsed successfully; git diff whitespace check passed.
- Real-data CPU smoke completed one epoch with 16 training images and 10 validation cases, producing last/best checkpoints and MLflow artifacts under artifacts/spatial-agent-smoke. Spatial evaluator completed 40 validation cases and generated reports/grids. These subset results are execution checks, not performance evidence.
- Default model: 203,107 parameters, latent 32×8×8. Full GPU training and full validation are pending on Colab. Original model/trainer/data/loss sources were not modified.

## 16×16 spatial pilot (2026-09-30)
- Three CPU unit tests passed: compressed shape/range, forward through latent, finite gradients and tiny-target learning; stochastic checkpoint update equivalence; invalid input/compression rejection.
- Real-data smoke paused after epoch1 with target2, resumed to epoch2, and spatial16 evaluation completed 40 cases with reports/grids. These are execution checks, not evidence of restoration quality.
- Notebook code cells parsed and git diff whitespace check passed. GPU pilot remains pending. Earlier model/trainer source files remain unchanged.

## Optuna screening (2026-09-30)
- Installed Optuna4.5.0 in the workspace CPU test environment.
- Two unit tests passed: ranking is independent of training-alpha loss, and completed observations reconstruct a usable Optuna study.
- Real-data debug search completed one trial, then a second invocation with total target2 retained trial0 and completed trial1. Identity baseline, checkpoints, histories, MLflow logs, leaderboard and selected YAML were produced. GPU/full-data search remains pending.
- Notebook code cells parsed; git diff whitespace checks passed. Existing trainer/model sources were not modified.

## Task2 classifier baseline (2026-10-01)
- Two CPU tests passed: exact batch class counts, seed replay, clean-target preservation, finite classifier gradients; hand-checked confusion-derived metrics and zero-support handling.
- One-epoch real-data debug smoke (8 source images/32 training examples,20 validation cases) completed and saved checkpoint, report and MLflow outputs under artifacts/classifier-agent-smoke. This tiny run checks execution only, not classification performance.
- Notebook cells parsed; git diff whitespace check passed. Full GPU training and classifier-specific interrupted-run equivalence remain unverified. Shared checkpoint helpers were tested previously.

## Task2 specialists (2026-10-01)
- Three CPU tests passed: independent parameter updates, condition-only manifest selection, predicted/oracle routing with exact clean identity and skipped unused experts.
- Separate real-data CPU debug smokes completed for salt, blur and occlusion (4 train images and3 fixed validation cases each). Salt paused after epoch1 and resumed through epoch2. Generated checkpoint/history/MLflow artifacts. These are execution checks, not restoration evidence; GPU pilots remain pending.
- Notebook code cells parsed and git diff whitespace check passed. Task1 and classifier source files were left unchanged. Full routing evaluation, specialist tuning and ONNX remain pending.

## Task2 evaluation pipeline and Optuna screening (2026-10-02)
- Seven unit tests passed in tests/test_evaluate_task2.py and tests/test_optimize_task2.py: condition-balanced metric aggregation, routing error taxonomy (clean-as-corrupted, corrupted-as-clean, cross-corruption), representative/failure selection, clean identity routing cost, classifier and specialist Optuna study reconstruction and ranking invariance.
- Real-data CPU smoke evaluation completed 40 validation cases using existing smoke checkpoints: verified end-to-end HardRouter execution, confusion matrix calculation, Oracle vs Predicted routing comparison, CSV/JSON metrics exports, and 5-column representative/routing-failure visual grids.
- Added resumable Optuna screening scripts for classifier (optimization/optimize_task2_classifier.py) and specialists (optimization/optimize_task2_specialists.py) with atomic state persistence.
- Added Task 2 evaluation Colab notebook (notebooks/task2_evaluation_colab.ipynb). Total 56 unit tests passing across entire repository. Full GPU runs remain pending on Colab.

## Kaggle migration notebook (2026-10-02)
- Added occlusion resume notebook with explicit checkpoint selection, condition/debug checks, source verification, one visible GPU for single-GPU RNG compatibility, read-only input copying, seed-offset reversal and backup archive.
- All code cells parsed as Python; no fingerprinted training/model code modified. Actual Kaggle provisioning, dependencies, GPU run and output persistence are not yet verified.

## Kaggle classifier notebook (2026-10-02)
- Added fresh/resumed classifier workflow with GPU check, dataset setup, metrics display and ZIP backup. Code cells parsed successfully; actual Kaggle execution remains pending. Existing classifier/trainer sources are unchanged.

## Full routed evaluator implementation (2026-10-02)
- Extended pre-existing uncommitted evaluator rather than replacing its reporting logic. Added safe loading, enforced inference/data fingerprints, optional Task1 comparison, all classifier probabilities, fixed alpha0.8 default, checkpoint epoch/debug metadata and actual-misroute-only examples.
- Two tests passed: equal-condition aggregation with optional Task1 metrics/no fabricated failures; validation provenance mismatch rejection.
- End-to-end CPU smoke evaluated40 cases with all five local debug checkpoints, producing JSON/CSV/report and image grids. Debug status was displayed. Full trained-model Kaggle run is pending; smoke scores are not performance evidence.
- Kaggle notebook cells parsed; git diff whitespace checks passed. No model/trainer/data sources changed.

## Task2 search completion (2026-10-02)
- Four objective/study reconstruction tests passed. Classifier and blur one-trial CPU debug searches completed and produced study, leaderboard, selected config and checkpoint files.
- Classifier search checkpoint resumed from epoch1 through epoch2 using the new search trainer while retaining target10. Original classifier/specialist trainer and model sources are unchanged.
- Notebook cells parsed and diff whitespace checks passed. Full GPU searches are pending; smoke metrics are not final evidence. Searches use separate folders and provisional winners.

- 2026-10-03: Four selected Task 2 ONNX graphs passed checker and sampled CPU PyTorch/ORT parity; routing identity/mixed dispatch passed. See task2_delivery.md and generated manifest. Three routing-contract tests passed.

- 2026-10-03: Four Task 2 API tests passed, including real ONNX HTTP inference, routing identity, missing models/invalid upload, and hash rejection. Docker not accessed; container verification pending.

- Task 3 stage 1: three SoftMoE model tests and actual selected-checkpoint two-step diagnostic passed. Expert freezing/unfreezing and gate gradients verified. One training image only; no Task 3 validation claims.

- Task 3 trainer: tiny CPU paused/resumed run exactly matched uninterrupted model tensors, progress/history and RNG through two warm-up epochs and one joint epoch. Notebook cells compile. CUDA and full-split training pending.

- Task 3 detailed evaluator: 3 aggregation/selection tests passed; real epoch-10 checkpoint smoke evaluation passed on 40 validation cases; routing heatmap inspected and notebook cells compile. Full CPU run was stopped after progress at 408/7360 because of runtime cost; no full-set evaluation report claimed. Run full evaluation on Kaggle GPU.

- Task 3 Optuna: three objective/config/dimension tests passed and a one-trial CPU diagnostic completed three epochs with warm-up/joint transition. Notebook code cells compile. Full eight-trial GPU search pending.

- Task 3 selected complete ONNX graph: CPU parity passed (image max error 1.85e-6, weights 1.20e-7), batches 1/3/4; three API tests passed with inline thread-offload test adapter. Docker verification pending. See task3_delivery.md.

- Task 4 FS2K preparation: two synthetic tests passed for deterministic style-stratified splitting and official pair mapping/missing-file rejection. Real dataset validation pending download.

- FS2K actual data: 898/898 train and 160/160 validation pairs decoded successfully; shape/range/finiteness and pair dimensions checked. Nine training pairs visually inspected. Two paired-loader tests passed. Official test pixels not decoded.

- Task 4 models: CPU synthetic test covers U-Net RGB128 output, PatchGAN14x14 logits, style-dependent output and nonzero embedding gradients, detached discriminator update and generator backward. No GAN quality claim.

- Task 4 real-pair diagnostic: 20 CPU steps on three training pairs passed finite-gradient and D/G isolation checks; both style embeddings updated. L1 .33268 to .16281 on those pairs only. Preview inspected; no held-out quality claim.

- Task 4 resume test: paused after one complete D+G batch, resumed through two epochs; exact equality with uninterrupted run for both models, both optimizers, progress/history and CPU RNG. Tiny debug subsets only. Notebook cells compile.

Task 4 search: config/objective unit tests pass; CPU debug screening creates study, leaderboard, selected config and checkpoints. Full GPU search remains to be run. See docs/task4_optuna.md.

Task 4 selected epoch 19: CPU full train/validation re-evaluation reproduced saved validation metrics (L1 0.10518875, SSIM 0.50645363). See task4_selected_validation.md; visual limitations remain and official test was untouched.

Task 4 deep candidate: output-range/style-gradient/frozen-BatchNorm test passed. Real CPU debug pause/resume over two epochs exactly matched uninterrupted generator/discriminator states (including buffers), both optimizer states, history/progress and CPU RNG. Debug subsets are not quality evidence. Candidate notebook cells compile; full GPU validation comparison pending.

Corrected Task 4 BN trainer: focused train-logit/input-gradient/frozen-state test passed; real CPU debug two-epoch paused/resumed run exactly matched uninterrupted model/buffer states, both optimizers, progress/history and RNG. Notebook code cells compile. No GPU quality claim yet.

Task 4 reconstruction-only: CPU debug two-epoch pause/resume exactly matched uninterrupted models, optimizer states, progress/history and RNG. Generator changed, discriminator state matched fresh initialization and its optimizer remained unused. Notebook cells compile; full GPU validation pending.

Task 4 fine-tuning: both control and GAN CPU debug runs initialized from selected reconstruction checkpoint successfully. Two-epoch GAN resumed versus uninterrupted states matched exactly (models, optimizers, progress/history, RNG). Notebook compiled. GPU validation remains pending.

Task4 sketch-gradient ablation: three loss tests passed; real-checkpoint CPU debug two-epoch resumed/uninterrupted generator, optimizer, progress/history and RNG matched exactly. Discriminator optimizer unused. Notebook compiled; full GPU comparison pending.

Fixed12 candidate diagnostic: CPU real-pair two-step resumed and uninterrupted model, optimizer, RNG and history matched exactly; notebook cells compile. Longer GPU fit remains pending.

Reconstruction epoch14 full eval: train898 L1 .08013179 SSIM .56682378; val160 L1 .08853802 SSIM .55807106. Blurred training examples confirm poor fitting is not restricted to held-out images; no causal claim about dropout. See task4_reconstruction_fit_review.md.
