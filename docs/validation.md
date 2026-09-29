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
