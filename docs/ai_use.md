# AI assistance record

Codex assisted with repository scaffolding, Docker configuration, a placeholder web shell, API transport validation, and infrastructure tests. Model architectures, training procedures, research comparisons and results remain unimplemented. See validation.md for executed checks and limitations. Student verification, corrections and understanding: TODO.

On 2026-09-29, Codex implemented the standard-library Pets split preparation script, generated split lists from the local official annotations, and added six tests for split integrity and failure handling. Dataset filenames were checked; image contents were not decoded or modified. The splitting algorithm and alternatives are recorded in research/pets_split.md for student review.

Codex implemented the clean PetsDataset, a development-only image scanner, five loader tests and a preprocessing decision note. All train/validation images passed decoding/tensor checks, and a 16-image grid was visually inspected. Bilinear resizing and [0,1] scaling are explicit provisional conventions; no comparison of downstream model quality was performed. Student review of the decisions and implementation remains necessary.

Codex implemented pure corruption functions, JSON replay specs, six tests and a training-only severity preview. Tests include target preservation, replay, pixel-noise statistics, Gaussian impulse/constant checks and 300 fixed-severity occlusion cases. Implementation choices about padding and mask geometry are provisional and documented in research/corruptions.md; no research comparison or model-quality claim was completed.

Codex added training/evaluation dataset wrappers, deterministic manifest generation, four integration tests and a real-data batching/replay check. Generated test specifications from filenames only. Validation's ten-case-per-image design and RNG/checkpoint constraints are documented for student review. No training or evaluation of a learned model was performed.

Codex implemented a provisional Task 1 autoencoder, TorchMetrics L1/SSIM loss wrapper, local MLflow logging, batch-resumable trainer and a Colab launcher notebook. Tested tensor/gradient contracts, synthetic learning, real-data smoke training and actual interruption/resume equivalence on CPU. The tiny learning check needed more updates than initially budgeted; both the initial failure and successful longer diagnostic are recorded in validation.md. Architecture and baseline settings are not research-selected; alternatives, limits and required student investigation are documented in research/task1_baseline.md. GPU/Colab execution remains unverified.

Codex implemented Task 1 validation analysis and added Colab analysis cells. Verified table aggregation and example selection with three tests and a 40-case local smoke evaluation. Input baselines and restored metrics are compared using identical SSIM settings. Failure candidates are selected by SSIM regression; their causes and report interpretation remain the student's responsibility. The trained Colab model has not been accessed locally.

Codex added training/diagnose_task1.py and documented the controlled clean-image experiment: same 16 training images as inputs/targets, no corruption or dropout, random initialization, separate outputs and resumable updates. Verified short CPU execution, output grids and exact resumed/uninterrupted results on a small fixture. The diagnostic has not established a cause for the trained baseline's smoothing; interpretation remains pending the Colab run.

Codex implemented the controlled tiny-set corruption diagnostic. It retains the clean diagnostic's architecture, initial seed, selected images, zero dropout and training hyperparameters while introducing runtime corruptions. Fixed evaluation uses the same training images and is explicitly not validation evidence. CPU tests and short visual smoke checks passed; the 1,000-update Colab experiment and its interpretation remain pending.

- Added an AI-assisted spatial convolutional Task 1 comparison, separate trainer/evaluator and Colab notebook. Architecture remains provisional; user must interpret results and justify methodology.

- Added AI-assisted 16×16 latent comparison with unchanged parameter count, separate checkpoint provenance, five-epoch pause and Colab workflow. Larger latent and changed receptive fields are documented; effectiveness remains to be tested on GPU.

- Added AI-assisted Optuna TPE screening across five hyperparameters with fixed-alpha final-epoch ranking, atomic JSON study persistence, unchanged trainer reuse, and a Colab notebook. Search bounds and final candidate choices require user interpretation; no final performance claim is made.

- Implemented AI-assisted Task2 CNN classifier baseline, exactly balanced paired-content batches, classification metrics, resumable training and Colab notebook. Architecture/settings are provisional; tuning, specialists and routing remain pending.

- Added AI-assisted independent specialist baseline training, condition-filtered validation, hard routing and Colab pilots. Settings reuse Task1 as a hypothesis; Task2 Optuna, full routing evaluation and export remain pending.

- Implemented AI-assisted Task 2 HardRouter evaluation pipeline (`evaluation/evaluate_task2.py`) comparing Oracle versus Predicted routing, classification metrics and confusion matrix, routing cost quantification, error taxonomy, visual grids, and a dedicated Colab evaluation notebook. Added resumable Optuna screening scripts for classifier and specialists (`optimization/optimize_task2_classifier.py`, `optimization/optimize_task2_specialists.py`) with atomic study persistence and unit tests. All findings remain provisional and require student verification on full GPU checkpoints.

- Added AI-assisted Kaggle migration notebook to resume an existing occlusion checkpoint without changing its training configuration. User must attach the private checkpoint dataset and preserve output backups.

- Added AI-assisted Kaggle classifier notebook with explicit manual backup and optional attached-checkpoint resume.

- Extended the existing Task2 evaluator with enforced checkpoint provenance, safe checkpoint loading, optional Task1 comparison, fixed-alpha ranking and actual-misroute example selection. Added Kaggle orchestration; full trained-model evaluation remains user-run.

- Reviewed existing Task2 search implementations; added separate resumable classifier search trainer, required specialist channel tuning, exact baseline LR and Kaggle search/backup notebook. Search bounds are provisional and require empirical/user interpretation.

- 2026-10-03: Codex implemented Task 2 selected-backup packaging, ONNX export and sampled numerical/routing verification, and documented results. No training weights changed; no official test images used.

- 2026-10-03: Codex implemented Task 2 ONNX backend, model readiness checks, Compose model mount and API tests. Model weights unchanged; Docker verification pending.

- Codex implemented Task 3 stage-1 model, loss, balanced restoration batch helper, tests and diagnostic using assignment-specified mixture/loss formulas. Training schedule remains provisional; full trainer and optimization pending.

- Codex implemented resumable Task 3 baseline training, fixed validation aggregation/routing summaries, MLflow logging and Kaggle notebook; tested CPU pause/resume equivalence. No full training or Optuna result claimed.

- Codex implemented Task 3 validation comparison and routing visualizations, tests and notebook evaluation cells. Fixed validation only; descriptive example thresholds documented.

- Codex implemented Task 3 Optuna screening and Kaggle notebook with proposed parameter ranges, baseline anchor, equal-budget ranking, atomic resume state and tests. Search outcomes are not yet available.

- Codex packaged selected Task 3 checkpoint, exported full mixture graph, verified numerical parity and implemented ONNX-only /soft-mixture service plus tests. Docker and frontend checks remain pending.

- Codex consulted the official FS2K README and split script, implemented annotation-based pair preparation and seed42 validation split, and tested with synthetic entries. No real FS2K training performed.

- Codex implemented FS2K paired tensor loader and shared horizontal flip, ran train/validation decoding checks, and inspected training-pair preview. GAN implementation pending.

- Codex implemented provisional style-conditioned U-Net/PatchGAN and BCE+L1 objectives, with learned embeddings in both networks and CPU model tests. Architecture choices documented; training and tuning pending.

- Codex added and executed real-pair GAN diagnostic with separate loss logging, gradient isolation assertions and preview. Full training pending.

- Codex implemented resumable dual-optimizer Task 4 trainer, validation/preview logging and Kaggle five-epoch baseline workflow. Initial hyperparameters are provisional; full GPU training and tuning pending.

- Added and ran CPU Task 4 best-checkpoint diagnostics on all train/validation pairs and same-photo style probes; recorded limitations in task4_diagnostic.md. No training source or saved configuration changed.

- Implemented resumable Task 4 Optuna screening and Kaggle notebook, including independent GAN learning rates and required architecture/loss parameters. Fixed validation objective, source/data fingerprint checks, and full fresh-retraining instructions added. CPU debug runs are not final evidence.

- Evaluated selected Task 4 epoch 19 locally on complete train/validation splits; generated same-photo three-style previews and documented remaining artifacts without attributing an unverified cause.

- Added a matched Task 4 L1-only versus GAN+L1 fixed-12-training-pair diagnostic, with identical generator initialization, disabled dropout/augmentation, and preserved baseline training source. Debug-only results must not be reported as validation performance.

- Reviewed user-supplied task4_face2sketch.zip; executed isolated alternative generator fitting and style-gradient checks. Documented training-only gains, metric incompatibility, resume limitations and dependency limits in task4_alternative_review.md.

- Integrated user-supplied deeper Task 4 architecture as a separate candidate, with [0,1] adapters, original split/metric conventions, discriminator BatchNorm freeze policy, separate resumable trainer and Kaggle notebook. Verified exact CPU debug resume; original baseline training sources unchanged.

- Tested candidate discriminator train/eval BatchNorm mismatch on six fixed train/validation batches using fresh model copies. Confirmed mode dependence introduced by the integration policy; documented correction proposal and causal limitations in task4_bn_diagnostic.md.

- Corrected the discriminator BatchNorm mode mismatch introduced during candidate integration using a separate trainer and buffer-preserving generator-phase context. Added focused state/gradient testing and a fresh-run Kaggle notebook; quality improvement remains unverified.

- Added separate full-split Task 4 reconstruction-only ablation retaining candidate initialization, data and metric conventions. No discriminator updates; preserved prior trainers. This diagnostic does not satisfy the final cGAN requirement alone.

- Added separate matched reconstruction-control versus low-weight adversarial fine-tuning, initialized from selected reconstruction weights with fresh optimizers, checkpoint/data provenance checks, and Kaggle backup workflow. Settings are provisional; debug runs are not quality evidence.

- Evaluated best fine-tuned GAN on all160 validation pairs with white-output baseline; inspected style-balanced representative and worst-SSIM panels/overlays. Documented missing linework and uncertainty about alignment; no official test usage.

- Implemented matched sketch-gradient versus reconstruction-only fine-tuning with target-sketch adjacent-pixel differences, unchanged validation metrics and separate trainer/checkpoints. Gradient weight is provisional; no quality claim from CPU debug runs.

- Added longer resumable fixed12 deeper-generator reconstruction diagnostic and Kaggle notebook, with periodic metrics/previews. Training-pair evidence explicitly separated from validation and final cGAN requirements.
