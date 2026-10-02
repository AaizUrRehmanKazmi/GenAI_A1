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
