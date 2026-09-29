# Task 1 baseline and training guide

Status: provisional implementation for learning and benchmarking, not an Optuna-selected final model. No performance comparison justifies the current settings yet.

## Architecture and loss

Input [3,128,128] → four stride-2 convolution blocks with 16/32/64/128 channels → [128,8,8] → 256-dimensional vector → [128,8,8] → four upsampling/convolution blocks → sigmoid RGB output [3,128,128]. There are no skip connections; every output depends on the vector bottleneck.

Nearest-neighbor upsampling followed by a convolution is the initial decoder choice. Transposed convolution and bilinear upsampling are alternatives to research, not experimentally rejected methods. Channel sizes and a vector latent are compact starting choices, not established optimal settings. Compare bottleneck capacity and reconstruction detail under equivalent training budgets.

Loss is alpha*L1 + (1-alpha)*(1-SSIM), initially alpha=0.8. SSIM uses the TorchMetrics functional implementation with data_range=1, Gaussian window, sigma=1.5 and kernel size 11. Both terms are calculated per image, then averaged over a training batch. All tensors and training operations are float32 initially; mixed precision is a future performance experiment. Adam at 0.001 and 20 total epochs are provisional budget settings. There is no learning-rate scheduler in this baseline.

Validation computes L1, SSIM and combined loss per condition/severity. Average the three severities within each corruption, then average clean/salt/blur/occlusion with equal weight. This prevents the three corruption severities from reducing clean's contribution to 10%. Lowest condition-balanced validation loss selects best.pt. The official test set is never loaded by this trainer.

## Local installation and smoke run

Activate the existing environment from the repository root:
```sh
source /home/aaiz/Documents/Codex/2026-09-28/for-x20/work/pets-venv/bin/activate
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -p test_task1.py -v
python -m training.train_task1 --device cpu --epochs 1 --train-limit 16 --val-images 1 --output-dir artifacts/task1-smoke
```

Debug limits use the first N saved training images and first N validation images with all ten cases each. Such runs are labelled debug subsets; their results are not assignment evaluation evidence. Use a new output directory for each independent experiment. The script refuses to overwrite an existing last.pt without an explicit --resume.

## GPU baseline

Install a CUDA-enabled PyTorch version appropriate to the lab driver or keep Colab's supplied compatible build. Do not copy the laptop CPU virtual environment to the lab. Check `torch.cuda.is_available()` first.

```sh
python -m training.train_task1 --device cuda --max-hours 4.5 --output-dir artifacts/task1-baseline
```

This uses all 2,944 training images and 7,360 fixed validation cases per epoch. Validation may take substantial time; benchmark before committing to many epochs. CPU work uses two threads by default and data loading uses no background workers. Both are conservative debugging choices; GPU utilization and optimized loader throughput remain to be measured.

The session time limit is soft: checked between batches, including validation. Leave time for the last batch, checkpoints and tracking artifacts to finish. Ctrl+C requests a graceful stop. Hard termination or lost storage can lose work since the last periodic checkpoint (default every 50 batches).

## Resume

```sh
python -m training.train_task1 --device cuda --max-hours 4.5 --output-dir artifacts/task1-baseline --resume artifacts/task1-baseline/last.pt
```

Keep configuration and debug limits unchanged. `--epochs` is the total target, not extra epochs; if overridden originally, use the same value on resume. Time budget and output location can change. Copy the complete run folder when moving between machines, and keep the same source revision, split files and manifest. Source hashes reject accidental algorithm changes. Image contents themselves are not hashed by the trainer; retain the same dataset archive.

Checkpoints store model and Adam states, completed epoch count, current shuffled image order, next image cursor, partial loss totals, torch CPU/CUDA RNG state, selected configuration, data/code fingerprints and history. The loop uses PyTorch RNG exclusively for shuffle/dropout/fresh corruption seeds. There is no scheduler or gradient scaler state because neither is used. A partially completed validation pass is restarted on resume, without repeating completed training batches.

Exact same-environment CPU continuation is tested. Different devices/library versions can yield different floating-point results even with restored state. Record environment versions when moving to the RTX 3080. GPU execution has not been verified in this CPU environment.

## Outputs and tracking

- last.pt: most recent resumable training state.
- best.pt: best completed-validation state (not created until validation finishes).
- history.json: epoch metrics with detailed condition/severity results.
- preview.png: fixed validation target/input/output/absolute-error examples.
- config.yaml, data_signature.json, environment.json: provenance.
- mlruns/: local MLflow records, parameters, metrics and artifacts.

MLflow is the provisional tracking choice: local file storage needs no account and remains usable offline. W&B is an alternative to research. Each resumed session creates a new MLflow run linked by resumes_run_id, while history.json carries the complete checkpoint history. Preserve the whole output folder on persistent storage. The minimal mlflow-skinny package logs experiments; a full matching `mlflow==2.22.0` installation is needed for its web UI. For moved runs, historical MLflow artifact URIs may retain their original absolute locations; the standalone checkpoints/history remain portable.

## Remaining research and implementation

Compare architecture/bottleneck/decoder alternatives, loss weighting, learning rates, batch sizes and dropout. Implement the required Optuna search after a successful baseline. Benchmark speed and memory on the actual GPU. Final test evaluation, Task 1 failure-case analysis and ONNX export are later milestones.

## Primary references

- PyTorch checkpoint tutorial: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- TorchMetrics SSIM implementation: https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/ssim.py
- MLflow logging API: https://mlflow.org/docs/latest/ml/tracking/tracking-api
- Original SSIM paper: Wang et al., Image Quality Assessment: From Error Visibility to Structural Similarity (2004). Read the paper before discussing its theoretical advantages in the report.
