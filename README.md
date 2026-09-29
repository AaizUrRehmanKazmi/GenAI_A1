# GenAI Assignment — Research and Implementation Skeleton

This repository is the infrastructure starting point for the four-task assignment. It provides Docker configuration, a React/Tailwind shell, a FastAPI skeleton, module locations and research templates. It contains no trained models, fabricated outputs or completed methodological conclusions.

## Start the shell
Requires Docker with Compose v2:
```sh
docker compose up --build -d --wait
```
Open http://localhost:8080 (UI) or http://localhost:8000/docs (API). The four inference endpoints deliberately return HTTP 503 for valid images until implemented; health reports `inference_ready: false`. See [Docker guide](docs/docker.md) for smoke tests and troubleshooting.

## Repository map
| Path | Purpose |
|---|---|
| configs/ | Fixed assignment constraints and unresolved model/search settings |
| data/, scripts/ | Dataset preparation, split and corruption manifest placeholders |
| src/ | Data, models, losses, metrics and utility stubs |
| training/, optimization/, evaluation/ | Explicitly unimplemented CLI entry points |
| exports/ | ONNX export, parity and model artifact placeholders |
| backend/ | Runnable API and upload validation |
| frontend/ | Four workspace placeholders, React and Tailwind |
| tests/ | Infrastructure behavior checks |
| docs/research/ | Alternatives and decision record templates |
| report/ | IEEE LaTeX outline, figures and tables |

## Develop locally
```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt 'httpx>=0.28,<1'
python -m unittest discover -s tests -v
uvicorn backend.app.main:app --reload
```
In another terminal:
```sh
cd frontend
corepack enable
corepack prepare pnpm@11.19.0 --activate
pnpm install --frozen-lockfile
pnpm run dev
```
Vite proxies `/api` to localhost:8000. Install root `requirements.txt` only after selecting the training environment, especially the appropriate PyTorch/CUDA build. Its ranges are provisional, not a fully locked research environment.

## Begin your research
Start with [next steps](docs/next_steps.md), [requirement map](docs/requirements.md), and [decision template](docs/research/decision-template.md). Complete original Google Stitch design evidence before implementing the final interface. Suggested plan choices such as PyTorch or MLflow are provisional rather than assignment-mandated conclusions.

Read [validation status](docs/validation.md) before treating infrastructure as verified. No datasets or model downloads are triggered automatically. Large models and raw data are ignored by Git. The repository has no remote until you connect your GitHub repository.

## Clean pet image loading

`src/data/pets_dataset.py` now loads the saved Pets splits into RGB float32 tensors at 128×128, with values in [0,1]. It does not apply corruptions yet. See [preprocessing choices and usage](docs/research/pets_preprocessing.md).

With PyTorch installed in your selected Python environment:
```sh
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -p test_pets_dataset.py -v
python -m scripts.check_pets_dataset
```

The scanner checks all training and validation images and writes a report and grid under `artifacts/data_checks/`. It does not inspect official test images.

## Corruption checks

The corruption module implements the four required conditions and replayable settings. See [corruption conventions and examples](docs/research/corruptions.md).

```sh
python -m unittest discover -s tests -p test_corruptions.py -v
python -m scripts.check_corruptions
```

The preview grid and its settings use training images only. `PetsDataset` still returns clean images: a dynamic training wrapper and fixed evaluation manifests are the next milestone.

## Dynamic restoration data and fixed evaluation

`TrainingPetsDataset` generates fresh corruption on each access. `EvaluationPetsDataset` replays the saved validation/test cases. Both return input, target, label, image ID, relative path and JSON corruption settings.

```sh
python -m scripts.generate_manifests
python -m scripts.check_restoration_data
```

See [data pipeline usage and reproducibility](docs/research/restoration_data.md) and [manifest format](data/manifests/README.md). Test manifest generation reads filenames only. Do not run model selection on the test split.

## Task 1 baseline training

A provisional universal autoencoder, L1+SSIM loss and resumable training loop are implemented. See the [training guide](docs/research/task1_baseline.md) for architecture decisions, time budgets, checkpoint recovery, tracking and limitations. [Colab notebook](notebooks/task1_colab.ipynb) runs these same scripts with persistent output storage.

```sh
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -p test_task1.py -v
python -m training.train_task1 --device cpu --epochs 1 --train-limit 16 --val-images 1 --output-dir artifacts/task1-smoke
```

After the smoke check, use a CUDA-enabled environment for the full baseline:
```sh
python -m training.train_task1 --device cuda --max-hours 4.5 --output-dir artifacts/task1-baseline
```

Resume using the same settings and output folder:
```sh
python -m training.train_task1 --device cuda --max-hours 4.5 --output-dir artifacts/task1-baseline --resume artifacts/task1-baseline/last.pt
```

The CPU environment previously created in work/pets-venv is for local checks. It does not enable GPU training. The baseline settings have not been tuned; Optuna, final testing and ONNX export remain pending. Initial predictions after a one-batch smoke run are expected to look nearly uniform and are not meaningful restoration results.

## Analyze the Task 1 baseline

Use the saved best checkpoint for validation-only comparisons against the corrupted inputs:
```sh
python -m evaluation.evaluate_task1 --checkpoint artifacts/task1-baseline/best.pt --device cuda
```

See [Colab analysis instructions and output definitions](docs/task1_validation_analysis.md). Analysis outputs default to a new validation_analysis directory beside the checkpoint. The official test set remains untouched.
