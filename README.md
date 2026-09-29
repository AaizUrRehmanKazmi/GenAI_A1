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
