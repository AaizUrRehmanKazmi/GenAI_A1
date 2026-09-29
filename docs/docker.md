# Docker operation
Install Docker Engine with Compose v2 or Docker Desktop on the machine where you will run this repository. Verify `docker version` and `docker compose version`. No host Python or Node install is required for container startup.

From repository root:
```sh
cp .env.example .env
docker compose config --quiet
docker compose up --build -d --wait
```
Visit http://localhost:8080 and http://localhost:8000/docs. `APP_PORT` and `API_PORT` can be changed in `.env` if occupied. Both ports bind to loopback.

```sh
docker compose ps
docker compose logs --tail=100
docker compose restart
docker compose down
```
Run `sh scripts/docker_smoke.sh` to build and check both the backend and the frontend's API proxy. It leaves the stack running. Startup depends on the backend health check. Health means the HTTP service works; it does not claim that inference is ready.

Backend runs as an unprivileged user. Frontend is built with Node and served by unprivileged Nginx. Nginx proxies `/api/` to the backend service on Docker's network, avoiding browser CORS configuration. Models mount read-only from `exports/models`. Raw datasets and checkpoints stay out of build contexts. Training is deliberately outside these inference containers: decide GPU/CUDA requirements before adding a training image.

If build fails, inspect network access to Docker Hub, PyPI and npm. If the Docker daemon is unavailable, start Docker on the host. If the backend is unhealthy, inspect its logs before increasing timeouts. A 503 from an inference route is currently expected: no trained pipeline is implemented.

After model development: add ONNX Runtime to backend dependencies; implement session lifecycle and validated preprocessing; document/download model artifacts; verify numerical parity; update readiness; rerun clean-clone container smoke tests. Never bake secrets or raw datasets into images.

Image tags and direct Python versions are specified, but images are not digest-pinned and Python transitive dependencies are not locked. Freeze those after testing the target platform before calling the final deployment reproducible.

References: https://docs.docker.com/compose/how-tos/startup-order/ ; https://fastapi.tiangolo.com/deployment/docker/ ; https://vite.dev/guide/build
