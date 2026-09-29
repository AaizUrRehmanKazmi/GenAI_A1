# Provisional infrastructure decisions

These decisions make the skeleton runnable and are separate from your model-methodology research. Revisit and explain them in your own final report.

| Choice | Practical reason | Alternative and trade-off |
|---|---|---|
| Separate frontend/backend images | Independently build Python service and static React assets | One image reduces services but couples processes/builds |
| Multi-stage frontend build | Final image serves compiled files without a Node build environment | Vite development server supports hot reload but is unnecessary for static deployment |
| Same-origin Nginx API proxy | Browser calls a single origin; Docker resolves backend service name | Direct backend calls need environment-specific URLs and CORS policy |
| CPU inference shell | Starts without choosing training GPU hardware | GPU image/runtime should be investigated once hardware and model needs are known |
| Explicit 503 for incomplete inference | Makes missing work visible without fabricated outputs | Mock predictions could obscure whether real inference works |
| Model bind mount | Allows model replacement without rebuilding images | Bundled artifacts simplify shipping but increase image size |

References to read and verify:
- https://docs.docker.com/compose/how-tos/startup-order/
- https://docs.docker.com/compose/how-tos/networking/
- https://fastapi.tiangolo.com/deployment/docker/
- https://vite.dev/guide/build
