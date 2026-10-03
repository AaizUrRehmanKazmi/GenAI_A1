# Task 2 selected checkpoint packaging and ONNX export

The selected classifier (trial 005, epoch 10), salt (trial 007, epoch 19), blur
(trial 007, epoch 20), and occlusion (trial 007, epoch 20) are packaged by
`python -m exports.task2_bundle`. The selection hashes are fixed to the full
7,360-case all-tuned validation evaluation. The command rejects changed hashes,
debug weights, wrong epochs/conditions and modified model source.

## Reproduce

From the repository root, using the existing training Python environment:

```sh
python -m pip install -r requirements-export.txt
python -m exports.task2_bundle --backups-dir /home/aaiz/Downloads --output-dir artifacts/task2-delivery
python -m unittest discover -s tests -p test_task2_bundle.py -v
```

Use a new output directory if the bundle already exists. Required backup names:
`classifier_trial005_continued.zip`, `salt_trial007_continued.zip`,
`blur_trial007_continued.zip`, `occlusion_trial007_continued.zip`.
The dataset and prepared validation split must already be available locally.

## Inference contract

Four ONNX graphs use opset 17, float32 RGB NCHW, variable batch size and fixed
3x128x128 dimensions. Preprocess EXIF orientation, convert RGB, square resize
with PIL bilinear, divide by 255. Classifier emits logits ordered clean, salt,
blur, occlusion. Take argmax and dispatch each subgroup to the matching expert;
clean rows are copied exactly. `exports.task2_bundle.route` implements this
application-side routing. No combined hard-router graph is claimed: separate
graphs avoid tracing Python conditional dispatch and execute only chosen experts.

The existing generic export/verify skeletons for other tasks are unchanged.
Training source files and checkpoints were not modified.

## Verified locally

ONNX checker and CPU ONNX Runtime inference passed for every graph. Parity uses
rtol=1e-4 and atol=1e-5 on 16 seeded validation-image cases (four images, each
under four conditions), zeros, ones and random inputs. Tested batches: 1, 3, 4.
Classifier argmax matched; predicted routing and forced mixed/all-clean/repeated
expert routing matched PyTorch. Clean outputs were bitwise equal to inputs.
Three focused routing-contract unit tests passed. This is sampled numerical
verification, not a full new quality evaluation. GPU, web API and Docker ONNX
execution are not covered by this check. Official test images were not used.

Maximum absolute output differences:

| Model | Maximum error |
|---|---:|
| classifier | 0.000122070312 |
| salt | 1.57952309e-06 |
| blur | 1.84774399e-06 |
| occlusion | 1.66893005e-06 |
| Routed reconstruction | 1.31130219e-06 |

The ignored `artifacts/task2-delivery.zip` contains each selected best.pt,
config, history and ONNX graph plus manifest hashes, epoch metadata, versions,
parity measurements and preprocessing contract. Preserve original backups for
training continuation; this bundle intentionally does not include last.pt or
MLflow logs. Keep model artifacts outside Git; commit source and documentation.

Full all-tuned validation reference: classifier accuracy 0.98478261, macro F1
0.97449829; condition-balanced reconstruction L1 0.02784590, SSIM 0.84931487,
fixed-alpha-0.8 loss 0.05241375. Remaining limitations include clean false
triggers and degradation of lightly blurred inputs. These are validation results,
not official test evidence.

## Backend inference

POST /hard-routing accepts a PNG/JPEG multipart `file` and returns image_base64
(PNG), inference_ms (model execution only), probabilities and selected_expert.
Output is 128x128. Clean identity preserves preprocessed RGB pixels, not original
upload resolution or bytes. ONNX sessions load once, validate manifest hashes
and tensor contracts, and warm up before health reports inference_ready true.
Missing models yield 503 on inference; /health remains a liveness endpoint.
Other task endpoints remain placeholders. No PyTorch is required by the backend.

Compose mounts artifacts/task2-delivery read-only; TASK2_MODEL_DIR overrides it.
Run from the repository root:

```sh
docker compose up -d --build backend
curl --fail http://127.0.0.1:8000/health
curl --fail -F 'file=@data/raw/oxford_pets/images/Abyssinian_1.jpg' http://127.0.0.1:8000/hard-routing -o artifacts/task2-api-response.json
```

Require inference_ready true. If false, run docker compose logs --tail=80 backend.
Four local tests passed via `python -m unittest discover -s tests -p test_task2_api.py -v`:
real ONNX HTTP inference, all routes/clean identity, unavailable models/invalid
upload, and hash mismatch rejection. Real-model test skips if bundle is missing.
Tests use httpx ASGI transport and the installed local FastAPI environment.
Docker build/container execution and frontend upload integration remain unverified.
