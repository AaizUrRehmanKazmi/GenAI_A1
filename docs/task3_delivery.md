# Task 3 delivery

Selected trial005 epoch10 checkpoint SHA256:
024a388659c2c55596b68a0bcf814c0aaa0b7c2aab016c168fd8a827f28f9a3d.
Temperature 1.5. Selection uses fixed-alpha .8 validation reconstruction loss
0.04444626, SSIM .87337465, L1 .02390149. Baseline has slightly better L1 and
clean preservation; retain it as comparison evidence. No official test used.

Export from repository root using the training environment plus requirements-export.txt:

```sh
python -m exports.task3_bundle --backup /home/aaiz/Downloads/task3_trial005_continued.zip --task2-bundle artifacts/task2-delivery --output-dir artifacts/task3-delivery
```

Choose a new output directory when reproducing. One complete ONNX graph embeds
gate, temperature, three experts, identity and weighted combination. Input image
is float32 NCHW RGB [N,3,128,128], EXIF transpose/PIL bilinear resize/[0,1].
Outputs restored [N,3,128,128] and weights [N,4] in identity/salt/blur/occlusion
order. Batch is dynamic; spatial dimensions fixed. Opset17. All branches run.

ONNX checker and CPU Runtime/PyTorch checks passed on 16 seeded validation cases
plus zeros/ones/random, batches 1/3/4. rtol1e-4, atol1e-5. Maximum absolute errors:
restored 1.847744e-6; weights 1.192093e-7. Routing normalization also checked.
ZIP includes selected best.pt, config, history, graph and verification manifest.
This is inference packaging; preserve original backups for resume. Files under
artifacts are ignored by Git; upload ZIP to Drive separately.

Backend /soft-mixture accepts multipart file PNG/JPEG and returns image_base64,
inference_ms and routing_weights. It does not pick one expert. Startup verifies
hash, input contract and warm-up outputs. /health tasks.soft-mixture is readiness;
legacy inference_ready continues to reflect Task 2. Missing model returns 503.
Compose mounts artifacts/task3-delivery at /models-task3 read-only; override host
path with TASK3_BUNDLE_DIR. No PyTorch dependency in serving container.

Three API tests passed (real graph/image/weights, invalid weights, unavailable
models). ASGI transport uses inline replacement for thread offload in the test to
avoid restricted sandbox event-loop sockets; deployment thread offload/container
execution is not covered. Docker verification is user-run:

```sh
sudo docker compose up -d --build backend
curl --fail http://127.0.0.1:8000/health
curl --fail -F 'file=@artifacts/task2-api-salt-input.png' http://127.0.0.1:8000/soft-mixture -o artifacts/task3-api-response.json
```

Frontend weight visualization remains pending; this change supplies the API.
