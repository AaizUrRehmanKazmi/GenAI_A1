# Task 3 detailed validation

Run `python -m evaluation.evaluate_task3 --checkpoint PATH/best.pt --bundle artifacts/task2-delivery --output-dir artifacts/task3-validation-analysis --device cpu`.
Use cuda on Kaggle. Output folder must be new/empty; --val-images is debug only.
The evaluator validates training-source/data fingerprints, selected initialization
hashes and rejects debug checkpoints. It compares the trained soft mixture with
the frozen selected Task 2 hard router using identical fixed validation inputs.
No official test data is loaded.

Outputs: summary.json, per_case.csv, routing_heatmap.png, representative_grid.png,
dominated_grid.png, distributed_grid.png and regressions_grid.png. Empty example
categories produce no grid. All absolute errors use fixed [0,1]. Examples include
clean targets, corrupted inputs, Task 2, Task 3 and Task 3 absolute errors.

Fixed evaluation loss is .8 L1 + .2(1-SSIM), averaging severities then conditions.
Raw case pooling would overweight corrupted conditions versus clean. Routing
weights are ordered identity/salt/blur/occlusion. Per-case maximum weight >=.9
is described as dominated; <.8 as distributed. These descriptive thresholds are
not trained decisions or guarantees of quality. Entropy is in natural-log units.
Mean routing weights alone cannot establish per-image mixture behavior; inspect
entropy/fractions and example grids. Gate argmax accuracy is not hard-route
restoration accuracy, because reconstruction uses all four weights.

Tests cover condition weighting, missing groups and empty failure/distributed
categories. A 40-case smoke test with actual checkpoint passed. Full evaluation
is run separately and its provenance records count, subset flag and checkpoint.
