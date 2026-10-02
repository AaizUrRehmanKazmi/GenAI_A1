# Task 2 routed validation

Run notebooks/task2_routing_kaggle.ipynb after pushing the evaluator. Attach private datasets containing extracted complete classifier, salt, blur, occlusion and Task1 trial005 output folders. Set exact best.pt paths. Expected baseline epochs: classifier10, specialists20, Task1 trial00520. Never select stale MLflow artifact copies merely because they share a filename.

The evaluator loads checkpoints with weights_only=True, verifies validation split/manifest and inference source hashes, checks specialist condition/architecture, and records SHA256/epochs and debug status. Task1 is optional in the CLI but required by the full-comparison notebook. No fingerprinted training code is changed.

All methods use fixed alpha0.8 L1+SSIM evaluation. Per-case CSV includes input, oracle and predicted scores, optional Task1 scores, all four classifier probabilities and routing costs. Summaries equally weight conditions, with per-severity breakdowns, classifier confusion and macro metrics, and clean-to-expert/corrupted-to-clean/cross-corruption counts. Clean oracle identity is exact by router construction. Positive routing cost means predicted routing loses quality versus oracle; negative cost can occur because a specialist is not guaranteed to improve an image. Counts of routing errors are not equivalent to counts of restoration failures.

Representative grids show target/input/oracle/predicted/error. Routing-error examples are deliberately selected misroutes, not representative prevalence estimates. Task1 comparison is numerical in the CSV and aggregate report, not a sixth grid column. Absolute error is displayed on fixed[0,1] scale. Debug checkpoints/subset reports are labeled and are not final evidence.

Download task2_routing_backup.zip; Kaggle does not automatically write to Drive. Share summary.json and image grids for review. This is validation analysis, not the required final test evaluation. Classifier/specialist Optuna and final frozen test comparisons remain pending.
