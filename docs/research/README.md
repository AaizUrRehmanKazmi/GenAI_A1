# Research workflow
Copy decision-template.md for each substantial decision. Fill sources and alternatives before coding; add experimental evidence before final selection. This scaffold makes no model-methodology selections.

Decision queue:
1. Framework and environment; tracking provider (MLflow or W&B).
2. Data split verification, paired transforms, exact occlusion union area, deterministic manifest schema.
3. Task 1 bottleneck, channel/decode design, limited skips, L1/SSIM implementation and Optuna ranges.
4. Task 2 classifier, balanced sampling, shared specialist search, oracle vs predicted analysis.
5. Task 3 warm-up, fine-tuning schedule, temperature, balance regularizer and routing-collapse criteria.
6. Task 4 style embedding injection in both networks, paired augmentations, GAN stability and trial budget.
7. Metrics, aggregation by severity, uncertainty and failure-case selection.
8. ONNX export options, tensor contract, normalization and parity tolerances.
9. Stitch design evidence and final UI integration.

Do not tune on official test data. Suggested starting values in the plan are hypotheses, not demonstrated choices.
