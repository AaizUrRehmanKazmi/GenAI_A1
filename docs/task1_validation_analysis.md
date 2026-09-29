# Task 1 validation analysis

Run on Colab after training. The checkpoint stays on Drive. Push this evaluator to GitHub, then in a new Colab cell:

```python
%cd /content/GenAI_A1
!git pull --ff-only
!python -m evaluation.evaluate_task1 --checkpoint /content/drive/MyDrive/GenAI_A1/task1-baseline/best.pt --device cuda --output-dir /content/drive/MyDrive/GenAI_A1/task1-baseline/validation_analysis
```

The evaluator uses the architecture and alpha stored in the checkpoint. It checks validation split/manifest hashes and the model/loss/preprocessing source hashes against training provenance. It reads the checkpoint's completed epoch rather than assuming epoch 13. It does not load official test data or alter checkpoints, training code, or manifests.

For each fixed validation case, compare the corrupted input with its clean target, then compare the model output with the same target using the same L1/SSIM implementation. Positive gain means improvement: input minus output for L1/combined loss, output minus input for SSIM. Negative values indicate regression on that metric. Metrics can disagree; inspect both and the actual images.

Outputs:
- README.md: human-readable condition comparison and caveats.
- summary.json: checkpoint provenance, equal-condition aggregate and selected examples.
- per_image.csv: all 7,360 fixed validation-case measurements.
- by_condition.csv: clean, salt, blur and occlusion summaries.
- by_condition_severity.csv: all ten condition/severity groups.
- representative_grid.png: up to twelve examples, three distinct images per condition, selected in manifest order without ranking by model success.
- ssim_regressions.png: up to four unique-image corrupted cases with the worst negative SSIM gains; omitted if none exist. Clean cases are excluded from this selection because their identity baseline is already perfect. The per-condition tables still quantify clean-image degradation.

Representative severity selection covers low/medium/high for each corruption. Grids show target, input, output and unamplified absolute error on a fixed [0,1] display scale. Selected cases and exact indices are saved in summary.json. Automatic rankings are candidates for discussion; they do not explain why a failure happened or replace the required student interpretation.

Overall averages equally weight the four conditions, matching the training validation selection rule. Raw averaging of all 7,360 cases would weight clean images less. No uncertainty estimates or claims of statistical significance are made.

The full evaluation uses 736 validation images × 10 cases. `--val-images N` exists only for smoke checks; those reports are explicitly labelled. Debug-trained checkpoints are also flagged. A non-empty output directory is protected against overwrite; use a new directory for reruns.

Next: read the tables and grids before changing architecture or designing the Optuna search. Do not interpret validation outputs as final test results.
