# Controlled tiny-set corruption diagnostic

This follows the 16-clean-image fitting experiment. It uses the same first 16 training-list entries, seed-42 fresh initialization, vector bottleneck, zero dropout, Adam at 0.001, alpha=0.8 and default 1,000 full-batch updates. The changed training factor is input corruption: each selected image receives a fresh random seed and equal-probability choice of clean, salt, blur or occlusion on each update. Clean targets never change. Training is from scratch, not from the clean-diagnostic checkpoint.

No baseline or clean-diagnostic source/checkpoints are modified. The script reuses helpers but records their hashes to prevent accidental resume under different code. This experiment remains training-set fitting evidence, not generalization evidence.

## Fixed measurements

At step 0, every 50 updates and final exit, evaluate all 16 images at ten fixed conditions each: clean plus three severities of each corruption. These 160 cases use a separate deterministic SHA-256 seed scheme and do not advance training RNG state. The fixed-case records are saved as fixed_training_cases.json. No validation/test images or evaluation manifests are loaded.

Compare input and output L1/SSIM/loss on identical cases. Positive gain means improvement. Aggregate severities within each condition and equally weight four conditions. history.json stores overall and detailed measurements; latest_comparison.json stores the latest condition/severity tables. The clean-condition rows can be compared with the clean diagnostic's fitting scores; the aggregate over corruptions is a different task, so do not compare the two overall losses as if they had identical inputs.

At each report, save one grid per image for the first four selected training images, showing all ten fixed cases with target/input/output/absolute error. The metrics cover all selected images, not just those pictured. Inspect per-condition scores and the panels rather than relying solely on the overall mean. best.pt is selected at report intervals by fixed-training-case loss, not validation loss.

## Run in Colab

After pushing the code, with Drive mounted, GPU available and dataset restored:

```python
%cd /content/GenAI_A1
!git pull --ff-only
!python -m training.diagnose_task1_corruptions --device cuda --steps 1000 --max-minutes 60 --output-dir /content/drive/MyDrive/GenAI_A1/task1-corruption-diagnostic
```

Resume interrupted work using the same command plus:
```text
--resume /content/drive/MyDrive/GenAI_A1/task1-corruption-diagnostic/last.pt
```

A fresh run refuses a non-empty directory. last.pt is saved every 25 updates and on graceful exit. The time budget is soft; allow additional time for final evaluation, grids and checkpoint copies. Total update target can be increased deliberately, while source/model/data/loss settings must match the checkpoint. MLflow logs to this experiment's own output folder.

## Share results

After finishing, share the final printed metrics, latest_comparison.json and a final grid such as step_001000_image_01.png. If paused earlier, share the highest-numbered grid. Preserve the complete output folder alongside the clean diagnostic.

A successful fit shows that this model can restore these familiar images under the tested corruptions. It does not establish useful restoration on unseen images. Failure may involve optimization, capacity, corruption difficulty or the allotted budget; it does not isolate the bottleneck as the cause. Use these findings to select the next controlled architecture/training comparison.
