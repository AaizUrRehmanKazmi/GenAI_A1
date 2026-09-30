# Task 1 Optuna screening

## Evidence and scope
The full 16×16 run improved validation loss from 0.09403 at epoch15 to 0.09223 at epoch20, with fluctuations. It remains weaker than returning the input on the aggregate fixed-alpha metric (0.08531), despite gains on severe salt noise and occlusion. Preserve this baseline. This search is limited screening, not a claim to find the globally optimal model or complete final assignment evidence.

## Search protocol
Eight sequential trials, five complete epochs per trial, same seed42 and full train/validation splits. No pruning: short trials can favor fast learners, and early pruning would amplify this. Trial0 reruns the current baseline under the same budget. Three further startup trials sample randomly; TPE uses completed observations for subsequent suggestions. The eight-trial budget is modest relative to five search dimensions.

| Parameter | Search space |
|---|---|
| Learning rate | log-uniform 0.0002 to 0.002 |
| Batch size | 8, 16 |
| Latent channels | 16, 32, 64 (16×16 spatial grid) |
| Dropout | 0, 0.05, 0.1 |
| Training alpha | 0.5, 0.65, 0.8, 0.9 |

These are provisional bounded choices for the available GPU/time, not research-established optima. Latent channels change scalar compression and parameter count. Equal epochs preserve image exposures; batch size changes optimizer update count and runtime. The 4-hour session cap is soft, not a Colab availability guarantee.

## Fair ranking
Rank the FINAL fifth-epoch checkpoint by `0.8*validation_L1 + 0.2*(1-validation_SSIM)`, equally weighting conditions. Training uses each trial's suggested alpha. Never rank directly by trial val_loss, because that changes its meaning with alpha. Compare input identity baseline computed on the same fixed validation cases; inspect per-condition metrics. `last.pt` is the screened checkpoint. `best.pt` uses the training-alpha criterion and can refer to another epoch; do not substitute it for the ranked checkpoint.

## Persistence and tracking
Run optimization.optimize_task1 with --output-dir pointing to a NEW Drive folder, normally task1-optuna. Each trial reuses the unchanged spatial16 trainer with target20 and pause-after5, including checkpoint/RNG recovery and MLflow logging. JSON study state is atomically replaced; each completed trial's parameters, distributions (defined in the source), and objective reconstruct an in-memory Optuna study. The sampler is seeded with 42+trial number, making subsequent proposals reproducible from saved observations without serializing Python objects or relying on SQLite locking on Drive. Use only ONE process per output folder.

Rerun the same command after reconnecting and restoring the dataset. --trials is the total number of completed trials requested. Active trials retain their sampled parameters and resume last.pt. Protocol changes, including source/version/config/subset changes, are rejected. Abrupt interruption before the first checkpoint requires preserving/renaming the partial trial folder before retry. Errors fail visibly rather than counting incomplete results as completed trials. Keep the whole output folder; an abrupt runtime loss can still lose writes not yet synchronized to Drive.

Outputs include study.json, leaderboard.json, selected_config.yaml, per-trial YAML files and per-trial checkpoint/history/MLflow folders. MLflow records the training metrics; leaderboard.json records the fixed-alpha ranking. Debug subset runs are labeled and must use a separate folder.

## Next decision
Use notebooks/task1_optuna_colab.ipynb. Share leaderboard.json after screening. Selected config is provisional, not a trained final winner. Review and continue the top candidates to the SAME total20-epoch budget alongside the existing baseline, using the same fixed-alpha criterion for comparison. Inspect clean/blur degradation and severe-corruption gains. Do not select the final model or claim improvement from five-epoch ranking alone. Do not touch the official test set until choices are frozen.

## Alternatives considered
Grid search grows quickly over five dimensions. Random search is a useful simpler alternative but does not adapt to observed trials. TPE is used for this small sequential experiment; eight trials are insufficient to claim it outperforms those alternatives. Larger studies and multiple seeds would strengthen conclusions but exceed this initial screening budget.

API reference: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html
