# Task 3 provisional Optuna protocol

Run notebooks/task3_optuna_kaggle.ipynb after pushing source. Attach the complete
selected task2-delivery bundle. Eight sequential trials screen for five TOTAL
epochs (two warm-up, three joint) on the full training/validation split.
The original ten-epoch baseline is preserved. Every new trial starts from the
same hash-verified Task 2 checkpoints, same seed and balanced batching. No trial
starts from the trained Task 3 baseline or another trial's tuned weights.

Search: joint LR log-uniform 5e-6..5e-5 (below fixed warm-up LR 1e-4), temperature
0.7/1/1.5/2, CE weight .03/.1/.3, balance weight 0/.01/.05, L1 fraction
.65/.8/.9 with complementary SSIM weight. Trial zero is baseline settings.
Batch size eight and two warm-up epochs stay fixed. These ranges are provisional
experimental choices around the baseline, not claims of optimality. Zero balance
is an ablation. Sharper/softer temperatures investigate near-hard versus mixed
routing. No pruning is implemented; each completed trial gets the same epochs.

Ranking: final screening epoch condition-balanced .8 L1 + .2(1-SSIM), independent
of training coefficients. Keep target epochs ten in each YAML; --stop-after-epoch
pauses training. Continue selected last.pt with its original YAML to ten before
comparing with the ten-epoch baseline. Never compare five vs ten as equal budgets.

Atomic study.json preserves pending and completed trial records. On resume,
completed observations reconstruct Optuna TPE with seed42+trial number and four
startup trials. This is a documented sequential reconstruction protocol, not a
persistent sampler RNG stream or a multi-worker study. Source/data/config/version
changes are rejected. Rerun command to resume a pending trial. --trials is total,
not additional. Full backup includes study, YAMLs, histories and checkpoints.
The soft two-hour budget may stop before eight trials; save/download regardless.

Tests: objective/config/search dimensions passed; tiny CPU full lifecycle tested
with one screening trial, idempotent rerun and subsequent continuation. Full GPU
search remains user-run; tiny scores are not validation evidence.
