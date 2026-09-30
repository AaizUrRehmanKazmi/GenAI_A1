# Task 1: 16×16 latent pilot

## Reason for this experiment
The 8×8 spatial model improved on the dense vector model, but epoch-17 evaluation degraded all clean images and L1 for all blur cases. High salt noise improved on both metrics; occlusion showed mixed results. We hypothesize that retaining more spatial information will improve preservation of intact detail. This is not a confirmed cause or guaranteed fix.

## Architecture and comparison
Keep four encoder convolutions with channels 16,32,64,128, but use strides 2,2,2,1 instead of 2,2,2,2. Keep the 1×1 projection to 32 latent channels, and remove the decoder's first 2× upsampling. Subsequent three upsampling stages restore 128×128 output. Every reconstruction passes through the latent: no skips or input residuals.

Both spatial models have 203,107 parameters. The new latent has 8,192 values (32×16×16), compared with 2,048 in the old model; the input has 49,152 values. Scalar-count compression is now 6:1 rather than 24:1. Stride changes also change receptive fields and computation. Thus this tests reduced downsampling with a larger latent, not spatial resolution isolated from all other factors.

Keep seed42, dropout0, alpha0.8, Adam LR0.001, batch16, data split, preprocessing, corruption sampler and validation manifest fixed. Do not load old model weights just because their tensor shapes may match: this is a fresh experiment. Distinct architecture tags and source checks reject incompatible resume/evaluation.

## Colab workflow
Push the new files and open notebooks/task1_spatial16_colab.ipynb. Run setup, smoke check, then the fresh five-epoch pilot. Training uses a 20-epoch target but `--stop-after-epoch 5` pauses safely after five completed epochs. This operational limit does not alter checkpoint configuration, so resume can continue toward 20 epochs unchanged. Full training images and all fixed validation cases are used in the pilot.

Outputs: MyDrive/GenAI_A1/task1-spatial16. The smoke run uses task1-spatial16-smoke. Existing spatial and vector folders are preserved.

Run the analysis cell after the pilot; share history.json and validation_analysis/{summary.json,representative_grid.png}. Compare best validation loss during the first five epochs against the old 8×8 run's first five epochs (best epoch5: loss0.13908, L1 0.06577, SSIM0.56769). Inspect clean and blur severity metrics as well as salt and occlusion; use the old history for matched-epoch condition metrics. Comparing only against old epoch17 would confound architecture with training budget. The two runs need not take equal wall time.

If results justify continuing, uncomment and run the resume cell, which omits `--stop-after-epoch`. Do not run the fresh-start cell again. Evaluate the final best checkpoint into a new directory such as validation_analysis_full, updating the display cell to the same path, so pilot evidence is preserved. Never change total epochs, config or source during resume. Official test images remain unused.

## Commands from repository root
```bash
python -m training.train_task1_spatial16 --device cuda --stop-after-epoch 5 --max-hours 4.5 --output-dir /content/drive/MyDrive/GenAI_A1/task1-spatial16
python -m evaluation.evaluate_task1_spatial16 --device cuda --checkpoint /content/drive/MyDrive/GenAI_A1/task1-spatial16/best.pt --output-dir /content/drive/MyDrive/GenAI_A1/task1-spatial16/validation_analysis
```

Keep this experiment provisional. Optuna tuning and final model selection require further evidence; success on a tiny training subset does not establish generalization.
