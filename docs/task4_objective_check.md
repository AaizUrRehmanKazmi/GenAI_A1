# Task 4 fixed-pair objective diagnostic

Run from the repository root:

```sh
python -m scripts.compare_task4_objectives --raw-dir data/raw/fs2k/FS2K --output-dir artifacts/task4-objective-check --steps 500 --device cuda
```

Uses the first four training pairs of each style, no validation/test images, no augmentation, and dropout zero in both arms. Both generators start with seed 42 and receive the same full batch of 12 pairs per step. Adam learning rates are 2e-4 with betas (0.5,0.999). L1 has coefficient 100 in both arms. The GAN arm adds BCE adversarial loss and trains the discriminator. This isolates the objective under a simplified fitting setup; it is not directly comparable to full-run metrics or a claim of generalization.

Outputs include comparison.png (photo, target, L1-only, GAN+L1), per-arm histories saved every 50 steps, report.json and debug-only weights. Existing output directories are rejected to protect evidence. This diagnostic is not resumable; use a fresh output directory for another budget. Preserve the full trained baseline separately.

Interpretation: if both arms fail to fit after an adequate budget, investigate generator/optimization/preprocessing. If L1 fits but GAN does not, investigate the adversarial interaction. If both fit, investigate full-dataset capacity and optimization. Visual artifacts and low loss must be assessed together; a short run is not enough to establish the cause. Cross-style fidelity is a separate check.

Local 100-step CPU check completed: L1-only training L1 0.143747 / SSIM 0.345388; GAN+L1 training L1 0.146798 / SSIM 0.337650. This short comparison does not establish a failure mechanism; neither arm has demonstrated sharp memorization. Run the 500-step Kaggle comparison before choosing a model change. Notebook cells compile and the script completed both arms with finite losses.
