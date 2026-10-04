# Fixed-12 candidate fitting test

Use task4_fixed12_kaggle.ipynb with FS2K; no pretrained model input required. CandidateGenerator width32/style8/dropout0, seed42, Adam2e-4 betas0.5/0.999, full batch12 (first four stored training pairs per style), 100*L1, no augmentation/discriminator. 1000 total updates, one-hour invocation budget. Same candidate class used in full training. This is training-set fitting evidence, never validation evidence.

Records eval-mode L1/SSIM every50 steps and previews at250-step intervals. BatchNorm buffers are part of checkpoint. Saves weights/optimizer/RNG/history and exact source/data/device fingerprint; automatically resumes own last.pt. Abrupt interruptions may lose up to49 updates. A graceful time limit saves progress; latest preview may precede a non-reporting stop. Steps target can increase without changing training semantics. After reset restore complete backup and same visible GPU count. Keep ZIP; initially share only history.json and preview.png.

Interpretation: sharp fitting on these pairs supports further investigation of full-dataset generalization/optimization. Persistent blur prompts deeper fitting diagnostics, including training/evaluation BatchNorm behavior. Neither outcome isolates a cause alone. Do not launch another full-data experiment based on loss alone.
