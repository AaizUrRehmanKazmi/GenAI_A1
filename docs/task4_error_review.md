# Task 4 broader error review

Evaluated GAN fine-tuning best checkpoint (fine-tuning epoch1), all160 existing validation pairs. CPU equal-style means reproduce L1≈0.08778845, SSIM≈0.557516. No official test data used. Representative panels select four equally spaced stored-order examples per style; worst panels select three lowest-SSIM cases per style. No randomized representativeness claim.

White-output baseline equal-style L1≈0.178906, SSIM≈0.479363. Learned output is better in both metrics on each style, but blank backgrounds contribute substantially to absolute SSIM. This does not quantify their entire contribution.

Visual review of 12 representative and nine worst cases: many photo/sketch landmarks roughly coincide, with artistic differences in contours/features. There is no obvious systematic wrong pairing in these panels, but this is not a quantitative registration audit. Errors strongly follow missing hair strokes, eye/mouth detail and facial contours. Hair-rich cases dominate several low-SSIM examples. Outputs retain broad face shapes but smooth linework, including seemingly well-aligned examples. Alignment alone is not established as the cause.

Next bounded hypothesis: compare an edge-sensitive reconstruction term against matched reconstruction control, using target sketch gradients (not photo edges). This is a proposal, not a demonstrated improvement, and must preserve validation/test discipline. Do not launch more epochs of the unchanged GAN expecting different evidence. Preserve current checkpoints and report the unsuccessful adversarial ablation.

Artifacts: artifacts/task4-error-review/report.json, representative.png, worst.png. Script: scripts/review_task4_errors.py.
