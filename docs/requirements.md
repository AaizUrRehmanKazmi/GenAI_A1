# Requirement map
The PDF is the assignment specification. The six-day DOCX is a suggested roadmap. The current user request authorizes repository, Docker and skeleton work; it does not ask for completed models, research conclusions, or a final submission.

| Area | Scaffold location | Work still required |
|---|---|---|
| Shared Pets 80/20 split, seed 42, RGB 128² | scripts/prepare_pets.py, src/data | Official trainval split; preserve official test |
| FS2K official pairs, style-stratified 15% validation | scripts/prepare_fs2k.py | Verify pairing; shared spatial augmentation |
| Runtime training corruption; fixed evaluation | configs/corruptions.yaml, scripts/generate_manifests.py | Save all parameters/seeds/rectangle coordinates; three fixed test severities |
| Task 1 universal AE | src/models/universal_ae.py | Genuine bottleneck; L1+SSIM; Optuna LR/batch/bottleneck/channels/dropout/alpha |
| Task 2 hard routing | classifier and specialist modules | Balanced classifier; independent experts; identity; oracle/predicted comparison; Optuna |
| Task 3 soft MoE | src/models/soft_moe.py | Task 2 initialization; frozen-expert warm-up; smaller-LR joint fine-tuning; four losses; Optuna |
| Task 4 style cGAN | generator/discriminator modules | U-Net/PatchGAN; learned style in both networks; separate GAN losses; Optuna |
| Evaluation | evaluation/ | Per-condition/severity metrics; Task 1 ≥12 examples and ≥4 failures; confusion matrix; routing analysis; GAN progression |
| Tracking and optimization | optimization/, src/utils/tracking.py | Select provider; record trials, best configs, checkpoints, plots and failures |
| Deployment | exports/, backend/, frontend/ | ONNX export/parity; real inference; runtime corruption; uploads/webcam; output downloads |
| Design evidence | docs/stitch/ | Make original design in Google Stitch before final UI implementation |
| Submission | report/, docs/demo_notes.md | IEEE LaTeX, GitHub URL, model links, 5–7 minute YouTube demo, AI-use appendix |
