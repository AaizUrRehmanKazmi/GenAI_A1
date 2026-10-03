# Task 3 implementation stage 1

Implemented the assignment's differentiable four-branch mixture. A gate loaded
from classifier trial 005 supplies softmax(logits / temperature); outputs from
identity and three selected trial 007 experts are combined by weighted sum.
All expert architectures are loaded from their own checkpoint configurations:
salt/blur latent16 and occlusion latent32 must not be assumed identical.
Checkpoint hashes pin initialization to the selected Task 2 validation run.

Warm-up freezes experts and keeps them in evaluation mode; gradients still reach
the gate through mixture weights. Joint mode re-enables expert gradients.
Provisional config: two warm-up epochs at 1e-4, eight joint epochs at 2e-5,
batch eight (two source images, four corruption types each), temperature one.
These are starting settings, not researched optimal choices.

The four-term objective uses assignment starting coefficients .8 L1, .2 SSIM
error, .1 classification CE, .01 sum of squared deviations of batch-mean gate
weights from 1/4. CE uses raw logits; temperature affects reconstruction routing.
Balanced batches carry matching clean targets. Uniformity is encouraged across
the batch, not on every image. Hard argmax is rejected for training because it
blocks reconstruction gradients into the gate. Random initialization is excluded
by the assignment. A new architecture or entropy penalty is not introduced at
this stage; compare alternatives in subsequent research and Optuna work.

Validation performed locally: three model tests passed; real selected checkpoints
passed a one-image/four-condition diagnostic with one warm-up step and one joint
step. Gate changed in both, experts only in joint, gradients finite. No original
weights overwritten, no official test images used. Diagnostic scores are not
quality evidence. Report: artifacts/task3-diagnostic.json.

Run from repository root in the existing training environment:

```sh
python -m unittest discover -s tests -p test_soft_moe.py -v
python -m scripts.check_task3 --bundle artifacts/task2-delivery
```

Next stage: resumable two-stage trainer, balanced validation and routing weight
summaries, MLflow logging and Kaggle notebook. The existing Task 3 training,
Optuna, evaluation and ONNX entry points are still placeholders. Do not launch
long GPU training yet. This stage intentionally verifies the model first.

## Stage 2: trainer and Kaggle baseline

The stage-1 pending-trainer note above is superseded: training/train_task3_moe.py
and notebooks/task3_moe_kaggle.ipynb now implement the baseline. Optuna, standalone
final evaluation/heatmaps and complete ONNX export remain future work.

Training batches contain all four corruptions for each source image, with clean
aligned targets. Batch size 8 means two source images and eight model inputs.
Warm-up trains gate only; joint stage unfreezes all experts and reduces LR.
A single Adam optimizer registers every parameter throughout; frozen experts
receive no gradient/state updates. At stage transition existing gate state is
preserved. Checkpoints retain source/data/initialization fingerprints, config,
optimizer, RNG, source-image shuffle and cursor. Checkpoint every 50 batches,
epoch and graceful stop; a hard runtime kill can lose work since the last save.
Validation interrupted by time/stop restarts without repeating optimizer steps.

Best selection uses fixed-alpha .8 reconstruction loss, equally averaging
conditions after severities. The train loss has four terms and is not the same
metric. history.json includes weights for each condition/severity. MLflow logs
metrics and average routing weights; preview.png uses the same fixed validation
image. All ten baseline epochs remain provisional: two warm-up, eight joint.

CPU verification: pause after one training batch, resume through epochs 1–3,
and compare with uninterrupted execution. Model tensors, progress/history and
CPU RNG were exactly equal across the warm-up/joint transition. Diagnostic used
two training images and one validation image only, never official test images.
Notebook code cells compile. Kaggle CUDA execution has not yet been tested.

Upload the task2-delivery ZIP as a private dataset and attach its extracted folder.
Push source changes before importing the new notebook. Run setup, dataset prep,
smoke then baseline in order. Preserve one visible CUDA device and source revision
on resume. Save/download the full task3-baseline_backup.zip before ending a session.
