# Task 4 model baseline

StyleUNet: four stride-two encoder stages (128 to 8 pixels), three upsample/conv
stages with encoder skip concatenation, final upsample/conv/sigmoid. RGB input
and output [0,1]. Learned three-category embedding is broadcast and concatenated
to photo channels before encoding. StylePatchGAN has a separate learned embedding
concatenated with photo and sketch; four feature convolutions and final patch
logits produce 14x14 scores with nominal 70x70 receptive fields.

Provisional base32, style embedding8, generator bottleneck decoder dropout .2.
No normalization in this initial compact model; revisit if training stability
requires it. Nearest-upsample plus convolution avoids introducing transposed
convolution as an additional variable. This is an assignment-aligned baseline,
not a claim these choices outperform alternatives. Unlike restoration bottleneck
models, the assignment explicitly allows U-Net skips for paired sketch synthesis.

BCE-with-logits adversarial losses and L1 reconstruction weight100 follow the
assignment's suggested starting objective. Discriminator real/fake and generator
adversarial/reconstruction terms remain separate for logging. Detach generated
sketch for discriminator update, freeze discriminator parameters for generator
update, then re-enable for the next discriminator update. Style labels int64
0/1/2 correspond to UI styles1/2/3. No pretrained GAN weights used.

CPU tests verify output range/shape, patch dimensions, different style outputs,
nonzero gradients into both embeddings, and isolated D/G backward/optimizer steps.
These are synthetic implementation tests, not trained style fidelity. Full trainer,
real-pair smoke diagnostic, Optuna, validation and ONNX are still pending.

## Real-pair diagnostic

`python -m scripts.check_task4_gan` ran 20 CPU optimizer steps on three fixed
training pairs, one per style. Both style embeddings changed; discriminator
updates did not backpropagate into G and generator updates did not accumulate
D parameter gradients. All losses/gradients were finite. Training-pair L1 fell
from .33268 to .16281. This short diagnostic is not validation or style fidelity
evidence: outputs remain blurred and primitive. Report/preview are under
artifacts/task4-diagnostic. No checkpoint promoted or original weights modified.
Full resumable training and Kaggle notebook remain next.
