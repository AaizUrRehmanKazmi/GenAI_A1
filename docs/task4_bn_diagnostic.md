# Candidate discriminator mode diagnostic

Epoch 14 best checkpoint from task4-candidate_continued.zip. Six fixed batches of eight pairs: three training, three validation; generator eval output held identical between discriminator modes. Each discriminator test uses a fresh in-memory copy; source checkpoint remains unchanged. No official test images used.

Train-mode fake probabilities ranged 0.00051–0.00094; eval-mode 0.544–0.586. Generator BCE adversarial loss ranged 7.12–7.76 in train mode versus 0.586–0.674 in eval mode. All image gradients finite; eval-mode RMS gradients were smaller in every tested batch. This directly establishes substantial discriminator mode dependence on these batches. It does not establish that this alone causes grid artifacts or that any fix improves validation.

The integration introduced eval mode for discriminator calls during generator updates to freeze BatchNorm running buffers. This changed normalization behavior, not just buffer mutation, so the generator was optimized against a different discriminator computation than the one trained on real/fake batches. Correction should keep batch-statistic normalization in both optimization phases while freezing discriminator parameters for the generator phase. If buffers must be preserved, snapshot/restore BatchNorm buffers around that phase instead of switching to eval mode. Test this in a separately versioned trainer/configuration to preserve existing checkpoint fingerprints. Prefer a fresh controlled run; do not silently resume under changed training semantics.

Reproduce: python -m scripts.check_task4_bn --checkpoint artifacts/task4-candidate-bn/best.pt --output artifacts/task4-candidate-bn/report.json
