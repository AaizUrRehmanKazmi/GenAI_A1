"""Generator-phase discriminator context: batch statistics, frozen state/weights."""
from contextlib import contextmanager
import torch

@contextmanager
def generator_discriminator_phase(discriminator):
    modes={m:m.training for m in discriminator.modules()}
    flags={p:p.requires_grad for p in discriminator.parameters()}
    buffers={n:b.detach().clone() for n,b in discriminator.named_buffers()}
    discriminator.train()
    discriminator.requires_grad_(False)
    try:
        # Backward must execute inside this context, before restoring buffers.
        yield
    finally:
        with torch.no_grad():
            for n,b in discriminator.named_buffers():b.copy_(buffers[n])
        for p,flag in flags.items():p.requires_grad_(flag)
        for m,mode in modes.items():m.training=mode
