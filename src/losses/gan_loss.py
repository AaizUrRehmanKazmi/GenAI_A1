"""Paired conditional GAN objectives; discriminator consumes logits."""
import torch
from torch.nn import functional as F

def discriminator_loss(real_logits,fake_logits):
    real=F.binary_cross_entropy_with_logits(real_logits,torch.ones_like(real_logits))
    fake=F.binary_cross_entropy_with_logits(fake_logits,torch.zeros_like(fake_logits))
    return {'loss':.5*(real+fake),'real':real,'fake':fake}

def generator_loss(fake_logits,prediction,target,l1_weight=100.):
    if l1_weight<0:raise ValueError('Negative reconstruction weight')
    adversarial=F.binary_cross_entropy_with_logits(fake_logits,torch.ones_like(fake_logits))
    reconstruction=F.l1_loss(prediction,target)
    return {'loss':adversarial+l1_weight*reconstruction,'adversarial':adversarial,'reconstruction':reconstruction}
