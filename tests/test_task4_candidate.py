import unittest
import torch
from src.models.task4_candidate import CandidateGenerator, CandidateDiscriminator

class CandidateTest(unittest.TestCase):
    def test_range_gradients_and_frozen_buffers(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        g=CandidateGenerator(base_channels=4,style_dim=4,dropout=0)
        d=CandidateDiscriminator(base_channels=4,style_dim=4)
        x=torch.rand(3,3,128,128);s=torch.arange(3)
        out=g(x,s)
        self.assertEqual(out.shape,x.shape)
        self.assertTrue(bool(((out>=0)&(out<=1)).all()))
        d.eval();d.requires_grad_(False)
        buffers={n:b.clone() for n,b in d.named_buffers()}
        d(x,out,s).mean().backward()
        self.assertGreater(g.embed.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in d.parameters()))
        self.assertTrue(all(torch.equal(buffers[n],b) for n,b in d.named_buffers()))
        g.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(g(x,s),g(x,s)))

if __name__=='__main__':unittest.main()
