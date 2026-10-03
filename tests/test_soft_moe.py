import unittest
import torch
from torch import nn
from src.models.soft_moe import SoftMoE

class Gate(nn.Module):
    def __init__(self):super().__init__();self.logits=nn.Parameter(torch.zeros(4))
    def forward(self,x):return self.logits.expand(len(x),4)
class Expert(nn.Module):
    def __init__(self,v):super().__init__();self.value=nn.Parameter(torch.tensor(float(v)))
    def forward(self,x):return self.value.expand_as(x)
class SoftMoETest(unittest.TestCase):
    def test_weighted_identity_and_experts(self):
        m=SoftMoE(Gate(),Expert(.2),Expert(.4),Expert(.6))
        x=torch.ones(2,3,4,4);r=m(x)
        torch.testing.assert_close(r['output'],torch.full_like(x,.55))
        torch.testing.assert_close(r['weights'].sum(1),torch.ones(2))
    def test_warmup_then_joint_gradients(self):
        m=SoftMoE(Gate(),Expert(.2),Expert(.4),Expert(.6));x=torch.ones(2,3,4,4)
        m.set_stage('warmup');m.train();m(x)['output'].sum().backward()
        self.assertIsNotNone(m.gate.logits.grad)
        self.assertTrue(all(p.grad is None for p in m.experts.parameters()))
        self.assertTrue(all(not e.training for e in m.experts))
        m.set_stage('joint');m.zero_grad();m(x)['output'].sum().backward()
        self.assertTrue(all(p.grad is not None for p in m.experts.parameters()))
    def test_invalid_temperature_and_shared_expert(self):
        for t in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):SoftMoE(Gate(),Expert(.2),Expert(.4),Expert(.6),t)
        e=Expert(.2)
        with self.assertRaises(ValueError):SoftMoE(Gate(),e,e,Expert(.6))
if __name__=='__main__':unittest.main()
