import copy,unittest
import torch
from src.models.task4_candidate import CandidateDiscriminator
from training.task4_bn_policy import generator_discriminator_phase

class BNPolicyTest(unittest.TestCase):
    def test_batch_logits_gradients_and_state(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        d=CandidateDiscriminator(base_channels=4,style_dim=4).eval()
        reference=copy.deepcopy(d).train()
        x=torch.rand(3,3,128,128);fake=torch.rand_like(x,requires_grad=True);s=torch.arange(3)
        state={k:v.clone() for k,v in d.state_dict().items()}
        expected=reference(x,fake.detach(),s).detach()
        with generator_discriminator_phase(d):
            out=d(x,fake,s)
            self.assertTrue(torch.equal(expected,out.detach()))
            out.mean().backward()
            self.assertTrue(all(p.grad is None for p in d.parameters()))
        self.assertGreater(fake.grad.abs().sum().item(),0)
        self.assertTrue(all(torch.equal(v,state[k]) for k,v in d.state_dict().items()))
        self.assertFalse(d.training)
        self.assertTrue(all(p.requires_grad for p in d.parameters()))

if __name__=='__main__':unittest.main()
