import unittest
import torch
from src.losses.sketch_gradient import sketch_gradient_loss

class GradientTest(unittest.TestCase):
    def test_identity_and_offset(self):
        x=torch.rand(2,3,8,8)
        self.assertEqual(sketch_gradient_loss(x,x).item(),0)
        self.assertLess(sketch_gradient_loss(x+.1,x).item(),1e-7)
    def test_missing_line_gradient(self):
        target=torch.zeros(1,1,8,8);target[:,:,:,4]=1
        pred=torch.zeros_like(target,requires_grad=True)
        loss=sketch_gradient_loss(pred,target)
        self.assertGreater(loss.item(),0);loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(pred.grad.abs().sum().item(),0)
    def test_invalid(self):
        with self.assertRaises(ValueError):sketch_gradient_loss(torch.zeros(1,1,1,8),torch.zeros(1,1,1,8))
if __name__=='__main__':unittest.main()
