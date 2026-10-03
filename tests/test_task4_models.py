import unittest
import torch
from src.models.generator_unet import StyleUNet
from src.models.discriminator_patchgan import StylePatchGAN
from src.losses.gan_loss import discriminator_loss,generator_loss
class Task4ModelsTest(unittest.TestCase):
    def test_shapes_styles_and_both_updates(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        g=StyleUNet(8,4,0);d=StylePatchGAN(8,4)
        x=torch.rand(1,3,128,128).repeat(3,1,1,1);y=torch.rand_like(x);style=torch.arange(3)
        output=g(x,style)
        self.assertEqual(output.shape,x.shape);self.assertTrue(((output>=0)&(output<=1)).all())
        self.assertFalse(torch.equal(output[0],output[1]))
        logits=d(x,output,style);self.assertEqual(logits.shape,(3,1,14,14))
        optd=torch.optim.Adam(d.parameters());optg=torch.optim.Adam(g.parameters())
        optd.zero_grad();dl=discriminator_loss(d(x,y,style),d(x,output.detach(),style));dl['loss'].backward()
        self.assertTrue(all(p.grad is None for p in g.parameters()))
        self.assertGreater(d.style_embedding.weight.grad.abs().sum().item(),0);optd.step()
        d.requires_grad_(False);optg.zero_grad();gl=generator_loss(d(x,output,style),output,y);gl['loss'].backward()
        self.assertGreater(g.style_embedding.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in g.parameters()));optg.step()
    def test_invalid_style_shape(self):
        with self.assertRaises(ValueError):StyleUNet(8)(torch.zeros(1,3,128,128),torch.tensor([1.]))
if __name__=='__main__':unittest.main()
