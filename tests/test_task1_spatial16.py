import copy
from pathlib import Path
import tempfile
import unittest
import torch
from src.models.spatial16_ae import Spatial16Autoencoder
from src.losses.reconstruction import ReconstructionLoss
from training.train_task1 import save_checkpoint, restore_checkpoint


class Spatial16Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_compression_no_bypass_and_learning(self):
        model = Spatial16Autoencoder(channels=(2,4,8,16), latent_channels=4)
        image = torch.rand(2,3,128,128)
        latent = model.encode(image)
        self.assertEqual(tuple(latent.shape), (2,4,16,16))
        self.assertTrue(torch.equal(model(image), model.decode(latent)))
        output = model(image)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(((output >= 0) & (output <= 1)).all())
        criterion = ReconstructionLoss()
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        target = torch.full_like(image, .2)
        initial = criterion(model(target), target)['loss'].mean().item()
        for _ in range(60):
            opt.zero_grad()
            loss = criterion(model(target), target)['loss'].mean()
            loss.backward()
            self.assertTrue(torch.isfinite(model.to_latent.weight.grad).all())
            opt.step()
        self.assertLess(criterion(model(target), target)['loss'].mean().item(), initial * .5)

    def test_resume_matches_next_stochastic_update(self):
        def make():
            m = Spatial16Autoencoder(channels=(2,4,8,16), latent_channels=4, dropout=.1)
            return m, torch.optim.Adam(m.parameters(), lr=.001)
        def step(m, opt):
            x = torch.rand(2,3,128,128)
            opt.zero_grad()
            ReconstructionLoss()(m(x), x)['loss'].mean().backward()
            opt.step()
        model, opt = make()
        step(model, opt)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'last.pt'
            save_checkpoint(p, model, opt, {'cursor':2}, {'architecture':'spatial'}, {}, torch.device('cpu'))
            step(model, opt)
            expected = copy.deepcopy(model.state_dict())
            other, other_opt = make()
            restore_checkpoint(p, other, other_opt, {'architecture':'spatial'}, {}, torch.device('cpu'))
            step(other, other_opt)
            for k, v in expected.items():
                self.assertTrue(torch.equal(v, other.state_dict()[k]), k)
            with self.assertRaises(ValueError):
                restore_checkpoint(p, other, other_opt, {'architecture':'vector'}, {}, torch.device('cpu'))

    def test_invalid_shapes_and_compression(self):
        for c in (0,128,768):
            with self.assertRaises(ValueError): Spatial16Autoencoder(latent_channels=c)
        with self.assertRaises(ValueError): Spatial16Autoencoder()(torch.rand(1,3,64,64))
