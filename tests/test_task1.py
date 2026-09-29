import copy
from pathlib import Path
import tempfile
import unittest
import torch
from src.models.universal_ae import UniversalAutoencoder
from src.losses.reconstruction import ReconstructionLoss
from training.train_task1 import save_checkpoint, restore_checkpoint


class Task1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        return UniversalAutoencoder(channels=[2, 4, 8, 16], latent_dim=8, dropout=.1)

    def test_shapes_bottleneck_and_gradient(self):
        model = self.model()
        image = torch.rand(2, 3, 128, 128)
        out = model(image)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(model.encode(image).shape, (2, 8))
        self.assertTrue((out >= 0).all() and (out <= 1).all())
        loss = ReconstructionLoss()(out, image)['loss'].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.to_latent.weight.grad).all())
        self.assertGreater(model.to_latent.weight.grad.abs().sum().item(), 0)

    def test_loss_identity_and_alpha_endpoints(self):
        image = torch.rand(2, 3, 128, 128)
        result = ReconstructionLoss()(image, image)
        self.assertTrue(torch.allclose(result['loss'], torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(result['ssim'], torch.ones(2), atol=1e-6))
        prediction = torch.zeros_like(image)
        self.assertTrue(torch.allclose(ReconstructionLoss(1)(prediction, image)['loss'], image.mean((1,2,3))))
        result = ReconstructionLoss(0)(prediction, image)
        self.assertTrue(torch.equal(result['loss'], 1-result['ssim']))

    def test_tiny_learning_check(self):
        torch.manual_seed(42)
        model = self.model()
        criterion = ReconstructionLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        target = torch.full((2,3,128,128), .2)
        initial = criterion(model(target), target)['loss'].mean().item()
        for _ in range(120):
            optimizer.zero_grad()
            loss = criterion(model(target), target)['loss'].mean()
            loss.backward()
            optimizer.step()
        final = criterion(model(target), target)['loss'].mean().item()
        self.assertLess(final, initial * .25)

    def test_checkpoint_resumes_optimizer_rng_and_progress(self):
        torch.manual_seed(7)
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        criterion = ReconstructionLoss()
        def step(m, opt):
            image = torch.rand(2,3,128,128)
            opt.zero_grad()
            loss = criterion(m(image), image)['loss'].mean()
            loss.backward()
            opt.step()
        step(model, optimizer)
        progress = {'epoch': 0, 'cursor': 2, 'order': [1,0,2,3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            save_checkpoint(path, model, optimizer, progress, {'test':1}, {'hash':'x'}, torch.device('cpu'))
            step(model, optimizer)
            uninterrupted = copy.deepcopy(model.state_dict())
            resumed = self.model()
            resumed_opt = torch.optim.Adam(resumed.parameters(), lr=.001)
            restored = restore_checkpoint(path, resumed, resumed_opt, {'test':1}, {'hash':'x'}, torch.device('cpu'))
            self.assertEqual(restored, progress)
            step(resumed, resumed_opt)
            for key, value in uninterrupted.items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, resumed, resumed_opt, {'test':2}, {'hash':'x'}, torch.device('cpu'))
