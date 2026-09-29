import json
import unittest
import torch
from src.data.corruptions import CLASSES, apply_corruption, corrupt, make_spec


class CorruptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_replay_contract_and_target_preservation(self):
        image = torch.rand(3, 128, 128)
        original = image.clone()
        for condition in CLASSES:
            with self.subTest(condition=condition):
                out, spec = corrupt(image, seed=42, condition=condition)
                self.assertTrue(torch.equal(out, apply_corruption(image, json.loads(json.dumps(spec)))))
                self.assertTrue(torch.equal(image, original))
                self.assertNotEqual(out.data_ptr(), image.data_ptr())
                self.assertEqual(out.shape, image.shape)
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue((out >= 0).all() and (out <= 1).all())
                if condition == 'clean':
                    self.assertTrue(torch.equal(out, image))

    def test_salt_pixel_probability_and_equal_colours(self):
        out, _ = corrupt(torch.full((3, 128, 128), 0.5), seed=11,
                         condition='salt', probability=0.15)
        self.assertTrue(torch.equal(out[0], out[1]) and torch.equal(out[1], out[2]))
        black, white = (out[0] == 0).float().mean(), (out[0] == 1).float().mean()
        self.assertAlmostEqual(black.item(), 0.075, delta=0.01)
        self.assertAlmostEqual(white.item(), 0.075, delta=0.01)

    def test_blur_constant_and_impulse(self):
        constant = torch.full((3, 128, 128), 0.5)
        out, _ = corrupt(constant, seed=0, condition='blur', kernel_size=7, sigma=2.5)
        self.assertTrue(torch.allclose(out, constant, atol=1e-6))
        impulse = torch.zeros_like(constant)
        impulse[0, 64, 64] = 1
        out, _ = corrupt(impulse, seed=0, condition='blur', kernel_size=3, sigma=0.7)
        self.assertAlmostEqual(out.sum().item(), 1, places=6)
        self.assertGreater(out[0, 64, 63].item(), 0)
        self.assertEqual(out[1:].sum().item(), 0)
        self.assertTrue(torch.allclose(out[0, 63:66, 63:66], out[0, 63:66, 63:66].flip(0)))

    def test_occlusion_fixed_severities_and_varied_seeds(self):
        image = torch.ones(3, 128, 128)
        for n, fraction in [(1, .10), (2, .20), (3, .35)]:
            for seed in range(100):
                out, spec = corrupt(image, seed=seed, condition='occlusion',
                                    rectangles=n, area_fraction=fraction)
                actual = (out[0] == 0).float().mean().item()
                self.assertEqual(len(spec['rectangles']), n)
                self.assertAlmostEqual(actual, fraction, delta=.005)
                self.assertEqual(actual, spec['actual_area_fraction'])
                self.assertGreaterEqual(actual, .10)
                self.assertLessEqual(actual, .35)

    def test_training_sampling_ranges_and_class_balance(self):
        counts = dict.fromkeys(CLASSES, 0)
        for seed in range(2000):
            spec = make_spec(None, seed=seed)
            counts[spec['condition']] += 1
            if spec['condition'] == 'salt':
                self.assertTrue(.02 <= spec['probability'] <= .15)
            if spec['condition'] == 'occlusion':
                self.assertTrue(.10 <= spec['actual_area_fraction'] <= .35)
        for count in counts.values():
            self.assertTrue(400 < count < 600, counts)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            corrupt(torch.ones(3, 32, 32), seed=42)
        for kwargs in [dict(condition='wrong'), dict(condition='salt', probability=.5),
                       dict(condition='blur', kernel_size=4),
                       dict(condition='occlusion', area_fraction=.9)]:
            with self.assertRaises(ValueError):
                make_spec(seed=42, **kwargs)
        spec = make_spec('occlusion', seed=42)
        spec['actual_area_fraction'] = .11
        with self.assertRaises(ValueError):
            apply_corruption(torch.ones(3, 128, 128), spec)
