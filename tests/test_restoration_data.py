import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import torch
from torch.utils.data import DataLoader
from src.data.manifests import generate_manifest, save_manifest
from src.data.pets_dataset import TrainingPetsDataset, EvaluationPetsDataset


class RestorationDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        Image.new('RGB', (40, 60), (128, 100, 200)).save(self.root / 'images/pet.jpg')
        self.split = self.root / 'split.json'
        self.split.write_text('["images/pet.jpg"]')
        self.file = self.root / 'manifest.json'
        save_manifest(generate_manifest(self.split, split='val'), self.file)

    def test_fresh_training_and_seed_replay(self):
        ds = TrainingPetsDataset(self.split, self.root, condition='salt')
        torch.manual_seed(42)
        a, b = ds[0], ds[0]
        self.assertNotEqual(a['spec_json'], b['spec_json'])
        self.assertFalse(torch.equal(a['input'], b['input']))
        self.assertTrue(torch.equal(a['target'], b['target']))
        torch.manual_seed(42)
        self.assertTrue(torch.equal(a['input'], ds[0]['input']))

    def test_fixed_evaluation_and_default_collation(self):
        ds = EvaluationPetsDataset(self.split, self.file, self.root)
        self.assertEqual(len(ds), 10)
        for i in range(10):
            a = ds[i]
            torch.manual_seed(i + 100)
            self.assertTrue(torch.equal(a['input'], ds[i]['input']))
            self.assertTrue(torch.equal(a['target'], ds[0]['target']))
        batch = next(iter(DataLoader(ds, batch_size=10, num_workers=0)))
        self.assertEqual(batch['input'].shape, (10, 3, 128, 128))
        self.assertEqual(batch['label'].tolist(), [0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.assertEqual(len(batch['spec_json']), 10)

    def test_manifests_do_not_read_images_and_rerun_is_identical(self):
        (self.root / 'images/pet.jpg').unlink()
        manifest = generate_manifest(self.split, split='test')
        path = self.root / 'test.json'
        save_manifest(manifest, path)
        before = path.read_bytes()
        save_manifest(generate_manifest(self.split, split='test'), path)
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(len(manifest['records']), 10)
        self.assertNotEqual(manifest['records'][1]['spec']['seed'],
                            generate_manifest(self.split, split='val')['records'][1]['spec']['seed'])

    def test_modified_cases_and_wrong_split_rejected(self):
        original = json.loads(self.file.read_text())
        changed = json.loads(self.file.read_text())
        changed['records'][1]['spec']['probability'] = .15
        with self.assertRaises(ValueError):
            save_manifest(changed, self.file)
        self.file.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            EvaluationPetsDataset(self.split, self.file, self.root)
        self.file.write_text(json.dumps(original))
        self.split.write_text('["images/other.jpg"]')
        with self.assertRaises(ValueError):
            EvaluationPetsDataset(self.split, self.file, self.root)
