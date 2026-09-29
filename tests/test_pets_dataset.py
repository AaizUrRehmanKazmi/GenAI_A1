import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch.utils.data import DataLoader
from src.data.pets_dataset import PetsDataset


class PetsDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'images').mkdir()
        self.split = self.root / 'split.json'

    def dataset(self, entries, **kwargs):
        self.split.write_text(json.dumps(entries))
        return PetsDataset(self.split, self.root, **kwargs)

    def test_rgb_channels_range_and_source_unchanged(self):
        path = self.root / 'images/red.png'
        Image.new('RGB', (30, 60), (255, 0, 0)).save(path)
        before = path.read_bytes()
        ds = self.dataset(['images/red.png'])
        sample = ds[0]
        self.assertEqual(sample['image_id'], 'red')
        self.assertEqual(sample['image'].shape, (3, 128, 128))
        self.assertEqual(sample['image'].dtype, torch.float32)
        self.assertTrue(torch.all(sample['image'][0] == 1))
        self.assertTrue(torch.all(sample['image'][1:] == 0))
        self.assertTrue(torch.equal(sample['image'], ds[0]['image']))
        self.assertEqual(before, path.read_bytes())

    def test_grayscale_rgba_and_batching(self):
        Image.new('L', (20, 10), 128).save(self.root / 'images/gray.png')
        Image.new('RGBA', (10, 20), (0, 255, 0, 255)).save(self.root / 'images/rgba.png')
        ds = self.dataset(['images/gray.png', 'images/rgba.png'], interpolation='bicubic')
        batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
        self.assertEqual(batch['image'].shape, (2, 3, 128, 128))
        self.assertTrue(torch.allclose(batch['image'][0], torch.full((3, 128, 128), 128 / 255)))
        self.assertEqual(batch['image_id'], ['gray', 'rgba'])

    def test_exif_orientation(self):
        im = Image.new('RGB', (20, 40), (0, 0, 255))
        im.paste((255, 0, 0), (0, 0, 20, 20))
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise: red top becomes red right.
        im.save(self.root / 'images/oriented.jpg', exif=exif)
        image = self.dataset(['images/oriented.jpg'])[0]['image']
        self.assertGreater(image[0, 64, 110].item(), 0.9)
        self.assertGreater(image[2, 64, 15].item(), 0.9)

    def test_corrupt_image_is_not_silently_skipped(self):
        (self.root / 'images/bad.jpg').write_bytes(b'broken')
        ds = self.dataset(['images/bad.jpg'])
        with self.assertRaisesRegex(RuntimeError, 'bad.jpg'):
            ds[0]

    def test_invalid_lists_and_paths(self):
        Image.new('RGB', (1, 1)).save(self.root / 'images/a.png')
        for entries in [[], {}, [3], ['../escape.jpg'], ['/absolute.jpg'],
                        ['images/a.png', 'images/a.png']]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.dataset(entries)
        with self.assertRaises(FileNotFoundError):
            self.dataset(['images/missing.jpg'])
        with self.assertRaises(ValueError):
            self.dataset(['images/a.png'], interpolation='unknown')
