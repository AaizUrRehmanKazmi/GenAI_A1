"""Exercise split integrity and failure handling with a small filename-only fixture."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import prepare_pets as pets


class PreparePetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw, self.output = self.root / 'raw', self.root / 'splits'
        (self.raw / 'annotations').mkdir(parents=True)
        (self.raw / 'images').mkdir()
        self.trainval = [f'pet_{i}' for i in range(10)]
        self.test = [f'pet_{i}' for i in range(10, 13)]
        for name, ids in [('trainval', self.trainval), ('test', self.test)]:
            (self.raw / f'annotations/{name}.txt').write_text(
                '# official list\n\n' + '\n'.join(f'{i} 1 1 1' for i in ids))
        for image_id in self.trainval + self.test + ['unlisted']:
            (self.raw / f'images/{image_id}.jpg').touch()
        constants = patch.multiple(pets, EXPECTED_TRAINVAL=10, EXPECTED_TEST=3, VALIDATION_COUNT=2)
        constants.start()
        self.addCleanup(constants.stop)

    def test_partition_test_order_and_byte_identical_rerun(self):
        result = pets.prepare(self.raw, self.output)
        train, val, test = [json.loads((self.output / f'pets_{s}.json').read_text())
                            for s in ['train', 'val', 'test']]
        self.assertEqual(result['counts'], {'train': 8, 'val': 2, 'test': 3})
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), {f'images/{i}.jpg' for i in self.trainval})
        self.assertEqual(test, [f'images/{i}.jpg' for i in self.test])
        self.assertEqual(result['unlisted_jpeg_files_excluded'], 1)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        pets.prepare(self.raw, self.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})
        candidate = self.root / 'candidate'
        pets.prepare(self.raw, candidate)
        self.assertEqual(before, {p.name: p.read_bytes() for p in candidate.iterdir()})

    def test_missing_image_prevents_output(self):
        (self.raw / 'images/pet_0.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing 1'):
            pets.prepare(self.raw, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_is_rejected(self):
        path = self.raw / 'annotations/trainval.txt'
        path.write_text(path.read_text() + '\npet_0 1 1 1\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            pets.prepare(self.raw, self.output)

    def test_overlap_is_rejected(self):
        (self.raw / 'annotations/test.txt').write_text('pet_0\npet_11\npet_12\n')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            pets.prepare(self.raw, self.output)

    def test_conflicting_output_is_not_overwritten(self):
        self.output.mkdir()
        target = self.output / 'pets_val.json'
        target.write_text('[]\n')
        with self.assertRaisesRegex(ValueError, 'Existing output differs'):
            pets.prepare(self.raw, self.output)
        self.assertEqual(target.read_text(), '[]\n')
        self.assertEqual(list(self.output.iterdir()), [target])

    def test_wrong_counts_are_rejected(self):
        (self.raw / 'annotations/test.txt').write_text('pet_10\n')
        with self.assertRaisesRegex(ValueError, 'Expected'):
            pets.prepare(self.raw, self.output)
