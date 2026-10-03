import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from src.data.fs2k_dataset import FS2KDataset
class PairedDatasetTest(unittest.TestCase):
    def test_shared_flip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pixels=np.zeros((128,128,3),np.uint8);pixels[:,:40]=255
            for name in ['photo.png','sketch.png']:Image.fromarray(pixels).save(root/name)
            (root/'split.json').write_text(json.dumps([{'image_id':'x','style':2,'photo':'photo.png','sketch':'sketch.png'}]))
            plain=FS2KDataset(root/'split.json',root)[0]
            with patch('torch.rand',return_value=torch.tensor(.1)):
                flipped=FS2KDataset(root/'split.json',root,augment=True)[0]
            torch.testing.assert_close(flipped['photo'],plain['photo'].flip(-1))
            torch.testing.assert_close(flipped['photo'],flipped['sketch'])
            self.assertEqual(flipped['style'],2)
    def test_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'split.json').write_text(json.dumps([{'image_id':'x','style':0,'photo':'../outside.png','sketch':'a.png'}]))
            with self.assertRaises(ValueError):FS2KDataset(root/'split.json',root)
if __name__=='__main__':unittest.main()
