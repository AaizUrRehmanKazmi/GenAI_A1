"""Exercise fixed-image diagnostic isolation and actual resumed optimizer updates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from PIL import Image
import torch
import yaml
from training.diagnose_task1 import prepare_images, run


class DiagnosticTest(unittest.TestCase):
    def test_fixed_images_resume_and_output_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'images').mkdir()
            paths=['images/a.png','images/b.png']
            Image.new('RGB',(32,24),(20,120,200)).save(root/paths[0])
            Image.new('L',(24,32),180).save(root/paths[1])
            split=root/'train.json';split.write_text(json.dumps(paths))
            config=root/'config.yaml'
            config.write_text(yaml.safe_dump({'model':{'channels':[2,4,8,16],'latent_dim':8,'dropout':.2},'loss':{'alpha':.8}}))
            image_bytes=[(root/p).read_bytes() for p in paths]
            tensors,selected=prepare_images(split,root,2)
            self.assertEqual(selected,paths)
            self.assertEqual(tuple(tensors.shape),(2,3,128,128))
            with self.assertRaises(ValueError): prepare_images(split,root,3)
            args=SimpleNamespace(config=config,train_split=split,raw_dir=root,output_dir=root/'full',
                resume=None,device='cpu',images=2,steps=4,learning_rate=.001,seed=42,
                report_every=1,checkpoint_every=1,max_minutes=5,cpu_threads=1)
            run(args)
            partial=copy.copy(args);partial.output_dir=root/'resumed';partial.steps=2
            run(partial)
            partial.resume=partial.output_dir/'last.pt';partial.steps=4
            run(partial)
            full=torch.load(args.output_dir/'last.pt',weights_only=True)
            resumed=torch.load(partial.output_dir/'last.pt',weights_only=True)
            self.assertEqual(full['config']['model']['dropout'],0.)
            self.assertEqual(full['progress']['history'],resumed['progress']['history'])
            for key in full['model']:
                self.assertTrue(torch.equal(full['model'][key],resumed['model'][key]),key)
            self.assertEqual(image_bytes,[(root/p).read_bytes() for p in paths])
            with self.assertRaisesRegex(ValueError,'Output is not empty'): run(args)
            partial.learning_rate=.002
            with self.assertRaisesRegex(ValueError,'configuration'): run(partial)
