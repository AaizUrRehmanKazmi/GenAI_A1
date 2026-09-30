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
from training.diagnose_task1_corruptions import prepare_images, run, fixed_cases, training_inputs


class CorruptionDiagnosticTest(unittest.TestCase):
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

    def test_fixed_cases_and_fresh_training(self):
        torch.set_num_threads(1)
        clean=torch.full((2,3,128,128),.5)
        before=clean.clone()
        paths=['images/a.png','images/b.png']
        torch.manual_seed(42)
        rng=torch.get_rng_state().clone()
        a,targets,records=fixed_cases(clean,paths,42)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        b,_,other=fixed_cases(clean,paths,42)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(records,other)
        self.assertEqual(len(records),20)
        self.assertEqual([r['condition'] for r in records[:10]],['clean']+['salt']*3+['blur']*3+['occlusion']*3)
        self.assertTrue(torch.equal(targets[0],clean[0]))
        batches=[training_inputs(clean) for _ in range(5)]
        self.assertTrue(any(not torch.equal(batches[0],v) for v in batches[1:]))
        self.assertTrue(torch.equal(clean,before))
