import unittest
from pathlib import Path
import yaml
from optimization.optimize_task4 import baseline_params, trial_config, distributions, score

class Task4SearchTest(unittest.TestCase):
    def test_baseline_and_isolated_config(self):
        base=yaml.safe_load(Path('configs/task4.yaml').read_text())
        params=baseline_params(base)
        self.assertEqual(trial_config(base,params),base)
        params.update(base_channels=48,style_dim=16,l1_weight=50.)
        cfg=trial_config(base,params)
        self.assertEqual(cfg['generator']['style_dim'],16)
        self.assertEqual(cfg['discriminator']['base_channels'],48)
        self.assertEqual(base['generator']['base_channels'],32)
        self.assertEqual(cfg['training']['epochs'],30)
        self.assertEqual(set(params),set(distributions()))
    def test_objective_independent_of_training_loss(self):
        self.assertAlmostEqual(score(dict(val_l1=.1,val_ssim=.8,g_loss=999)),.12)

if __name__=='__main__':unittest.main()
