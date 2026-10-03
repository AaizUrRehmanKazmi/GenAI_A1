import unittest
import yaml
from pathlib import Path
from optimization.optimize_task3 import trial_config, score, distributions

class Task3SearchTest(unittest.TestCase):
    def test_config_keeps_total_and_baseline(self):
        base=yaml.safe_load(Path('configs/task3.yaml').read_text())
        p={'joint_learning_rate':1e-5,'temperature':1.5,'classification_weight':.03,'balance_weight':0.,'alpha':.65}
        cfg=trial_config(base,p)
        self.assertEqual(cfg['training']['warmup_epochs'],2)
        self.assertEqual(cfg['training']['joint_epochs'],8)
        self.assertEqual(base['model']['temperature'],1.)
        self.assertAlmostEqual(cfg['loss']['l1_weight']+cfg['loss']['ssim_weight'],1.)
    def test_fixed_objective(self):
        self.assertAlmostEqual(score({'val_l1':.1,'val_ssim':.8,'val_loss':99}),.12)
    def test_required_search_parameters(self):
        self.assertEqual(set(distributions()),{'joint_learning_rate','temperature','classification_weight','balance_weight','alpha'})
if __name__=='__main__':unittest.main()
