import unittest
import optuna
from optimization.optimize_task1 import score, distributions

class SearchTest(unittest.TestCase):
    def test_ranking_ignores_training_alpha_loss(self):
        a={'val_l1':.04,'val_ssim':.75,'val_loss':.01}
        b={'val_l1':.04,'val_ssim':.75,'val_loss':.9}
        self.assertAlmostEqual(score(a), .082)
        self.assertEqual(score(a),score(b))
        self.assertLess(score(dict(val_l1=.03,val_ssim=.8)),score(a))

    def test_completed_trials_can_reconstruct_search(self):
        params=dict(learning_rate=.001,batch_size=16,latent_channels=32,dropout=0.,alpha=.8)
        s=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=43,n_startup_trials=4))
        s.add_trial(optuna.trial.create_trial(params=params,distributions=distributions(),value=.1))
        t=s.ask(distributions())
        self.assertEqual(set(t.params),set(params))
        self.assertEqual(s.best_trial.params,params)
