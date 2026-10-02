import unittest
import optuna
from optimization.optimize_task2_classifier import score as classifier_score, distributions as classifier_distributions
from optimization.optimize_task2_specialists import score as specialist_score, distributions as specialist_distributions


class Task2OptimizationTest(unittest.TestCase):
    def test_classifier_ranking_objective(self):
        a = {'macro_f1': 0.85, 'cross_entropy': 0.3}
        b = {'macro_f1': 0.70, 'cross_entropy': 0.1}
        # Higher macro F1 must yield a lower (better) minimization score
        self.assertAlmostEqual(classifier_score(a), 0.15)
        self.assertAlmostEqual(classifier_score(b), 0.30)
        self.assertLess(classifier_score(a), classifier_score(b))

    def test_classifier_optuna_study_reconstruction(self):
        params = dict(learning_rate=0.001, batch_size=32, dropout=0.1, weight_decay=0.0001, channels_preset='standard')
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4))
        study.add_trial(optuna.trial.create_trial(params=params, distributions=classifier_distributions(), value=0.15))
        trial = study.ask(classifier_distributions())
        self.assertEqual(set(trial.params), set(params))
        self.assertEqual(study.best_trial.params, params)

    def test_specialist_ranking_ignores_training_alpha(self):
        # Specialist ranking must be based on a fixed 0.8*L1 + 0.2*(1-SSIM) metric
        a = {'val_l1': 0.05, 'val_ssim': 0.70, 'val_loss': 0.1}
        b = {'val_l1': 0.05, 'val_ssim': 0.70, 'val_loss': 0.9}
        self.assertAlmostEqual(specialist_score(a), 0.8 * 0.05 + 0.2 * 0.30)
        self.assertEqual(specialist_score(a), specialist_score(b))
        self.assertLess(specialist_score(dict(val_l1=0.03, val_ssim=0.85)), specialist_score(a))

    def test_specialist_optuna_study_reconstruction(self):
        params = dict(learning_rate=0.0007, batch_size=16, latent_channels=32, dropout=0.0, alpha=0.5)
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4))
        study.add_trial(optuna.trial.create_trial(params=params, distributions=specialist_distributions(), value=0.10))
        trial = study.ask(specialist_distributions())
        self.assertEqual(set(trial.params), set(params))
        self.assertEqual(study.best_trial.params, params)


if __name__ == '__main__':
    unittest.main()
