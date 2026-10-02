import unittest
import torch
from evaluation.evaluate_task2 import summarize_task2, select_examples_task2


class Task2AnalysisTest(unittest.TestCase):
    def sample_rows(self):
        rows = []
        # 4 conditions, 3 images per condition
        # 3 clean images (1 case each) + 3 * 3 = 9 cases for each corrupted condition
        # Total = 3 + 9 + 9 + 9 = 30 rows
        for label, condition in enumerate(['clean', 'salt', 'blur', 'occlusion']):
            for image in range(3):
                severities = ['clean'] if label == 0 else ['low', 'medium', 'high']
                for sev_idx, severity in enumerate(severities):
                    # Make image 0 misclassified, others correct
                    if image == 0:
                        pred_label = (label + 1) % 4
                        pred_cond = ['clean', 'salt', 'blur', 'occlusion'][pred_label]
                        correct = False
                    else:
                        pred_label = label
                        pred_cond = condition
                        correct = True

                    inp_ssim = 1.0 if label == 0 else 0.5
                    orc_ssim = 1.0 if label == 0 else 0.7
                    prd_ssim = orc_ssim if correct else (0.8 if label == 0 else 0.4)

                    inp_l1 = 0.0 if label == 0 else 0.2
                    orc_l1 = 0.0 if label == 0 else 0.1
                    prd_l1 = orc_l1 if correct else (0.05 if label == 0 else 0.25)

                    row = {
                        'index': len(rows),
                        'image_id': f'img_{condition}_{image}',
                        'path': f'/path/{condition}_{image}.jpg',
                        'true_condition': condition,
                        'severity': severity,
                        'true_label': label,
                        'pred_condition': pred_cond,
                        'pred_label': pred_label,
                        'pred_confidence': 0.95 if correct else 0.6,
                        'correct_routing': correct,
                        'input_l1': inp_l1,
                        'input_ssim': inp_ssim,
                        'input_loss': inp_l1 * 0.5 + (1 - inp_ssim) * 0.5,
                        'oracle_l1': orc_l1,
                        'oracle_ssim': orc_ssim,
                        'oracle_loss': orc_l1 * 0.5 + (1 - orc_ssim) * 0.5,
                        'oracle_l1_gain': inp_l1 - orc_l1,
                        'oracle_ssim_gain': orc_ssim - inp_ssim,
                        'oracle_loss_gain': (inp_l1 * 0.5 + (1 - inp_ssim) * 0.5) - (orc_l1 * 0.5 + (1 - orc_ssim) * 0.5),
                        'pred_l1': prd_l1,
                        'pred_ssim': prd_ssim,
                        'pred_loss': prd_l1 * 0.5 + (1 - prd_ssim) * 0.5,
                        'pred_l1_gain': inp_l1 - prd_l1,
                        'pred_ssim_gain': prd_ssim - inp_ssim,
                        'pred_loss_gain': (inp_l1 * 0.5 + (1 - inp_ssim) * 0.5) - (prd_l1 * 0.5 + (1 - prd_ssim) * 0.5),
                        'l1_routing_cost': prd_l1 - orc_l1,
                        'ssim_routing_cost': orc_ssim - prd_ssim,
                    }
                    rows.append(row)
        return rows

    def test_equal_condition_weighting_and_taxonomy(self):
        rows = self.sample_rows()
        cm = torch.zeros(4, 4, dtype=torch.int64)
        for r in rows:
            cm[r['true_label'], r['pred_label']] += 1
        summary = summarize_task2(rows, cm, cross_entropy=0.25)

        # Check condition-balanced weights each condition equally (0.25 each)
        self.assertEqual(len(summary['by_condition']), 4)
        self.assertEqual(len(summary['by_condition_severity']), 10)

        # Each condition has 3 images: clean has 3 cases, others have 9 cases
        self.assertEqual(summary['by_condition'][0]['count'], 3)
        self.assertEqual(summary['by_condition'][1]['count'], 9)

        # Check routing taxonomy counts
        tax = summary['routing_taxonomy']
        self.assertEqual(tax['total_cases'], 30)
        # Image 0 was misclassified: clean image 0 (1 case) + salt img 0 (3 cases) + blur img 0 (3 cases) + occ img 0 (3 cases) = 10 misrouted cases
        self.assertEqual(tax['misrouted_count'], 10)
        self.assertEqual(tax['correct_routing_count'], 20)
        self.assertEqual(tax['clean_as_corrupted_count'], 1)  # clean img 0 predicted as salt
        self.assertEqual(tax['corrupted_as_clean_count'], 3)  # occlusion img 0 predicted as clean (label (3+1)%4 = 0)
        self.assertEqual(tax['cross_corruption_count'], 6)    # salt -> blur (3), blur -> occ (3)

        # Accuracy
        self.assertAlmostEqual(tax['overall_accuracy'], 20 / 30)

    def test_representatives_and_routing_failures_selection(self):
        rows = self.sample_rows()
        reps, failures = select_examples_task2(rows)

        # 3 representatives per condition = 12 total
        self.assertEqual(len(reps), 12)
        cond_counts = {}
        for r in reps:
            cond_counts[r['true_condition']] = cond_counts.get(r['true_condition'], 0) + 1
        self.assertEqual(cond_counts, {'clean': 3, 'salt': 3, 'blur': 3, 'occlusion': 3})

        # Routing failures should have selected misrouted cases with unique images
        self.assertGreater(len(failures), 0)
        self.assertEqual(len({f['image_id'] for f in failures}), len(failures))
        self.assertTrue(all(not f['correct_routing'] for f in failures))

    def test_clean_identity_routing_cost(self):
        # When clean image is correctly predicted, routing cost is zero
        rows = self.sample_rows()
        clean_correct = [r for r in rows if r['true_condition'] == 'clean' and r['correct_routing']]
        self.assertTrue(all(r['ssim_routing_cost'] == 0.0 for r in clean_correct))
        self.assertTrue(all(r['l1_routing_cost'] == 0.0 for r in clean_correct))


if __name__ == '__main__':
    unittest.main()
