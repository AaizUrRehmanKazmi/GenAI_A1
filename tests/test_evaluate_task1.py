import unittest
from evaluation.evaluate_task1 import summarize, select_examples


class AnalysisTest(unittest.TestCase):
    def rows(self):
        rows=[]
        for label, condition in enumerate(['clean','salt','blur','occlusion']):
            for image in range(3):
                for severity in (['clean'] if label==0 else ['low','medium','high']):
                    row={'index':len(rows), 'image_id':f'image{image}', 'condition':condition,
                         'severity':severity, 'input_l1':0.,'output_l1':float(label),
                         'input_ssim':1.,'output_ssim':.5,'input_loss':0.,'output_loss':float(label),
                         'l1_gain':-float(label),'ssim_gain':-.5,'loss_gain':-float(label)}
                    rows.append(row)
        return rows

    def test_equal_condition_weighting(self):
        result=summarize(self.rows())
        self.assertEqual(result['condition_balanced']['output_l1'],1.5)
        self.assertEqual(len(result['by_condition_severity']),10)
        self.assertEqual(result['by_condition'][0]['count'],3)
        self.assertEqual(result['by_condition'][1]['count'],9)

    def test_representatives_and_unique_corrupted_regressions(self):
        representatives, failures=select_examples(self.rows())
        self.assertEqual(len(representatives),12)
        self.assertEqual(len(failures),3)
        self.assertEqual(len({r['image_id'] for r in failures}),3)
        self.assertTrue(all(r['condition']!='clean' for r in failures))

    def test_no_fabricated_failures(self):
        rows=self.rows()
        for row in rows: row['ssim_gain']=.1
        self.assertEqual(select_examples(rows)[1],[])
