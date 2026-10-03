import unittest
from evaluation.evaluate_task3 import summarize, select_examples
from src.data.corruptions import CLASSES

class Task3AnalysisTest(unittest.TestCase):
    def rows(self):
        rows=[]
        for i,c in enumerate(CLASSES):
            for s in (['clean'] if i==0 else ['low','medium','high']):
                r={'index':len(rows),'image_id':str(len(rows)),'condition':c,'severity':s,'gate_correct':True,'entropy':0.,'max_weight':1.}
                for m in ['input','task2','task3']:
                    for k in ['l1','ssim','loss']:r[f'{m}_{k}']=float(i)
                for name in CLASSES:r['weight_'+name]=float(name==c)
                rows.append(r)
        return rows
    def test_equal_condition_weighting(self):
        s=summarize(self.rows())
        self.assertEqual(s['condition_balanced']['task3_loss'],1.5)
        self.assertEqual(len(s['by_condition_severity']),10)
    def test_no_fabricated_distributed_or_regression_cases(self):
        e=select_examples(self.rows())
        self.assertEqual(e['distributed'],[]);self.assertEqual(e['regressions'],[])
    def test_missing_condition_rejected(self):
        with self.assertRaises(ValueError):summarize(self.rows()[1:])
if __name__=='__main__':unittest.main()
