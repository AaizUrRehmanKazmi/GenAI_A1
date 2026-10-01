import unittest
import torch
from src.data.classifier_batches import balanced_batch
from src.models.corruption_classifier import CorruptionClassifier
from evaluation.classifier_metrics import metrics_from_confusion
class ClassifierTest(unittest.TestCase):
    def test_balance_and_target_preservation(self):
        torch.set_num_threads(1)
        data=[{'image':torch.rand(3,128,128)} for _ in range(3)]
        original=[r['image'].clone() for r in data]
        torch.manual_seed(12);x,y=balanced_batch(data,[0,1,2])
        self.assertEqual(torch.bincount(y).tolist(),[3]*4)
        torch.manual_seed(12);xx,yy=balanced_batch(data,[0,1,2])
        self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy))
        for r,o in zip(data,original):self.assertTrue(torch.equal(r['image'],o))
        m=CorruptionClassifier();loss=torch.nn.functional.cross_entropy(m(x),y);loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_metrics(self):
        r=metrics_from_confusion([[2,0,0,0],[0,1,1,0],[0,0,2,0],[0,0,0,2]])
        self.assertEqual(r['accuracy'],7/8)
        self.assertEqual(r['recall'],[1,.5,1,1])
        self.assertEqual(r['normalized_confusion'][1],[0,.5,.5,0])
        self.assertEqual(metrics_from_confusion(torch.zeros(4,4))['macro_f1'],0)
