import unittest
from evaluation.evaluate_task2 import summarize_task2, select_examples_task2, verify_checkpoint
from src.data.pets_dataset import REPO_ROOT
from evaluation.evaluate_task1 import digest
class EvaluationTest(unittest.TestCase):
    def test_balanced_and_no_fabricated_failures(self):
        rows=[]
        for label,c in enumerate(['clean','salt','blur','occlusion']):
            for severity in (['clean'] if label==0 else ['low','medium','high']):
                r=dict(index=len(rows),image_id=str(len(rows)),true_condition=c,pred_condition=c,severity=severity,correct_routing=True)
                for mode in ['input','oracle','pred','task1']:
                    for m in ['l1','ssim','loss']:r[mode+'_'+m]=float(label)
                for mode in ['oracle','pred']:
                    for m in ['l1','ssim','loss']:r[mode+'_'+m+'_gain']=0.
                r.update(ssim_routing_cost=0.,l1_routing_cost=0.)
                rows.append(r)
        result=summarize_task2(rows,[[1,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,3]],0.)
        self.assertEqual(result['condition_balanced']['task1_l1'],1.5)
        self.assertEqual(result['routing_taxonomy']['misrouted_count'],0)
        self.assertEqual(select_examples_task2(rows)[1],[])
    def test_provenance_rejects_changed_validation(self):
        split=REPO_ROOT/'data/splits/pets_val.json';manifest=REPO_ROOT/'data/manifests/pets_val_corruptions.json'
        paths=['src/models/corruption_classifier.py','src/data/pets_dataset.py','src/data/corruptions.py','data/splits/pets_val.json','data/manifests/pets_val_corruptions.json']
        saved={'data_signature':{'hashes':{p:digest(REPO_ROOT/p) for p in paths},'train_limit':0,'val_images':0}}
        self.assertFalse(verify_checkpoint(saved,'classifier',split,manifest))
        saved['data_signature']['hashes']['data/splits/pets_val.json']='wrong'
        with self.assertRaises(ValueError):verify_checkpoint(saved,'classifier',split,manifest)
