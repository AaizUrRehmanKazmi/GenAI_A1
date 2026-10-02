import unittest
import torch
from torch import nn
from src.models.hard_router import HardRouter
from src.models.specialist_ae import SpecialistAutoencoder
from src.data.pets_dataset import EvaluationPetsDataset, REPO_ROOT
from training.specialist_validation import select_condition
from torch.utils.data import Subset
class Spy(nn.Module):
    def __init__(self,value):super().__init__();self.value=value;self.calls=0
    def forward(self,x):self.calls+=1;return torch.full_like(x,self.value)
class Classifier(nn.Module):
    def forward(self,x):return torch.eye(4)[[3,0,2,1]]*10
class SpecialistsTest(unittest.TestCase):
    def test_routing_and_identity(self):
        x=torch.rand(4,3,128,128); original=x.clone()
        experts=[Spy(.1),Spy(.2),Spy(.3)]
        router=HardRouter(Classifier(),*experts)
        r=router(x)
        self.assertEqual(r['labels'].tolist(),[3,0,2,1])
        self.assertTrue(torch.equal(r['output'][1],x[1]))
        self.assertTrue(torch.allclose(r['output'][0],torch.full_like(x[0],.3)))
        self.assertTrue(torch.equal(x,original))
        for m in experts:m.calls=0
        r=router(x,torch.zeros(4,dtype=torch.long))
        self.assertTrue(torch.equal(r['output'],x));self.assertEqual([m.calls for m in experts],[0,0,0])
        with self.assertRaises(ValueError):HardRouter(Classifier(),experts[0],experts[0],experts[2])
    def test_independent_weights(self):
        torch.set_num_threads(1)
        a=SpecialistAutoencoder();b=SpecialistAutoencoder()
        before={k:v.clone() for k,v in b.state_dict().items()}
        loss=a(torch.rand(1,3,128,128)).mean();loss.backward()
        torch.optim.Adam(a.parameters()).step()
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in before.items()))
    def test_manifest_filter(self):
        d=EvaluationPetsDataset(REPO_ROOT/'data/splits/pets_val.json',REPO_ROOT/'data/manifests/pets_val_corruptions.json')
        for label,name in enumerate(['salt','blur','occlusion'],1):
            selected=select_condition(Subset(d,range(20)),name)
            self.assertEqual(len(selected),6)
            self.assertTrue(all(d.manifest['records'][i]['spec']['label']==label for i in selected.indices))
