"""Condition-specific validation with equal severity weighting."""
import torch
from torch.utils.data import Subset
from src.data.corruptions import CLASSES
from training.train_task1 import batch_at

def select_condition(dataset, condition):
    label=CLASSES.index(condition)
    if isinstance(dataset,Subset):
        base=dataset.dataset; indices=list(dataset.indices)
    else:base=dataset;indices=range(len(base))
    selected=[i for i in indices if base.manifest['records'][i]['spec']['label']==label]
    if not selected:raise ValueError('No validation cases for selected condition')
    return Subset(base,selected)

def validation(model,dataset,criterion,device,batch_size,should_stop):
    model.eval(); groups={}
    with torch.no_grad():
        for start in range(0,len(dataset),batch_size):
            if should_stop():return None
            b=batch_at(dataset,range(start,min(start+batch_size,len(dataset))))
            target=b['target'].to(device);x=b['input'].to(device)
            output=criterion(model(x),target);inputs=criterion(x,target)
            for i,severity in enumerate(b['severity']):
                key=f'{b["label"][i].item()}/{severity}'
                row=groups.setdefault(key,{'count':0,**{k:0. for k in ['loss','l1','ssim','input_l1','input_ssim']}})
                row['count']+=1
                for k in ['loss','l1','ssim']:row[k]+=output[k][i].item()
                for k in ['l1','ssim']:row['input_'+k]+=inputs[k][i].item()
    means={k:{m:v/r['count'] for m,v in r.items() if m!='count'} for k,r in groups.items()}
    if len(means)!=3:raise ValueError('Expected all three fixed severities')
    return {'balanced':{m:sum(r[m] for r in means.values())/3 for m in ['loss','l1','ssim']},'by_condition_severity':means}
