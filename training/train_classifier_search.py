"""Task 2 classifier baseline, balanced batches, fixed validation and batch resume."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import mlflow
import torch
import yaml
from torch.utils.data import Subset
from src.data.pets_dataset import REPO_ROOT, PetsDataset, EvaluationPetsDataset
from src.data.classifier_batches import balanced_batch
from src.data.corruptions import CLASSES
from src.models.corruption_classifier import CorruptionClassifier
from evaluation.classifier_metrics import metrics_from_confusion
from training.train_task1 import save_checkpoint, restore_checkpoint, stop_requests, batch_at

def evaluate(model,data,device,size,stop):
    cm=torch.zeros(4,4,dtype=torch.int64); groups={}; total=0.
    model.eval()
    with torch.no_grad():
        for start in range(0,len(data),size):
            if stop(): return None
            b=batch_at(data,range(start,min(start+size,len(data))))
            y=b['label'].to(device); logits=model(b['input'].to(device)); pred=logits.argmax(1)
            total+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
            cm+=torch.bincount((y*4+pred).cpu(),minlength=16).reshape(4,4)
            for truth,guess,severity in zip(y.tolist(),pred.tolist(),b['severity']):
                key=f'{CLASSES[truth]}/{severity}'
                r=groups.setdefault(key,{'correct':0,'count':0});r['count']+=1;r['correct']+=int(truth==guess)
    return {**metrics_from_confusion(cm),'cross_entropy':total/len(data),
            'class_order':list(CLASSES),'by_condition_severity':groups}

def run(a):
    started=time.monotonic(); cfg=yaml.safe_load(a.config.read_text()); t=cfg['training']
    if t['batch_size']<4 or t['batch_size']%4: raise ValueError('Batch size must be a positive multiple of four')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    device=torch.device(a.device)
    train=PetsDataset(raw_dir=a.raw_dir)
    val=EvaluationPetsDataset(REPO_ROOT/'data/splits/pets_val.json',REPO_ROOT/'data/manifests/pets_val_corruptions.json',a.raw_dir)
    if a.train_limit:train=Subset(train,range(min(len(train),a.train_limit)))
    if a.val_images:val=Subset(val,range(min(len(val),a.val_images*10)))
    paths=['training/train_classifier.py','training/train_classifier_search.py','src/models/corruption_classifier.py','src/data/classifier_batches.py',
        'src/data/pets_dataset.py','src/data/corruptions.py','evaluation/classifier_metrics.py','training/train_task1.py',
        'data/splits/pets_train.json','data/splits/pets_val.json','data/manifests/pets_val_corruptions.json']
    signature={'hashes':{p:hashlib.sha256((REPO_ROOT/p).read_bytes()).hexdigest() for p in paths},
        'train_limit':a.train_limit,'val_images':a.val_images}
    model=CorruptionClassifier(**cfg['model']).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.resume and any(out.iterdir()):raise ValueError('Choose an empty output folder or resume')
    progress={'epoch':0,'cursor':0,'order':None,'sum':0.,'count':0,'best':-1.,'history':[]}
    if a.resume:progress=restore_checkpoint(a.resume,model,opt,cfg,signature,device)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg));(out/'provenance.json').write_text(json.dumps(signature,indent=2))
    (out/'environment.json').write_text(json.dumps({'torch':str(torch.__version__),'device':str(device),
        'parameters':sum(p.numel() for p in model.parameters()),'debug_subset':bool(a.train_limit or a.val_images)},indent=2))
    mlflow.set_tracking_uri((out/'mlruns').as_uri());mlflow.set_experiment('task2-classifier')
    with mlflow.start_run(),stop_requests() as state:
        mlflow.log_params({**t,'channels':str(cfg['model']['channels']),'dropout':cfg['model']['dropout'],'seed':cfg['seed']})
        stop=lambda:state['requested'] or time.monotonic()-started>=a.max_hours*3600
        save=lambda name:save_checkpoint(out/name,model,opt,progress,cfg,signature,device)
        while progress['epoch']<min(t['epochs'], a.stop_after_epoch or t['epochs']) and not stop():
            if progress['order'] is None:progress['order']=torch.randperm(len(train)).tolist()
            model.train()
            while progress['cursor']<len(train) and not stop():
                ids=progress['order'][progress['cursor']:progress['cursor']+t['batch_size']//4]
                x,y=balanced_batch(train,ids);opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                loss.backward();opt.step()
                progress['cursor']+=len(ids);progress['sum']+=loss.item()*len(y);progress['count']+=len(y)
                if progress['cursor']%(t['batch_size']//4*50)==0:save('last.pt')
            if stop():break
            result=evaluate(model,val,device,t['batch_size'],stop)
            if result is None:break
            epoch=progress['epoch']+1
            row={'epoch':epoch,'train_cross_entropy':progress['sum']/progress['count'],**result}
            progress['history'].append(row);improved=result['macro_f1']>progress['best']
            if improved:progress['best']=result['macro_f1']
            progress.update(epoch=epoch,cursor=0,order=None,sum=0.,count=0)
            save('last.pt')
            if improved:save('best.pt');(out/'best_validation.json').write_text(json.dumps(row,indent=2))
            (out/'history.json').write_text(json.dumps(progress['history'],indent=2))
            mlflow.log_metrics({k:row[k] for k in ['train_cross_entropy','cross_entropy','accuracy','macro_f1','macro_precision','macro_recall']},step=epoch)
            print(f'Epoch {epoch}: accuracy={result["accuracy"]:.4f}, macro F1={result["macro_f1"]:.4f}',flush=True)
        save('last.pt');(out/'history.json').write_text(json.dumps(progress['history'],indent=2))
        for name in ['config.yaml','history.json','provenance.json','environment.json']:mlflow.log_artifact(str(out/name))
    print(f'Saved {out}; completed epochs {progress["epoch"]}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stop-after-epoch',type=int,default=0)
    p.add_argument('--config',type=Path,default=REPO_ROOT/'configs/task2_classifier.yaml')
    p.add_argument('--raw-dir',type=Path,default=REPO_ROOT/'data/raw/oxford_pets')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--max-hours',type=float,default=4)
    p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-images',type=int,default=0)
    a=p.parse_args()
    if a.stop_after_epoch<0 or a.max_hours<=0 or min(a.train_limit,a.val_images)<0:p.error('Invalid time/subset limits')
    run(a)
if __name__=='__main__':main()
