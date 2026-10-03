"""Resumable Task 3 warm-up/joint baseline, balanced batches, fixed validation."""
import argparse
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import mlflow
import torch
from torch.utils.data import Subset
import yaml
from src.models.soft_moe import from_task2_bundle
from src.losses.moe_loss import MoELoss
from src.losses.reconstruction import ReconstructionLoss
from src.data.moe_batches import balanced_restoration_batch
from src.data.pets_dataset import PetsDataset, EvaluationPetsDataset, REPO_ROOT
from exports.task2_bundle import SELECTED, digest
from training.train_task1 import save_checkpoint, restore_checkpoint, stop_requests, batch_at, preview

class OutputOnly(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model=model
    def forward(self, x): return self.model(x)['output']


def validate(model, dataset, device, batch_size, stop):
    model.eval(); groups={}; criterion=ReconstructionLoss(alpha=.8)
    with torch.no_grad():
        for start in range(0,len(dataset),batch_size):
            if stop(): return None
            batch=batch_at(dataset,range(start,min(start+batch_size,len(dataset))))
            result=model(batch['input'].to(device)); metrics=criterion(result['output'],batch['target'].to(device))
            for i,label in enumerate(batch['label'].tolist()):
                key=f'{label}/{batch["severity"][i]}'
                row=groups.setdefault(key,{'count':0,'loss':0.,'l1':0.,'ssim':0.,'weights':[0.]*4})
                row['count']+=1
                for k in ('loss','l1','ssim'): row[k]+=metrics[k][i].item()
                for j,w in enumerate(result['weights'][i].tolist()): row['weights'][j]+=w
    for row in groups.values():
        for k in ('loss','l1','ssim'): row[k]/=row['count']
        row['weights']=[w/row['count'] for w in row['weights']]
    balanced={}
    for k in ('loss','l1','ssim'):
        conditions=[]
        for label in range(4):
            rows=[r[k] for key,r in groups.items() if key.startswith(f'{label}/')]
            if not rows: raise ValueError('Validation must cover all four conditions')
            conditions.append(sum(rows)/len(rows))
        balanced[k]=sum(conditions)/4
    return {'balanced':balanced,'by_condition_severity':groups}


def run(a):
    started=time.monotonic(); cfg=yaml.safe_load(a.config.read_text()); t=cfg['training']
    total=t['warmup_epochs']+t['joint_epochs']; batch=t['batch_size']
    if batch<4 or batch%4 or min(t['warmup_epochs'],t['joint_epochs'])<1 or not 0<t['joint_learning_rate']<t['warmup_learning_rate']:
        raise ValueError('Require balanced batch multiple of four, positive stages and smaller joint LR')
    if a.max_hours<=0 or min(a.train_limit,a.val_images,a.stop_after_epoch,a.max_steps)<0: raise ValueError('Invalid limits')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device(a.device)
    train_path=REPO_ROOT/'data/splits/pets_train.json';val_path=REPO_ROOT/'data/splits/pets_val.json';manifest=REPO_ROOT/'data/manifests/pets_val_corruptions.json'
    train=PetsDataset(train_path);val=EvaluationPetsDataset(val_path,manifest)
    if a.train_limit: train=Subset(train,range(min(a.train_limit,len(train))))
    if a.val_images: val=Subset(val,range(min(a.val_images*10,len(val))))
    source=['src/models/soft_moe.py','src/models/corruption_classifier.py','src/models/specialist_ae.py','src/models/spatial16_ae.py','src/data/moe_batches.py','src/data/pets_dataset.py','src/data/corruptions.py','src/losses/moe_loss.py','src/losses/reconstruction.py','training/train_task3_moe.py','training/train_task1.py']
    signature={'files':{p.name:digest(p) for p in (train_path,val_path,manifest)},'source':{s:digest(REPO_ROOT/s) for s in source},'initialization':{k:v[2] for k,v in SELECTED.items()},'train_limit':a.train_limit,'val_images':a.val_images}
    model=from_task2_bundle(a.bundle,**cfg['model']).to(device);criterion=MoELoss(**cfg['loss']).to(device)
    # All parameters registered once: frozen experts receive no gradients or Adam state until joint stage.
    opt=torch.optim.Adam(model.parameters(),lr=t['warmup_learning_rate'])
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.resume and any(out.iterdir()):raise ValueError('Use a new output folder or --resume')
    progress={'epoch':0,'cursor':0,'order':None,'train_sum':0.,'train_count':0,'best':math.inf,'history':[]}
    if a.resume:progress=restore_checkpoint(a.resume,model,opt,cfg,signature,device)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg));(out/'data_signature.json').write_text(json.dumps(signature,indent=2))
    (out/'environment.json').write_text(json.dumps({'torch':str(torch.__version__),'device':str(device),'debug_subset':bool(a.train_limit or a.val_images)},indent=2))
    steps=0
    mlflow.set_tracking_uri((out/'mlruns').as_uri());mlflow.set_experiment('task3-soft-moe')
    with mlflow.start_run(),stop_requests() as stopped:
        mlflow.log_params({'batch_size':batch,'temperature':cfg['model']['temperature'],'debug_subset':bool(a.train_limit or a.val_images),**t,**cfg['loss']})
        def stop():return stopped['requested'] or time.monotonic()-started>=a.max_hours*3600 or bool(a.max_steps and steps>=a.max_steps)
        def save():save_checkpoint(out/'last.pt',model,opt,progress,cfg,signature,device)
        while progress['epoch']<min(total,a.stop_after_epoch or total) and not stop():
            stage='warmup' if progress['epoch']<t['warmup_epochs'] else 'joint'
            model.set_stage(stage);model.train()
            for group in opt.param_groups:group['lr']=t[f'{stage}_learning_rate']
            if progress['order'] is None:progress['order']=torch.randperm(len(train)).tolist()
            while progress['cursor']<len(train) and not stop():
                indices=progress['order'][progress['cursor']:progress['cursor']+batch//4]
                x,y,labels=balanced_restoration_batch(train,indices)
                opt.zero_grad(set_to_none=True);metrics=criterion(model(x.to(device)),y.to(device),labels.to(device))
                if not torch.isfinite(metrics['loss']):raise RuntimeError('Nonfinite loss')
                metrics['loss'].backward();opt.step();steps+=1
                progress['cursor']+=len(indices);progress['train_sum']+=metrics['loss'].item()*len(x);progress['train_count']+=len(x)
                if steps%50==0:save();print(f'{stage}: {progress["cursor"]}/{len(train)} source images',flush=True)
            if stop():break
            result=validate(model,val,device,batch,stop)
            if result is None:break
            epoch=progress['epoch']+1
            row={'epoch':epoch,'stage':stage,'train_loss':progress['train_sum']/progress['train_count'],**{f'val_{k}':v for k,v in result['balanced'].items()},'validation':result}
            if not all(math.isfinite(row[k]) for k in ['train_loss','val_loss','val_l1','val_ssim']):raise RuntimeError('Nonfinite metrics')
            improved=row['val_loss']<progress['best']
            if improved:progress['best']=row['val_loss']
            progress['history'].append(row);progress.update(epoch=epoch,cursor=0,order=None,train_sum=0.,train_count=0)
            save()
            if improved:save_checkpoint(out/'best.pt',model,opt,progress,cfg,signature,device)
            (out/'history.json').write_text(json.dumps(progress['history'],indent=2))
            mlflow.log_metrics({k:row[k] for k in ['train_loss','val_loss','val_l1','val_ssim']},step=epoch)
            for key,r in result['by_condition_severity'].items():
                mlflow.log_metrics({f'weights/{key}/{i}':w for i,w in enumerate(r['weights'])},step=epoch)
            preview(OutputOnly(model),val,device,out/'preview.png')
            print(f'Epoch {epoch}/{total} ({stage}): val={row["val_loss"]:.5f}, SSIM={row["val_ssim"]:.4f}',flush=True)
        save();(out/'history.json').write_text(json.dumps(progress['history'],indent=2))
        mlflow.log_artifact(str(out/'history.json'))
    print(f'Saved {out}; epochs {progress["epoch"]}; cursor {progress["cursor"]}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=REPO_ROOT/'configs/task3.yaml')
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--max-hours',type=float,default=2);p.add_argument('--stop-after-epoch',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=0,help='Per invocation pause at safe batch boundary')
    p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-images',type=int,default=0)
    run(p.parse_args())
if __name__=='__main__':main()
