"""Resumable paired FS2K conditional GAN candidate; official test untouched."""
import argparse
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import mlflow
import torch
import yaml
from torch.utils.data import Subset
from src.data.fs2k_dataset import FS2KDataset
from src.models.task4_candidate import CandidateGenerator as StyleUNet
from src.models.task4_candidate import CandidateDiscriminator as StylePatchGAN
from training.task4_bn_policy import generator_discriminator_phase
from src.losses.gan_loss import discriminator_loss,generator_loss
from src.losses.reconstruction import ReconstructionLoss
from training.train_task1 import atomic_save,stop_requests,batch_at
from exports.task2_bundle import digest
ROOT=Path(__file__).resolve().parents[1]

def validation(g,data,device,batch_size,stop):
    g.eval();groups={};criterion=ReconstructionLoss()
    with torch.no_grad():
        for start in range(0,len(data),batch_size):
            if stop():return None
            b=batch_at(data,range(start,min(start+batch_size,len(data))));x=b['photo'].to(device);y=b['sketch'].to(device)
            metrics=criterion(g(x,b['style'].to(device)),y)
            for i,s in enumerate(b['style'].tolist()):
                r=groups.setdefault(str(s),{'count':0,'l1':0.,'ssim':0.});r['count']+=1
                for key in ['l1','ssim']:r[key]+=metrics[key][i].item()
    for r in groups.values():
        for key in ['l1','ssim']:r[key]/=r['count']
    if len(groups)!=3:raise ValueError('Validation must contain all styles')
    return {'by_style':groups,'l1':sum(r['l1'] for r in groups.values())/3,'ssim':sum(r['ssim'] for r in groups.values())/3}

def preview(g,data,device,path):
    from PIL import Image,ImageDraw
    g.eval();canvas=Image.new('RGB',(396,450),'white');draw=ImageDraw.Draw(canvas)
    for style in range(3):
        sample=next(data[i] for i in range(len(data)) if data[i]['style']==style)
        with torch.no_grad():out=g(sample['photo'][None].to(device),torch.tensor([style],device=device))[0].cpu()
        draw.text((0,style*150),f'Style {style+1}: photo / target / output',fill='black')
        for j,t in enumerate([sample['photo'],sample['sketch'],out]):
            canvas.paste(Image.fromarray((t.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')),(132*j,style*150+18))
    canvas.save(path)

def run(a):
    started=time.monotonic();cfg=yaml.safe_load(a.config.read_text());t=cfg['training']
    if cfg.get('architecture')!='deep_unet_v1' or cfg.get('discriminator_policy')!='batch_stats_preserve_buffers' or cfg['loss'].get('adversarial_weight',-1)<0:raise ValueError('Use candidate configuration')
    if min(t['epochs'],t['batch_size'],a.max_hours)<=0 or min(a.train_limit,a.val_per_style,a.max_steps,a.stop_after_epoch)<0:raise ValueError('Invalid limits')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=torch.device(a.device)
    train=FS2KDataset(a.splits_dir/'train.json',a.raw_dir,augment=True);val=FS2KDataset(a.splits_dir/'val.json',a.raw_dir)
    if a.train_limit:train=Subset(train,range(min(a.train_limit,len(train))))
    if a.val_per_style:
        indices=[i for s in range(3) for i in [j for j,r in enumerate(val.rows) if r['style']==s][:a.val_per_style]];val=Subset(val,indices)
    sources=['training/train_task4_finetune.py','src/data/fs2k_dataset.py','src/models/task4_candidate.py','training/task4_bn_policy.py','src/losses/gan_loss.py','src/losses/reconstruction.py','training/train_task1.py']
    signature={'source':{s:digest(ROOT/s) for s in sources},'splits':{n:digest(a.splits_dir/n) for n in ['train.json','val.json','metadata.json']},'train_limit':a.train_limit,'val_per_style':a.val_per_style}
    # Hash train/validation image bytes as well; do not touch test pixels.
    rows=json.loads((a.splits_dir/'train.json').read_text())+json.loads((a.splits_dir/'val.json').read_text())
    signature['images']={r[k]:digest(a.raw_dir/r[k]) for r in rows for k in ['photo','sketch']}
    signature['initialization_sha256']=digest(a.init_checkpoint)
    g=StyleUNet(**cfg['generator']).to(device);d=StylePatchGAN(**cfg['discriminator']).to(device)
    go=torch.optim.Adam(g.parameters(),lr=t['generator_lr'],betas=(.5,.999));do=torch.optim.Adam(d.parameters(),lr=t['discriminator_lr'],betas=(.5,.999))
    if not a.resume:
        initial=torch.load(a.init_checkpoint,map_location='cpu',weights_only=True)
        if initial['config'].get('objective')!='reconstruction_only' or initial['config']['generator']!=cfg['generator']:
            raise ValueError('Initialize from matching reconstruction-only generator')
        if initial['signature']['splits']!=signature['splits'] or initial['signature']['images']!=signature['images']:
            raise ValueError('Initialization dataset differs')
        g.load_state_dict(initial['generator'])
        # Both comparison arms reset Adam; this is a new experiment, not exact continuation.
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if not a.resume and any(out.iterdir()):raise ValueError('Use new output folder or --resume')
    progress={'epoch':0,'cursor':0,'order':None,'sums':{},'count':0,'best':math.inf,'history':[]}
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=True)
        if saved['config']!=cfg or saved['signature']!=signature:raise ValueError('Resume config/data/source mismatch')
        if saved['device']!=a.device:raise ValueError('Resume requires same device type')
        g.load_state_dict(saved['generator']);d.load_state_dict(saved['discriminator']);go.load_state_dict(saved['generator_optimizer']);do.load_state_dict(saved['discriminator_optimizer']);progress=saved['progress']
        torch.set_rng_state(saved['rng'])
        if device.type=='cuda':
            if len(saved['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA count changed')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def save(name='last.pt'):
        atomic_save({'version':1,'config':cfg,'signature':signature,'generator':g.state_dict(),'discriminator':d.state_dict(),'generator_optimizer':go.state_dict(),'discriminator_optimizer':do.state_dict(),'progress':progress,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'device':a.device},out/name)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg));(out/'data_signature.json').write_text(json.dumps(signature,indent=2))
    (out/'environment.json').write_text(json.dumps({'torch':str(torch.__version__),'device':a.device,'debug_subset':bool(a.train_limit or a.val_per_style)}))
    steps=0;mlflow.set_tracking_uri((out/'mlruns').as_uri());mlflow.set_experiment('task4-finetune-comparison')
    with mlflow.start_run(),stop_requests() as stopped:
        mlflow.log_params({**t,'l1_weight':cfg['loss']['l1_weight'],'debug_subset':bool(a.train_limit or a.val_per_style)})
        def stop():return stopped['requested'] or time.monotonic()-started>=a.max_hours*3600 or bool(a.max_steps and steps>=a.max_steps)
        while progress['epoch']<min(t['epochs'],a.stop_after_epoch or t['epochs']) and not stop():
            g.train();d.train()
            if progress['order'] is None:progress['order']=torch.randperm(len(train)).tolist()
            while progress['cursor']<len(train) and not stop():
                ids=progress['order'][progress['cursor']:progress['cursor']+t['batch_size']];b=batch_at(train,ids);x=b['photo'].to(device);y=b['sketch'].to(device);s=b['style'].to(device)
                go.zero_grad(set_to_none=True);fake=g(x,s)
                weight=cfg['loss']['adversarial_weight']
                if weight:
                    d.train();d.requires_grad_(True);do.zero_grad(set_to_none=True)
                    dl=discriminator_loss(d(x,y,s),d(x,fake.detach(),s))
                    if not torch.isfinite(dl['loss']):raise RuntimeError('Nonfinite D loss')
                    dl['loss'].backward();do.step();do.zero_grad(set_to_none=True)
                    with generator_discriminator_phase(d):
                        gl=generator_loss(d(x,fake,s),fake,y,cfg['loss']['l1_weight'])
                        loss=weight*gl['adversarial']+cfg['loss']['l1_weight']*gl['reconstruction']
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite G loss')
                        loss.backward()
                    metrics={'d_loss':float(dl['loss'].detach()),'d_real':float(dl['real'].detach()),'d_fake':float(dl['fake'].detach()),'g_adversarial':float(gl['adversarial'].detach()),'g_reconstruction':float(gl['reconstruction'].detach())}
                else:
                    reconstruction=(fake-y).abs().mean();loss=cfg['loss']['l1_weight']*reconstruction
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite G loss')
                    loss.backward();metrics={'g_reconstruction':float(reconstruction.detach())}
                go.step();metrics['g_loss']=float(loss.detach())
                for k,v in metrics.items():progress['sums'][k]=progress['sums'].get(k,0)+v*len(ids)
                progress['count']+=len(ids);progress['cursor']+=len(ids);steps+=1
                if steps%50==0:save();print('Training pairs',progress['cursor'],'/',len(train),flush=True)
            if stop():break
            result=validation(g,val,device,t['batch_size'],stop)
            if result is None:break
            epoch=progress['epoch']+1;row={'epoch':epoch,**{k:v/progress['count'] for k,v in progress['sums'].items()},'val_l1':result['l1'],'val_ssim':result['ssim'],'validation':result}
            if not all(math.isfinite(v) for k,v in row.items() if k!='validation'):raise RuntimeError('Nonfinite metrics')
            improved=row['val_l1']<progress['best']
            if improved:progress['best']=row['val_l1']
            progress['history'].append(row);progress.update(epoch=epoch,cursor=0,order=None,sums={},count=0);save()
            if improved:save('best.pt')
            (out/'history.json').write_text(json.dumps(progress['history'],indent=2));mlflow.log_metrics({k:v for k,v in row.items() if k not in ['validation','epoch']},step=epoch)
            preview(g,val,device,out/'preview.png');mlflow.log_artifact(str(out/'preview.png'),artifact_path=f'epoch_{epoch}')
            print(f'Epoch {epoch}: val L1={row["val_l1"]:.5f}, SSIM={row["val_ssim"]:.4f}',flush=True)
        save();(out/'history.json').write_text(json.dumps(progress['history'],indent=2))
    print('Saved',out,'epoch',progress['epoch'],'cursor',progress['cursor'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/task4_finetune_gan.yaml');p.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw/fs2k/FS2K');p.add_argument('--splits-dir',type=Path,default=ROOT/'data/splits/fs2k');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--init-checkpoint',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--max-hours',type=float,default=2)
    for flag in ['train-limit','val-per-style','max-steps','stop-after-epoch']:p.add_argument('--'+flag,type=int,default=0)
    run(p.parse_args())
if __name__=='__main__':main()
