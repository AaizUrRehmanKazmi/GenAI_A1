"""Resumable fixed-12 training-pair fitting test; never validation evidence."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from PIL import Image,ImageDraw
from src.models.task4_candidate import CandidateGenerator
from src.data.fs2k_dataset import FS2KDataset
from src.losses.reconstruction import ReconstructionLoss
from training.train_task1 import atomic_save
from exports.task2_bundle import digest


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--max-hours',type=float,default=1)
 a=p.parse_args()
 if a.steps<1 or a.max_hours<=0:p.error('Positive step/time limits required')
 torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 ds=FS2KDataset('data/splits/fs2k/train.json',a.raw_dir,augment=False)
 ids=[i for s in range(3) for i in [j for j,r in enumerate(ds.rows) if r['style']==s][:4]]
 samples=[ds[i] for i in ids];x=torch.stack([r['photo'] for r in samples]).to(a.device);y=torch.stack([r['sketch'] for r in samples]).to(a.device);styles=torch.tensor([r['style'] for r in samples],device=a.device)
 signature={'device':a.device,'ids':[r['image_id'] for r in samples], 'sources':{f:digest(Path(f)) for f in ['scripts/overfit_task4_candidate.py','src/models/task4_candidate.py','src/data/fs2k_dataset.py','src/losses/reconstruction.py','training/train_task1.py']},'images':{ds.rows[i][k]:digest(a.raw_dir/ds.rows[i][k]) for i in ids for k in ('photo','sketch')}}
 g=CandidateGenerator(base_channels=32,style_dim=8,dropout=0).to(a.device);opt=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999));history=[];step=0
 out=a.output_dir;out.mkdir(parents=True,exist_ok=True);checkpoint=out/'last.pt'
 if checkpoint.exists():
  ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
  if ck['signature']!=signature:raise ValueError('Source/data/device mismatch')
  g.load_state_dict(ck['generator']);opt.load_state_dict(ck['optimizer']);step=ck['step'];history=ck['history'];torch.set_rng_state(ck['rng'])
  if a.device=='cuda':
   if len(ck['cuda_rng'])!=torch.cuda.device_count():raise ValueError('Visible GPU count changed')
   torch.cuda.set_rng_state_all(ck['cuda_rng'])
 elif any(out.iterdir()):raise ValueError('Nonempty output without checkpoint')
 def save():
  atomic_save({'debug_only':True,'signature':signature,'generator':g.state_dict(),'optimizer':opt.state_dict(),'step':step,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if a.device=='cuda' else []},checkpoint)
  (out/'history.json').write_text(json.dumps(history,indent=2))
  (out/'report.json').write_text(json.dumps({'debug_only':True,'image_ids':signature['ids'],'completed_steps':step,'history':history,'note':'Fixed 12 training pairs, dropout0, no augmentation, coefficient100 L1. Not validation evidence.'},indent=2))
 started=time.monotonic()
 while step<a.steps and time.monotonic()-started<a.max_hours*3600:
  g.train();opt.zero_grad(set_to_none=True);prediction=g(x,styles);loss=100*(prediction-y).abs().mean()
  if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
  loss.backward();opt.step();step+=1
  if step==1 or step%50==0 or step==a.steps:
   g.eval()
   with torch.inference_mode():
    prediction=g(x,styles);m=ReconstructionLoss()(prediction,y)
   row={'step':step,'l1':m['l1'].mean().item(),'ssim':m['ssim'].mean().item()};history.append(row);print(row,flush=True)
   canvas=Image.new('RGB',(396,150*12),'white');draw=ImageDraw.Draw(canvas)
   for i,r in enumerate(samples):
    draw.text((0,i*150),r['image_id']+' | photo / target / output',fill='black')
    for j,t in enumerate([r['photo'],r['sketch'],prediction[i].cpu()]):canvas.paste(Image.fromarray((t.permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8')),(132*j,150*i+18))
   canvas.save(out/'preview.png')
   if step%250==0 or step==1:canvas.save(out/f'step_{step:06d}.png')
   save()
 save();print('Saved step',step,flush=True)
if __name__=='__main__':main()
