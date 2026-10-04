"""Compare discriminator modes without modifying saved checkpoint or base model."""
import argparse,copy,json
from pathlib import Path
import torch
from torch.nn import functional as F
from src.models.task4_candidate import CandidateGenerator,CandidateDiscriminator
from src.data.fs2k_dataset import FS2KDataset

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 g=CandidateGenerator(**ck['config']['generator']);g.load_state_dict(ck['generator']);g.eval()
 d=CandidateDiscriminator(**ck['config']['discriminator']);d.load_state_dict(ck['discriminator'])
 report={'epoch':ck['progress']['epoch'],'debug_only':True,'batches':[]}
 for split in ('train','val'):
  ds=FS2KDataset(f'data/splits/fs2k/{split}.json','data/raw/fs2k/FS2K')
  for batch in range(3):
   samples=[ds[i] for i in range(batch*8,(batch+1)*8)]
   x=torch.stack([r['photo'] for r in samples]);y=torch.stack([r['sketch'] for r in samples]);s=torch.tensor([r['style'] for r in samples])
   with torch.no_grad():fake=g(x,s)
   row={'split':split,'batch':batch,'ids':[r['image_id'] for r in samples]}
   for mode in ('train','eval'):
    model=copy.deepcopy(d);model.train(mode=='train');model.requires_grad_(False)
    with torch.no_grad():
     real=model(x,y,s);out=model(x,fake,s)
     row[mode]={'real_probability':real.sigmoid().mean().item(),'fake_probability':out.sigmoid().mean().item(),'d_loss':(.5*(F.binary_cross_entropy_with_logits(real,torch.ones_like(real))+F.binary_cross_entropy_with_logits(out,torch.zeros_like(out)))).item(),'g_adversarial':F.binary_cross_entropy_with_logits(out,torch.ones_like(out)).item()}
    probe=fake.detach().clone().requires_grad_(True)
    loss=F.binary_cross_entropy_with_logits(model(x,probe,s),torch.ones_like(out))
    grad=torch.autograd.grad(loss,probe)[0]
    row[mode]['gradient_rms']=grad.square().mean().sqrt().item()
    row[mode]['gradient_finite']=bool(torch.isfinite(grad).all())
   report['batches'].append(row);print(split,batch,row['train'],row['eval'],flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
