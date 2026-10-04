"""Matched fixed-training-set objective ablation; not validation evidence."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PIL import Image, ImageDraw
from src.data.fs2k_dataset import FS2KDataset
from src.models.generator_unet import StyleUNet
from src.models.discriminator_patchgan import StylePatchGAN
from src.losses.gan_loss import discriminator_loss, generator_loss
from src.losses.reconstruction import ReconstructionLoss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=Path('data/raw/fs2k/FS2K'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    if a.steps<1:p.error('steps must be positive')
    if a.output_dir.exists():raise ValueError('Use a new output directory')
    a.output_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    ds=FS2KDataset('data/splits/fs2k/train.json',a.raw_dir,augment=False)
    ids=[i for s in range(3) for i in [j for j,r in enumerate(ds.rows) if r['style']==s][:4]]
    samples=[ds[i] for i in ids]
    x=torch.stack([s['photo'] for s in samples]).to(a.device)
    y=torch.stack([s['sketch'] for s in samples]).to(a.device)
    styles=torch.tensor([s['style'] for s in samples],device=a.device)
    report={'debug_only':True,'steps':a.steps,'seed':42,'dropout':0.,
            'note':'12 fixed training pairs; no augmentation; same initialization and batches; L1 coefficient 100 in both arms. Not generalization evidence.',
            'image_ids':[s['image_id'] for s in samples],'arms':{}}
    outputs=[]
    for mode in ('l1_only','gan_l1'):
        torch.manual_seed(42)
        g=StyleUNet(dropout=0.).to(a.device);d=StylePatchGAN().to(a.device)
        go=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
        do=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(.5,.999))
        history=[]
        for step in range(1,a.steps+1):
            g.train();go.zero_grad(set_to_none=True)
            fake=g(x,styles)
            if mode=='gan_l1':
                d.requires_grad_(True);do.zero_grad(set_to_none=True)
                dl=discriminator_loss(d(x,y,styles),d(x,fake.detach(),styles))['loss']
                if not torch.isfinite(dl):raise ValueError('Nonfinite discriminator loss')
                dl.backward();do.step();d.requires_grad_(False)
                loss=generator_loss(d(x,fake,styles),fake,y)['loss']
            else:loss=100*(fake-y).abs().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite generator loss')
            loss.backward();go.step()
            if step==1 or step%50==0 or step==a.steps:
                g.eval()
                with torch.inference_mode():
                    out=g(x,styles);m=ReconstructionLoss()(out,y)
                row={'step':step,'l1':m['l1'].mean().item(),'ssim':m['ssim'].mean().item()}
                history.append(row);print(mode,row,flush=True)
                (a.output_dir/f'{mode}_history.json').write_text(json.dumps(history,indent=2))
        outputs.append(out.cpu());report['arms'][mode]=history
        torch.save({'generator':g.state_dict(),'debug_only':True,'steps':a.steps},a.output_dir/f'{mode}.pt')
    canvas=Image.new('RGB',(528,len(samples)*150),'white');draw=ImageDraw.Draw(canvas)
    for i,s in enumerate(samples):
        draw.text((0,i*150),f'{s["image_id"]} | photo / target / L1 only / GAN+L1',fill='black')
        for j,t in enumerate([s['photo'],s['sketch'],outputs[0][i],outputs[1][i]]):
            canvas.paste(Image.fromarray((t.permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8')),(132*j,i*150+18))
    canvas.save(a.output_dir/'comparison.png')
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
