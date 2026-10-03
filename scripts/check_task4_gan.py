"""Real training-pair diagnostic, not a trained model or validation result."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from src.data.fs2k_dataset import FS2KDataset
from src.models.generator_unet import StyleUNet
from src.models.discriminator_patchgan import StylePatchGAN
from src.losses.gan_loss import discriminator_loss,generator_loss

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--raw-dir',type=Path,default=Path('data/raw/fs2k/FS2K'))
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/task4-diagnostic'))
    a=p.parse_args()
    if a.steps<1:p.error('Steps must be positive')
    if a.output_dir.exists():raise ValueError('Choose new output directory')
    torch.set_num_threads(2);torch.manual_seed(42)
    dataset=FS2KDataset('data/splits/fs2k/train.json',a.raw_dir)
    samples=[dataset[next(i for i,r in enumerate(dataset.rows) if r['style']==s)] for s in range(3)]
    x=torch.stack([s['photo'] for s in samples]).to(a.device);y=torch.stack([s['sketch'] for s in samples]).to(a.device);styles=torch.arange(3,device=a.device)
    g=StyleUNet().to(a.device);d=StylePatchGAN().to(a.device)
    go=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999));do=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(.5,.999))
    initial_g=g.style_embedding.weight.detach().clone();initial_d=d.style_embedding.weight.detach().clone();history=[]
    for step in range(1,a.steps+1):
        g.train();d.train();d.requires_grad_(True);go.zero_grad(set_to_none=True);do.zero_grad(set_to_none=True)
        fake=g(x,styles)
        dl=discriminator_loss(d(x,y,styles),d(x,fake.detach(),styles));dl['loss'].backward()
        assert all(p.grad is None for p in g.parameters())
        assert torch.isfinite(dl['loss']) and all(p.grad is None or torch.isfinite(p.grad).all() for p in d.parameters())
        do.step();do.zero_grad(set_to_none=True);d.requires_grad_(False)
        gl=generator_loss(d(x,fake,styles),fake,y);gl['loss'].backward()
        assert all(p.grad is None for p in d.parameters())
        assert torch.isfinite(gl['loss']) and all(p.grad is None or torch.isfinite(p.grad).all() for p in g.parameters())
        go.step()
        row={'step':step,**{'d_'+k:float(v.detach()) for k,v in dl.items()},**{'g_'+k:float(v.detach()) for k,v in gl.items()}}
        history.append(row)
    assert not torch.equal(initial_g,g.style_embedding.weight) and not torch.equal(initial_d,d.style_embedding.weight)
    g.eval()
    with torch.no_grad():result=g(x,styles).cpu()
    canvas=Image.new('RGB',(396,3*150),'white');draw=ImageDraw.Draw(canvas)
    for i,s in enumerate(samples):
        draw.text((0,i*150),f'Style {i+1}: photo / target / diagnostic output',fill='black')
        for j,tensor in enumerate([s['photo'],s['sketch'],result[i]]):
            pixels=(tensor.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8');canvas.paste(Image.fromarray(pixels),(j*132,i*150+18))
    a.output_dir.mkdir(parents=True);canvas.save(a.output_dir/'preview.png')
    report={'debug_only':True,'training_pairs':[s['image_id'] for s in samples],'steps':a.steps,'both_style_embeddings_updated':True,'history':history}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'debug_only':True,'steps':a.steps,'first':history[0],'last':history[-1],'both_style_embeddings_updated':True},indent=2))
if __name__=='__main__':main()
