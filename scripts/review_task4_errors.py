"""Validation-only representative/error review; no automatic alignment claim."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from src.models.task4_candidate import CandidateGenerator
from src.data.fs2k_dataset import FS2KDataset
from src.losses.reconstruction import ReconstructionLoss

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 g=CandidateGenerator(**ck['config']['generator']);g.load_state_dict(ck['generator']);g.eval()
 ds=FS2KDataset('data/splits/fs2k/val.json','data/raw/fs2k/FS2K');criterion=ReconstructionLoss();rows=[];outputs=[]
 with torch.inference_mode():
  for start in range(0,len(ds),8):
   samples=[ds[i] for i in range(start,min(start+8,len(ds)))];x=torch.stack([r['photo'] for r in samples]);y=torch.stack([r['sketch'] for r in samples]);s=torch.tensor([r['style'] for r in samples]);out=g(x,s)
   m=criterion(out,y);white=criterion(torch.ones_like(y),y)
   for i,r in enumerate(samples):
    rows.append({'index':start+i,'image_id':r['image_id'],'style':r['style'],'l1':m['l1'][i].item(),'ssim':m['ssim'][i].item(),'white_l1':white['l1'][i].item(),'white_ssim':white['ssim'][i].item()});outputs.append(out[i])
 a.output_dir.mkdir(parents=True,exist_ok=True)
 def grid(indices,name):
  canvas=Image.new('RGB',(660,150*len(indices)),'white');draw=ImageDraw.Draw(canvas)
  for n,i in enumerate(indices):
   r=ds[i];out=outputs[i];err=(out-r['sketch']).abs()
   draw.text((0,n*150),f"{r['image_id']} style {r['style']+1} SSIM {rows[i]['ssim']:.3f} | photo / target / output / error / overlay",fill='black')
   for j,t in enumerate([r['photo'],r['sketch'],out,err,.5*r['photo']+.5*r['sketch']]):
    canvas.paste(Image.fromarray((t.permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8')),(132*j,n*150+20))
  canvas.save(a.output_dir/name)
 representative=[];worst=[]
 for s in range(3):
  group=[r for r in rows if r['style']==s]
  representative += [group[round(j*(len(group)-1)/3)]['index'] for j in range(4)]
  worst += [r['index'] for r in sorted(group,key=lambda r:r['ssim'])[:3]]
 grid(representative,'representative.png');grid(worst,'worst.png')
 summary={str(s):{k:sum(r[k] for r in rows if r['style']==s)/sum(r['style']==s for r in rows) for k in ('l1','ssim','white_l1','white_ssim')} for s in range(3)}
 (a.output_dir/'report.json').write_text(json.dumps({'epoch':ck['progress']['epoch'],'by_style':summary,'rows':rows,'official_test_evaluated':False,'note':'Representative cases equally spaced in stored style order; worst cases selected by SSIM. Overlay is visual aid, not alignment measurement.'},indent=2));print(summary)
if __name__=='__main__':main()
