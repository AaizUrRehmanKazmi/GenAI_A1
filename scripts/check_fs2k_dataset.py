"""Decode all training/validation pairs; preview training samples only."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from src.data.fs2k_dataset import FS2KDataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=Path('data/raw/fs2k/FS2K'))
    p.add_argument('--splits-dir',type=Path,default=Path('data/splits/fs2k'))
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/fs2k-data-checks'))
    a=p.parse_args();report={};samples={i:[] for i in range(3)}
    for split in ['train','val']:
        dataset=FS2KDataset(a.splits_dir/f'{split}.json',a.raw_dir)
        for i in range(len(dataset)):
            sample=dataset[i]
            for key in ['photo','sketch']:
                t=sample[key]
                assert t.shape==(3,128,128) and t.dtype==torch.float32 and torch.isfinite(t).all() and 0<=t.min()<=t.max()<=1
            if split=='train' and len(samples[sample['style']])<3:samples[sample['style']].append(sample)
        report[split]={'passed':len(dataset),'total':len(dataset)}
        print(split,report[split],flush=True)
    canvas=Image.new('RGB',(6*132,3*154),'white');draw=ImageDraw.Draw(canvas)
    for style,group in samples.items():
        for i,sample in enumerate(group):
            x=i*264;y=style*154
            draw.text((x,y),f'Style {style+1} {sample["image_id"]}',fill='black')
            for j,key in enumerate(['photo','sketch']):
                pixels=(sample[key].permute(1,2,0).numpy()*255).round().astype('uint8')
                canvas.paste(Image.fromarray(pixels),(x+j*132,y+20))
    a.output_dir.mkdir(parents=True,exist_ok=True);canvas.save(a.output_dir/'pairs.png')
    report['test']='not decoded';(a.output_dir/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
