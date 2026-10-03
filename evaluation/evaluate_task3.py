"""Fixed validation comparison of Task 3 soft routing with selected Task 2 hard routing."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from src.data.pets_dataset import EvaluationPetsDataset, REPO_ROOT
from src.data.corruptions import CLASSES
from src.models.soft_moe import from_task2_bundle
from src.models.hard_router import HardRouter
from src.losses.reconstruction import ReconstructionLoss
from exports.task2_bundle import SELECTED, digest
from training.train_task1 import batch_at
from evaluation.evaluate_task1 import write_csv


def summarize(rows):
    keys=[f'{model}_{metric}' for model in ('input','task2','task3') for metric in ('l1','ssim','loss')]
    keys += [f'weight_{c}' for c in CLASSES]+['entropy','max_weight']
    def mean(group):
        return {'count':len(group),**{k:float(np.mean([r[k] for r in group])) for k in keys},
                'dominated_fraction':sum(r['max_weight']>=.9 for r in group)/len(group),
                'distributed_fraction':sum(r['max_weight']<.8 for r in group)/len(group)}
    groups=[];conditions=[]
    for c in CLASSES:
        selected=[r for r in rows if r['condition']==c]
        if not selected:raise ValueError('All conditions required')
        levels=[]
        for s in ('clean','low','medium','high'):
            subset=[r for r in selected if r['severity']==s]
            if subset:
                item={'condition':c,'severity':s,**mean(subset)};groups.append(item);levels.append(item)
        conditions.append({'condition':c,**{k:sum(r[k] for r in levels)/len(levels) for k in keys}})
    return {'by_condition_severity':groups,'by_condition':conditions,
            'condition_balanced':{k:sum(r[k] for r in conditions)/4 for k in keys},
            'routing_thresholds':{'dominated':'>=0.9 maximum weight','distributed':'<0.8 maximum weight','entropy':'natural log, maximum ln(4)'},
            'gate_argmax_accuracy':sum(r['gate_correct'] for r in rows)/len(rows)}


def select_examples(rows):
    representatives=[]
    for c in CLASSES:
        for severity in ('clean','low','medium','high'):
            subset=[r for r in rows if r['condition']==c and r['severity']==severity]
            representatives.extend(subset[:3] if c=='clean' else subset[:1])
    def unique(ordered):
        selected=[];seen=set()
        for r in ordered:
            if r['image_id'] not in seen:
                selected.append(r);seen.add(r['image_id'])
            if len(selected)==4:break
        return selected
    return {'representative':representatives,
            'dominated':unique(sorted([r for r in rows if r['max_weight']>=.9],key=lambda r:-r['max_weight'])),
            'distributed':unique(sorted([r for r in rows if r['max_weight']<.8],key=lambda r:-r['entropy'])),
            'regressions':unique(sorted([r for r in rows if r['condition']!='clean' and r['task3_ssim']<r['input_ssim']-1e-6],key=lambda r:r['task3_ssim']-r['input_ssim']))}


def panel(path, selected, dataset, model, hard, device):
    if not selected:return
    canvas=Image.new('RGB',(5*132,len(selected)*166),'white');draw=ImageDraw.Draw(canvas)
    for row,r in enumerate(selected):
        sample=dataset[r['index']];x=sample['input'][None].to(device)
        with torch.no_grad():soft=model(x)['output'][0].cpu();baseline=hard(x)['output'][0].cpu()
        tensors=[sample['target'],sample['input'],baseline,soft,(soft-sample['target']).abs()]
        weights=', '.join(f'{r[f"weight_{c}"]:.2f}' for c in CLASSES)
        draw.text((0,row*166),f'{r["image_id"]} {r["condition"]}/{r["severity"]} weights [{weights}]',fill='black')
        for col,(name,tensor) in enumerate(zip(['Target','Input','Task 2','Task 3','Abs error [0,1]'],tensors)):
            draw.text((col*132,row*166+14),name,fill='black')
            pixels=(tensor.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
            canvas.paste(Image.fromarray(pixels),(col*132,row*166+30))
    canvas.save(path)


def run(a):
    torch.set_num_threads(2);device=torch.device(a.device)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    sig=saved['data_signature']
    for source,expected in sig['source'].items():
        if digest(REPO_ROOT/source)!=expected:raise ValueError(f'Changed checkpoint source: {source}')
    if sig['initialization']!={k:v[2] for k,v in SELECTED.items()}:raise ValueError('Different Task 2 initialization')
    files=[REPO_ROOT/'data/splits/pets_train.json',REPO_ROOT/'data/splits/pets_val.json',REPO_ROOT/'data/manifests/pets_val_corruptions.json']
    for p in files:
        if digest(p)!=sig['files'][p.name]:raise ValueError(f'Data mismatch: {p}')
    if sig['train_limit'] or sig['val_images']:raise ValueError('Use a full-split Task 3 checkpoint')
    model=from_task2_bundle(a.bundle,**saved['config']['model']);model.load_state_dict(saved['model']);model.to(device).eval()
    baseline=from_task2_bundle(a.bundle)
    hard=HardRouter(baseline.gate,*baseline.experts).to(device).eval()
    dataset=EvaluationPetsDataset(files[1],files[2]);count=min(len(dataset),a.val_images*10) if a.val_images else len(dataset)
    loss=ReconstructionLoss(alpha=.8);rows=[]
    with torch.no_grad():
        for start in range(0,count,a.batch_size):
            b=batch_at(dataset,range(start,min(count,start+a.batch_size)));x=b['input'].to(device);target=b['target'].to(device)
            result=model(x);hard_result=hard(x)
            metrics={k:loss(v,target) for k,v in [('input',x),('task2',hard_result['output']),('task3',result['output'])]}
            for i in range(len(x)):
                w=result['weights'][i].cpu().numpy()
                r={'index':start+i,'image_id':b['image_id'][i],'condition':CLASSES[b['label'][i]],'severity':b['severity'][i],
                   'gate_correct':int(w.argmax())==int(b['label'][i]),'max_weight':float(w.max()),'entropy':float(-(w*np.log(np.maximum(w,1e-30))).sum())}
                r.update({f'weight_{c}':float(w[j]) for j,c in enumerate(CLASSES)})
                r.update({f'{k}_{m}':float(v[m][i]) for k,v in metrics.items() for m in ('l1','ssim','loss')});rows.append(r)
            if start%400==0:print(f'Analyzed {min(start+a.batch_size,count)}/{count}',flush=True)
    out=a.output_dir
    if out.exists() and any(out.iterdir()):raise ValueError('Choose empty output folder to preserve previous analysis')
    out.mkdir(parents=True,exist_ok=True)
    summary=summarize(rows);examples=select_examples(rows)
    summary['provenance']={'checkpoint_sha256':digest(a.checkpoint),'epoch':saved['progress']['epoch'],'case_count':count,'analysis_is_subset':bool(a.val_images),'split':'validation','alpha':.8,'task2_checkpoints':{k:v[2] for k,v in SELECTED.items()},'torch':str(torch.__version__)}
    summary['examples']=examples
    (out/'summary.json').write_text(json.dumps(summary,indent=2));write_csv(out/'per_case.csv',rows)
    for name,selected in examples.items():panel(out/f'{name}_grid.png',selected,dataset,model,hard,device)
    groups=summary['by_condition_severity'];heat=Image.new('RGB',(600,40+30*len(groups)),'white');draw=ImageDraw.Draw(heat)
    for j,c in enumerate(CLASSES):draw.text((170+105*j,5),c,fill='black')
    for i,r in enumerate(groups):
        y=30+i*30;draw.text((0,y),r['condition']+'/'+r['severity'],fill='black')
        for j,c in enumerate(CLASSES):
            w=r['weight_'+c];x=170+j*105;draw.rectangle((x,y,x+100,y+27),fill=(int(255*(1-w)),int(255*(1-w)),255));draw.text((x+5,y+5),f'{w:.3f}',fill='black' if w<.6 else 'white')
    heat.save(out/'routing_heatmap.png')
    (out/'README.md').write_text('Task 3 validation analysis. Fixed alpha .8; conditions equally weighted after severities.\nWeights ordered clean/salt/blur/occlusion. Error panels use fixed [0,1].\nDominated >=.9, distributed <.8 max weight; these are descriptive thresholds.\nExamples are selected, not population summaries. Empty categories produce no grid.\nNo official test images used. See summary.json and per_case.csv.\n')
    print(json.dumps(summary['condition_balanced'],indent=2));print('Saved',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--val-images',type=int,default=0)
    a=p.parse_args()
    if a.batch_size<1 or a.val_images<0:p.error('Invalid batch size or subset limit')
    run(a)
if __name__=='__main__':main()
