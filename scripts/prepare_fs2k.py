"""Validate official FS2K pairs and reserve a seed-42 style-stratified validation split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


def read_pairs(root, annotation):
    entries=json.loads((root/annotation).read_text())
    rows=[];seen=set()
    for item in entries:
        name=item['image_name'];style=item['style']
        if not isinstance(name,str) or not re.fullmatch(r'photo[123]/image[^/\\.]+',name):
            raise ValueError(f'Unexpected image name: {name}')
        if type(style) is not int or style not in (0,1,2):raise ValueError('Expected official style 0,1,2')
        if name in seen:raise ValueError('Duplicate image ID: '+name)
        seen.add(name)
        sketch=name.replace('photo','sketch').replace('image','sketch')
        row={'image_id':name,'style':style}
        for kind,stem in [('photo',name),('sketch',sketch)]:
            base=root/kind/stem
            files=[p for p in base.parent.glob(base.name+'.*')
                   if p.is_file() and p.stem==base.name and p.suffix.lower() in ('.jpg','.jpeg','.png')]
            if len(files)!=1:raise ValueError(f'Expected one {kind} file for {name}, found {files}')
            path=files[0]
            if not path.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes root')
            row[kind]=path.relative_to(root).as_posix()
        rows.append(row)
    return rows


def split_train(rows):
    train=[];val=[];rng=random.Random(42)
    for style in range(3):
        group=sorted([r for r in rows if r['style']==style],key=lambda r:r['image_id'])
        if len(group)<2:raise ValueError('Each style needs at least two training pairs')
        rng.shuffle(group)
        count=max(1,min(len(group)-1,int(len(group)*.15+.5)))
        val.extend(group[:count]);train.extend(group[count:])
    return sorted(train,key=lambda r:r['image_id']),sorted(val,key=lambda r:r['image_id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('data/splits/fs2k'))
    a=p.parse_args();official=read_pairs(a.raw_dir,'anno_train.json');test=read_pairs(a.raw_dir,'anno_test.json')
    if len(official)!=1058 or len(test)!=1046:raise ValueError('Expected official counts: train1058/test1046')
    if {r['image_id'] for r in official}&{r['image_id'] for r in test}:raise ValueError('Official split overlap')
    train,val=split_train(official)
    metadata={'seed':42,'validation_fraction':.15,'rounding':'nearest integer per style, halves up','style_labels':'official0/1/2 map to UI Style1/2/3',
        'annotation_sha256':{n:hashlib.sha256((a.raw_dir/n).read_bytes()).hexdigest() for n in ['anno_train.json','anno_test.json']},
        'counts':{k:{'total':len(v),'by_style':dict(Counter(r['style'] for r in v))} for k,v in [('train',train),('val',val),('test',test)]}}
    outputs={'train.json':train,'val.json':val,'test.json':test,'metadata.json':metadata}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        path=a.output_dir/name;text=json.dumps(value,indent=2)+'\n'
        if path.exists() and path.read_text()!=text:raise ValueError('Existing split differs: '+str(path))
    for name,value in outputs.items():(a.output_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
