"""Aligned FS2K RGB pairs in [0,1], with one shared optional horizontal flip."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

class FS2KDataset(Dataset):
    def __init__(self, split_file, raw_dir, *, augment=False):
        self.root=Path(raw_dir).resolve()
        self.rows=json.loads(Path(split_file).read_text())
        self.augment=augment
        if not isinstance(self.rows,list) or not self.rows:raise ValueError('Empty/invalid split')
        seen=set()
        for row in self.rows:
            if row['image_id'] in seen:raise ValueError('Duplicate pair ID')
            seen.add(row['image_id'])
            if type(row['style']) is not int or row['style'] not in (0,1,2):raise ValueError('Invalid style')
            for kind in ('photo','sketch'):
                relative=Path(row[kind]);path=(self.root/relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(self.root):raise ValueError('Path escapes root')
                if not path.is_file():raise FileNotFoundError(path)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];images=[];sizes=[]
        flip=self.augment and bool(torch.rand(())<.5)
        for kind in ('photo','sketch'):
            path=self.root/row[kind]
            try:
                with Image.open(path) as source:
                    image=ImageOps.exif_transpose(source).convert('RGB');sizes.append(image.size)
                    image=image.resize((128,128),Image.Resampling.BILINEAR)
                    if flip:image=ImageOps.mirror(image)
                    array=np.array(image,dtype=np.float32,copy=True)/255.
            except (OSError,ValueError) as exc:raise RuntimeError(f'Cannot decode {path}') from exc
            images.append(torch.from_numpy(array).permute(2,0,1).contiguous())
        if sizes[0]!=sizes[1]:raise ValueError(f'Pair dimensions differ: {row["image_id"]}: {sizes}')
        return {'photo':images[0],'sketch':images[1],'style':row['style'],'image_id':row['image_id']}
