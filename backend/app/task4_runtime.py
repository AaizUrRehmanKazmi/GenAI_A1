"""Complete soft mixture ONNX service; exposes weights without hard dispatch."""
import base64
import hashlib
import json
from pathlib import Path
from io import BytesIO
from time import perf_counter
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

class SketchService:
    def __init__(self, model_dir):
        root=Path(model_dir);m=json.loads((root/'manifest.json').read_text())
        if m.get('status')!='verified' or m.get('task')!=4 or m.get('style_ids')!=[0,1,2]:raise ValueError('Invalid Task 4 manifest')
        with (root/'model.onnx').open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=m['onnx_sha256']:raise ValueError('Task 4 model hash mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=2
        self.session=ort.InferenceSession(str(root/'model.onnx'),options,providers=['CPUExecutionProvider'])
        for style in (1,2,3):self.infer(np.zeros((1,3,128,128),np.float32),style)

    def infer(self,x,style):
        if type(style)!=int or style not in (1,2,3):raise ValueError('Style must be 1, 2 or 3')
        restored=self.session.run(['sketch'],{'image':x,'style':np.full((len(x),),style-1,dtype=np.int64)})[0]
        if restored.shape!=x.shape or not np.isfinite(restored).all() or restored.min() < -1e-5 or restored.max()>1+1e-5:raise ValueError('Invalid sketch output')
        return restored

    def predict(self,payload,style):
        with Image.open(BytesIO(payload)) as im:
            im=ImageOps.exif_transpose(im).convert('RGB').resize((128,128),Image.Resampling.BILINEAR)
            x=np.ascontiguousarray((np.asarray(im,dtype=np.float32)/255).transpose(2,0,1)[None])
        start=perf_counter();restored=self.infer(x,style);elapsed=(perf_counter()-start)*1000
        pixels=np.rint(np.clip(restored[0].transpose(1,2,0),0,1)*255).astype(np.uint8)
        stream=BytesIO();Image.fromarray(pixels).save(stream,format='PNG')
        return {'image_base64':base64.b64encode(stream.getvalue()).decode(),'inference_ms':elapsed,'style':style}
