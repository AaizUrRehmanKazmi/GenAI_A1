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
from .task2_runtime import CLASSES

class SoftMixtureService:
    def __init__(self, model_dir):
        root=Path(model_dir);m=json.loads((root/'manifest.json').read_text())
        if m.get('status')!='verified' or m.get('task')!=3 or m.get('class_order')!=list(CLASSES):raise ValueError('Invalid Task 3 manifest')
        with (root/'model.onnx').open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=m['onnx_sha256']:raise ValueError('Task 3 model hash mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=2
        self.session=ort.InferenceSession(str(root/'model.onnx'),options,providers=['CPUExecutionProvider'])
        inputs=self.session.get_inputs()
        if len(inputs)!=1 or inputs[0].name!='image' or inputs[0].type!='tensor(float)' or inputs[0].shape[1:]!=[3,128,128]:raise ValueError('Invalid input contract')
        self.infer(np.zeros((1,3,128,128),np.float32))

    def infer(self,x):
        restored,weights=self.session.run(['restored','weights'],{'image':x})
        if restored.shape!=x.shape or weights.shape!=(len(x),4) or not np.isfinite(restored).all() or not np.isfinite(weights).all():raise ValueError('Invalid outputs')
        if weights.min()<0 or not np.allclose(weights.sum(1),1,atol=1e-5):raise ValueError('Invalid routing weights')
        return restored,weights

    def predict(self,payload):
        with Image.open(BytesIO(payload)) as im:
            im=ImageOps.exif_transpose(im).convert('RGB').resize((128,128),Image.Resampling.BILINEAR)
            x=np.ascontiguousarray((np.asarray(im,dtype=np.float32)/255).transpose(2,0,1)[None])
        start=perf_counter();restored,weights=self.infer(x);elapsed=(perf_counter()-start)*1000
        pixels=np.rint(np.clip(restored[0].transpose(1,2,0),0,1)*255).astype(np.uint8)
        stream=BytesIO();Image.fromarray(pixels).save(stream,format='PNG')
        return {'image_base64':base64.b64encode(stream.getvalue()).decode(),'inference_ms':elapsed,'routing_weights':dict(zip(CLASSES,map(float,weights[0])))}
