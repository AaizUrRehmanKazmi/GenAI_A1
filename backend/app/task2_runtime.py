"""Verified CPU ONNX inference; no training dependencies required."""
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

CLASSES = ('clean', 'salt', 'blur', 'occlusion')

class HardRoutingService:
    def __init__(self, model_dir):
        root = Path(model_dir)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('status') != 'verified' or manifest.get('class_order') != list(CLASSES):
            raise ValueError('Unsupported model manifest')
        self.sessions = {}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        for name in ('classifier', 'salt', 'blur', 'occlusion'):
            path = root / name / 'model.onnx'
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != manifest['models'][name]['onnx_sha256']:
                raise ValueError(f'Model hash mismatch: {name}')
            session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            inputs = session.get_inputs()
            if len(inputs) != 1 or inputs[0].name != 'image' or inputs[0].type != 'tensor(float)' or inputs[0].shape[1:] != [3,128,128]:
                raise ValueError(f'Invalid input contract: {name}')
            output = session.run(None, {'image': np.zeros((1,3,128,128), np.float32)})[0]
            shape = (1,4) if name == 'classifier' else (1,3,128,128)
            if output.shape != shape or not np.isfinite(output).all():
                raise ValueError(f'Invalid output contract: {name}')
            self.sessions[name] = session

    def predict(self, payload):
        with Image.open(BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image).convert('RGB').resize((128,128), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = np.ascontiguousarray(pixels.transpose(2,0,1)[None])
        start = perf_counter()
        logits = self.sessions['classifier'].run(None, {'image': tensor})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        label = int(probabilities.argmax())
        output = tensor if label == 0 else self.sessions[CLASSES[label]].run(None, {'image': tensor})[0]
        elapsed = (perf_counter() - start) * 1000
        pixels = np.rint(np.clip(output[0].transpose(1,2,0),0,1)*255).astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format='PNG')
        return {'image_base64': base64.b64encode(buffer.getvalue()).decode('ascii'),
                'inference_ms': elapsed, 'probabilities': dict(zip(CLASSES, map(float, probabilities))),
                'selected_expert': CLASSES[label]}
