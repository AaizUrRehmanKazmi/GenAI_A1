"""Task 1 universal restoration ONNX service."""
import base64
import hashlib
import json
from pathlib import Path
from io import BytesIO
from time import perf_counter
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


class UniversalRestorationService:
    def __init__(self, model_dir):
        root = Path(model_dir)
        m = json.loads((root / 'manifest.json').read_text())
        if m.get('status') != 'verified' or m.get('task') != 1:
            raise ValueError('Invalid Task 1 manifest')
        with (root / 'model.onnx').open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != m['onnx_sha256']:
                raise ValueError('Task 1 model hash mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(root / 'model.onnx'), options,
            providers=['CPUExecutionProvider']
        )
        inputs = self.session.get_inputs()
        if (len(inputs) != 1 or inputs[0].name != 'image'
                or inputs[0].type != 'tensor(float)'
                or inputs[0].shape[1:] != [3, 128, 128]):
            raise ValueError('Invalid input contract')
        # Warm-up
        out = self.session.run(None, {'image': np.zeros((1, 3, 128, 128), np.float32)})[0]
        if out.shape != (1, 3, 128, 128) or not np.isfinite(out).all():
            raise ValueError('Invalid output contract')

    def predict(self, payload):
        with Image.open(BytesIO(payload)) as im:
            im = ImageOps.exif_transpose(im).convert('RGB').resize(
                (128, 128), Image.Resampling.BILINEAR
            )
            x = np.ascontiguousarray(
                (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
            )
        start = perf_counter()
        restored = self.session.run(['restored'], {'image': x})[0]
        elapsed = (perf_counter() - start) * 1000
        pixels = np.rint(np.clip(restored[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format='PNG')
        return {
            'image_base64': base64.b64encode(stream.getvalue()).decode('ascii'),
            'inference_ms': elapsed
        }
