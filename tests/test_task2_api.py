"""HTTP transport, real ONNX smoke test and deterministic expert dispatch."""
import base64
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import asyncio
import httpx
from backend.app.main import app
from backend.app.task2_runtime import HardRoutingService

ROOT = Path(__file__).resolve().parents[1]

def png():
    image = Image.fromarray(np.arange(128*128*3,dtype=np.uint8).reshape(128,128,3))
    stream = BytesIO()
    image.save(stream,format='PNG')
    return stream.getvalue()

class FakeSession:
    def __init__(self, label=None, value=None): self.label,self.value,self.calls=label,value,0
    def run(self, _, inputs):
        self.calls += 1
        if self.label is not None:
            logits=np.zeros((1,4),np.float32);logits[0,self.label]=10
            return [logits]
        return [np.full_like(inputs['image'],self.value)]

class Task2APITest(unittest.TestCase):
    def test_all_routes_and_clean_pixels(self):
        payload=png()
        for label in range(4):
            service=HardRoutingService.__new__(HardRoutingService)
            service.sessions={'classifier':FakeSession(label=label),**{n:FakeSession(value=.5) for n in ['salt','blur','occlusion']}}
            result=service.predict(payload)
            image=np.asarray(Image.open(BytesIO(base64.b64decode(result['image_base64']))))
            self.assertEqual(result['selected_expert'],['clean','salt','blur','occlusion'][label])
            self.assertAlmostEqual(sum(result['probabilities'].values()),1,places=6)
            if label==0: np.testing.assert_array_equal(image,np.asarray(Image.open(BytesIO(payload))))
            else: np.testing.assert_array_equal(image,np.full_like(image,128))
            for i,name in enumerate(['salt','blur','occlusion'],1):
                self.assertEqual(service.sessions[name].calls,int(i==label))

    def test_missing_bundle_and_invalid_upload(self):
        async def check():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                    self.assertFalse((await client.get('/health')).json()['inference_ready'])
                    self.assertEqual((await client.post('/hard-routing',files={'file':('x.png',png(),'image/png')})).status_code,503)
                    self.assertEqual((await client.post('/hard-routing',files={'file':('x.png',b'bad','image/png')})).status_code,415)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'MODEL_DIR':tmp}), self.assertLogs(level='ERROR'):
            asyncio.run(check())

    @unittest.skipUnless((ROOT/'artifacts/task2-delivery/manifest.json').exists(),'Local bundle required')
    def test_real_bundle_http(self):
        async def check():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                    self.assertTrue((await client.get('/health')).json()['inference_ready'])
                    result=await client.post('/hard-routing',files={'file':('x.png',png(),'image/png')})
                    self.assertEqual(result.status_code,200,result.text)
                    data=result.json()
                    self.assertEqual(Image.open(BytesIO(base64.b64decode(data['image_base64']))).size,(128,128))
                    self.assertAlmostEqual(sum(data['probabilities'].values()),1,places=5)
                    self.assertGreaterEqual(data['inference_ms'],0)
        with patch.dict(os.environ,{'MODEL_DIR':str(ROOT/'artifacts/task2-delivery')}):
            asyncio.run(check())

    def test_hash_mismatch_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'classifier').mkdir();(root/'classifier/model.onnx').write_bytes(b'wrong')
            (root/'manifest.json').write_text(json.dumps({'status':'verified','class_order':['clean','salt','blur','occlusion'],'models':{'classifier':{'onnx_sha256':'wrong'}}}))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):HardRoutingService(root)

if __name__=='__main__':unittest.main()
