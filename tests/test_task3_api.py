import asyncio
import base64
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
import numpy as np
from PIL import Image
from backend.app.main import app
from backend.app.task3_runtime import SoftMixtureService
ROOT=Path(__file__).resolve().parents[1]

async def inline(function,*args):return function(*args)

class Task3APITest(unittest.TestCase):
    @unittest.skipUnless((ROOT/'artifacts/task3-delivery/model.onnx').exists(),'Requires exported bundle')
    def test_real_api_weights_and_image(self):
        async def check():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                    self.assertTrue((await client.get('/health')).json()['tasks']['soft-mixture'])
                    stream=BytesIO();Image.new('RGB',(150,100),(50,100,150)).save(stream,format='PNG')
                    response=await client.post('/soft-mixture',files={'file':('test.png',stream.getvalue(),'image/png')})
                    self.assertEqual(response.status_code,200,response.text)
                    r=response.json();self.assertAlmostEqual(sum(r['routing_weights'].values()),1,places=5)
                    self.assertIsNone(r['selected_expert'])
                    self.assertEqual(Image.open(BytesIO(base64.b64decode(r['image_base64']))).size,(128,128))
                    self.assertEqual((await client.post('/soft-mixture',files={'file':('bad.png',b'bad','image/png')})).status_code,415)
        with patch.dict(os.environ,{'MODEL_DIR':str(ROOT/'artifacts/task2-delivery'),'TASK3_MODEL_DIR':str(ROOT/'artifacts/task3-delivery')}),patch('backend.app.main.run_in_threadpool',inline):asyncio.run(check())
    def test_unavailable_returns_503(self):
        async def check():
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
                    self.assertFalse((await c.get('/health')).json()['tasks']['soft-mixture'])
                    stream=BytesIO();Image.new('RGB',(16,16)).save(stream,format='PNG')
                    self.assertEqual((await c.post('/soft-mixture',files={'file':('x.png',stream.getvalue(),'image/png')})).status_code,503)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'MODEL_DIR':tmp,'TASK3_MODEL_DIR':tmp}),self.assertLogs(level='ERROR'):asyncio.run(check())
    def test_invalid_weights_rejected(self):
        class Bad:
            def run(self,*args):return np.zeros((1,3,128,128)),np.zeros((1,4))
        service=SoftMixtureService.__new__(SoftMixtureService);service.session=Bad()
        with self.assertRaises(ValueError):service.infer(np.zeros((1,3,128,128),np.float32))
if __name__=='__main__':unittest.main()
