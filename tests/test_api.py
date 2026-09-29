import io
import unittest
from PIL import Image
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.preprocessing import validate_image
from fastapi import HTTPException, UploadFile

class APIContractTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        output = io.BytesIO()
        Image.new('RGB', (16, 16)).save(output, format='PNG')
        self.png = output.getvalue()

    async def test_health_is_not_model_readiness(self):
        result = await self.client.get('/health')
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()['inference_ready'])

    async def test_all_pipelines_fail_honestly(self):
        for path in ['universal-restoration', 'hard-routing', 'soft-mixture', 'face-to-sketch']:
            with self.subTest(path=path):
                result = await self.client.post('/' + path, files={'file': ('test.png', self.png, 'image/png')}, data={'style': '1'})
                self.assertEqual(result.status_code, 503)

    async def test_invalid_image(self):
        result = await self.client.post('/universal-restoration', files={'file': ('bad.png', b'not an image', 'image/png')})
        self.assertEqual(result.status_code, 415)

    async def test_invalid_style(self):
        result = await self.client.post('/face-to-sketch', files={'file': ('test.png', self.png, 'image/png')}, data={'style': '4'})
        self.assertEqual(result.status_code, 422)

    async def test_oversized_upload(self):
        upload = UploadFile(filename='large.png', file=io.BytesIO(b'x' * (10 * 1024 * 1024 + 1)))
        with self.assertRaises(HTTPException) as caught:
            await validate_image(upload)
        self.assertEqual(caught.exception.status_code, 413)

    async def asyncTearDown(self):
        await self.client.aclose()
