import asyncio,base64,io,unittest
from unittest.mock import patch
from PIL import Image
import httpx
from backend.app.main import app
from backend.app.task4_runtime import SketchService

class Task4APITest(unittest.TestCase):
 def test_styles_and_validation(self):
  service=SketchService('artifacts/task4-delivery')
  stream=io.BytesIO();Image.new('RGB',(160,140),'gray').save(stream,format='PNG');payload=stream.getvalue()
  async def inline(fn,*args):return fn(*args)
  async def check():
   app.state.sketch=service
   async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
    for style in (1,2,3):
     r=await client.post('/face-to-sketch',files={'file':('test.png',payload,'image/png')},data={'style':str(style)})
     self.assertEqual(r.status_code,200);result=r.json();self.assertEqual(result['style'],style)
     self.assertEqual(Image.open(io.BytesIO(base64.b64decode(result['image_base64']))).size,(128,128))
    r=await client.post('/face-to-sketch',files={'file':('test.png',payload,'image/png')},data={'style':'4'});self.assertEqual(r.status_code,422)
    app.state.sketch=None
    r=await client.post('/face-to-sketch',files={'file':('test.png',payload,'image/png')},data={'style':'1'});self.assertEqual(r.status_code,503)
  with patch('backend.app.main.run_in_threadpool',inline):asyncio.run(check())
if __name__=='__main__':unittest.main()
