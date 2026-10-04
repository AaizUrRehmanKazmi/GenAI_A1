"""Package selected fine-tuned cGAN generator and verify all style inputs."""
import argparse,json,zipfile,shutil
from pathlib import Path
import torch,numpy as np,onnx,onnxruntime as ort
from src.models.task4_candidate import CandidateGenerator
from exports.task2_bundle import digest,compare
from src.data.fs2k_dataset import FS2KDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--backup',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise ValueError('Use new output directory')
 out=a.output_dir;out.mkdir(parents=True);torch.set_num_threads(2)
 with zipfile.ZipFile(a.backup) as z:
  for name in ('best.pt','history.json','config.yaml'):(out/name).write_bytes(z.read('task4-finetune-comparison/gan/'+name))
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
 assert ck['progress']['epoch']==1 and ck['config']['loss']['adversarial_weight']==.1
 assert not ck['signature']['train_limit'] and not ck['signature']['val_per_style']
 for file,sha in ck['signature']['source'].items():
  if digest(Path(file))!=sha:raise ValueError('Source mismatch '+file)
 g=CandidateGenerator(**ck['config']['generator']);g.load_state_dict(ck['generator']);g.eval()
 path=out/'model.onnx'
 torch.onnx.export(g,(torch.zeros(1,3,128,128),torch.zeros(1,dtype=torch.long)),str(path),input_names=['image','style'],output_names=['sketch'],dynamic_axes={k:{0:'batch'} for k in ('image','style','sketch')},opset_version=17,dynamo=False)
 onnx.checker.check_model(onnx.load(str(path)))
 options=ort.SessionOptions();options.intra_op_num_threads=2
 session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
 ds=FS2KDataset('data/splits/fs2k/val.json','data/raw/fs2k/FS2K');real=torch.stack([ds[i]['photo'] for i in range(3)])
 checks=[]
 with torch.inference_mode():
  for batch in (real,torch.zeros(1,3,128,128),torch.ones(1,3,128,128)):
   for style in range(3):
    s=torch.full((len(batch),),style,dtype=torch.long);ref=g(batch,s).numpy();actual=session.run(['sketch'],{'image':batch.numpy(),'style':s.numpy()})[0]
    checks.append({'style':style,'batch':len(batch),**compare(ref,actual)})
    assert np.isfinite(actual).all() and actual.min()>=-1e-6 and actual.max()<=1+1e-6
 manifest={'status':'verified','task':4,'style_ids':[0,1,2],'api_styles':[1,2,3],'checkpoint_sha256':digest(out/'best.pt'),'onnx_sha256':digest(path),'epoch':1,'initialization_sha256':ck['signature']['initialization_sha256'],'checks':checks,'input':'RGB EXIF-transposed bilinear128 float32 NCHW [0,1]; style int64[N] 0..2','limitation':'Blurred fine detail; adversarial ablation did not materially improve validation. Official test not evaluated.'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(shutil.make_archive(str(out),'zip',out.parent,out.name))
if __name__=='__main__':main()
