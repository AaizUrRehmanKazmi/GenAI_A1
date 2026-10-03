"""Export selected Task 3 complete mixture with image and weight outputs."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import torch
import onnx
import onnxruntime as ort
from src.models.soft_moe import from_task2_bundle
from src.data.pets_dataset import PetsDataset, REPO_ROOT
from src.data.corruptions import corrupt, CLASSES
from exports.task2_bundle import digest, compare

EXPECTED='024a388659c2c55596b68a0bcf814c0aaa0b7c2aab016c168fd8a827f28f9a3d'
class Inference(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):
        result=self.model(x)
        return result['output'],result['weights']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup',type=Path,required=True);p.add_argument('--task2-bundle',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True);torch.set_num_threads(2)
    with zipfile.ZipFile(a.backup) as z:
        for name in ['best.pt','config.yaml','history.json']:(out/name).write_bytes(z.read('trial_005/'+name))
    if digest(out/'best.pt')!=EXPECTED:raise ValueError('Wrong selected checkpoint')
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    if saved['data_signature']['train_limit'] or saved['data_signature']['val_images']:raise ValueError('Debug weights')
    for source,value in saved['data_signature']['source'].items():
        if digest(REPO_ROOT/source)!=value:raise ValueError('Source mismatch: '+source)
    model=from_task2_bundle(a.task2_bundle,**saved['config']['model']);model.load_state_dict(saved['model']);wrapper=Inference(model).eval()
    path=out/'model.onnx'
    torch.onnx.export(wrapper,torch.zeros(1,3,128,128),str(path),input_names=['image'],output_names=['restored','weights'],dynamic_axes={n:{0:'batch'} for n in ['image','restored','weights']},opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    dataset=PetsDataset(REPO_ROOT/'data/splits/pets_val.json');batches=[]
    for i in range(4):batches.append(torch.stack([corrupt(dataset[i]['image'],seed=i*4+j,condition=c)[0] for j,c in enumerate(CLASSES)]))
    batches.extend([torch.zeros(1,3,128,128),torch.ones(1,3,128,128),torch.rand(3,3,128,128,generator=torch.Generator().manual_seed(42))])
    checks=[]
    with torch.inference_mode():
        for x in batches:
            ref=wrapper(x);actual=session.run(['restored','weights'],{'image':x.numpy()})
            checks.append({n:compare(r.numpy(),v) for n,r,v in zip(['restored','weights'],ref,actual)})
            np.testing.assert_allclose(actual[1].sum(1),1,atol=1e-6)
            assert actual[1].min()>=0 and actual[0].min()>=-1e-6 and actual[0].max()<=1+1e-6
    manifest={'status':'verified','task':3,'checkpoint_sha256':EXPECTED,'onnx_sha256':digest(path),'epoch':saved['progress']['epoch'],'temperature':saved['config']['model']['temperature'],'class_order':list(CLASSES),'input':'float32 NCHW RGB [N,3,128,128], EXIF transpose, PIL bilinear resize, [0,1]','outputs':['restored','weights'],'opset':17,'checks':checks,'versions':{'torch':str(torch.__version__),'onnx':onnx.__version__,'onnxruntime':ort.__version__},'verification':'CPU, 16 validation cases + zeros/ones/random; batches 1,3,4; rtol1e-4 atol1e-5; no official test images'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'README.md').write_text('Complete selected Task 3 mixture in one ONNX graph. Outputs: restored image and four routing weights ordered clean/salt/blur/occlusion. All branches contribute; no argmax dispatch. See manifest.json for preprocessing, provenance and parity. Retain original training backup for resume.\n')
    print(shutil.make_archive(str(out),'zip',out.parent,out.name))
if __name__=='__main__':main()
