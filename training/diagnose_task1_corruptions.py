"""Fit the same tiny training set with fresh corruptions and fixed evaluation cases.

Not validation/generalization evidence: evaluation uses the selected training images.
The clean diagnostic and baseline implementation are left unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import mlflow
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from src.data.pets_dataset import PetsDataset, REPO_ROOT
from src.models.universal_ae import UniversalAutoencoder
from src.losses.reconstruction import ReconstructionLoss
from training.train_task1 import save_checkpoint, restore_checkpoint, stop_requests
from training.diagnose_task1 import prepare_images
from src.data.corruptions import corrupt, apply_corruption, make_spec, CLASSES
from src.data.manifests import CASES
from evaluation.evaluate_task1 import summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_cases(targets, paths, seed):
    inputs, clean, records = [], [], []
    for i, path in enumerate(paths):
        for condition, severity, options in CASES:
            key = json.dumps(["tiny-corruption-evaluation", seed, path, condition, severity]).encode()
            case_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63)
            spec = make_spec(condition, seed=case_seed, **options)
            inputs.append(apply_corruption(targets[i], spec))
            clean.append(targets[i])
            records.append({'image_id':Path(path).stem, 'path':path, 'condition':condition,
                            'severity':severity, 'spec':spec})
    return torch.stack(inputs), torch.stack(clean), records


def training_inputs(targets):
    # targets stay on CPU for the corruption pipeline; fresh seed per image/access.
    return torch.stack([corrupt(image, seed=torch.randint(0, 2**63-1, ()).item())[0]
                        for image in targets])


def measure_fixed(model, criterion, inputs, targets, records, device):
    rows, outputs = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 16):
            x, y = inputs[start:start+16].to(device), targets[start:start+16].to(device)
            prediction = model(x)
            before = {k:v.cpu().tolist() for k,v in criterion(x,y).items()}
            after = {k:v.cpu().tolist() for k,v in criterion(prediction,y).items()}
            outputs.append(prediction.cpu())
            for i in range(len(x)):
                record = records[start+i]
                row = {'index':start+i, 'image_id':record['image_id'], 'condition':record['condition'],
                       'severity':record['severity']}
                for metric in ['l1','ssim','loss']:
                    row['input_'+metric] = before[metric][i]
                    row['output_'+metric] = after[metric][i]
                    row[metric+'_gain'] = (after[metric][i]-before[metric][i]) * (1 if metric=='ssim' else -1)
                rows.append(row)
    return summarize(rows), torch.cat(outputs)


def save_grid(inputs, targets, predictions, records, destination):
    # All ten cases for the first image, then additional images in separate grids.
    canvas=Image.new('RGB',(640,len(records)*166),'white')
    draw=ImageDraw.Draw(canvas)
    for i,record in enumerate(records):
        y=i*166
        draw.text((2,y),f"{record['image_id']} | {record['condition']} {record['severity']}",fill='black')
        for j,(label,tensor) in enumerate(zip(['Target','Input','Output','Abs. error'],
                [targets[i],inputs[i],predictions[i],(targets[i]-predictions[i]).abs()])):
            draw.text((j*160+2,y+14),label,fill='black')
            pixels=(tensor.clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(pixels),(j*160+16,y+29))
    canvas.save(destination)


def run(args):
    if min(args.steps, args.images, args.report_every, args.checkpoint_every, args.cpu_threads) < 1 or args.max_minutes <= 0:
        raise ValueError('Counts and time budget must be positive.')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available()
                          else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; select a GPU runtime or use CPU for a short smoke check.')
    original = yaml.safe_load(args.config.read_text())
    model_config = dict(original['model'])
    model_config['dropout'] = 0.0
    config = {'purpose': 'corrupted_training_reconstruction_diagnostic', 'model': model_config,
              'loss': original['loss'], 'learning_rate': args.learning_rate,
              'seed': args.seed, 'images': args.images}
    targets, paths = prepare_images(args.train_split, args.raw_dir, args.images)
    fixed_inputs, fixed_targets, records = fixed_cases(targets, paths, args.seed)
    signature = {'train_split_sha256': sha(args.train_split), 'selected_paths': paths,
                 'preprocessed_tensor_sha256': hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
                 'source': {name: sha(REPO_ROOT/name) for name in [
                     'training/diagnose_task1_corruptions.py', 'training/diagnose_task1.py', 'training/train_task1.py',
                     'src/data/corruptions.py', 'src/data/manifests.py', 'evaluation/evaluate_task1.py',
                     'src/models/universal_ae.py', 'src/losses/reconstruction.py', 'src/data/pets_dataset.py']}}
    model = UniversalAutoencoder(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = ReconstructionLoss(**config['loss']).to(device)
    progress = {'step': 0, 'best': math.inf, 'history': [], 'previous_run_id': None}
    output = args.output_dir.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty. Use a new folder or --resume; baseline folders must remain unchanged.')
    if args.resume:
        progress = restore_checkpoint(args.resume, model, optimizer, config, signature, device)
        if args.steps <= progress['step']:
            raise ValueError('--steps must exceed the number of completed updates when resuming.')
        if output.exists() and any(output.iterdir()) and args.resume.resolve().parent != output:
            raise ValueError('Resume into its original folder or a new empty directory.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output/'selected_images.json').write_text(json.dumps(paths, indent=2)+'\n')
    (output/'provenance.json').write_text(json.dumps(signature, indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps({'torch':str(torch.__version__),
        'device':str(device), 'cpu_threads':args.cpu_threads, 'parameters':sum(p.numel() for p in model.parameters())}, indent=2)+'\n')
    cpu_targets = targets
    targets = targets.to(device)
    (output/'fixed_training_cases.json').write_text(json.dumps(records,indent=2)+'\n')
    mlflow.set_tracking_uri((output/'mlruns').as_uri())
    mlflow.set_experiment('task1-corruption-diagnostic')
    started = time.monotonic()
    with mlflow.start_run(run_name='corrupted-image-fit') as tracking, stop_requests() as stop:
        if progress['previous_run_id']:
            mlflow.set_tag('resumes_run_id', progress['previous_run_id'])
        progress['previous_run_id'] = tracking.info.run_id
        mlflow.log_params({'images':args.images, 'learning_rate':args.learning_rate,
                          'seed':args.seed, 'dropout':0., 'latent_dim':model_config['latent_dim'],
                          'channels':str(model_config['channels']), 'alpha':config['loss']['alpha'],
                          'device':str(device), 'target_steps':args.steps})
        def save(name='last.pt'):
            save_checkpoint(output/name, model, optimizer, progress, config, signature, device)
        def measure():
            comparison, predictions = measure_fixed(model, criterion, fixed_inputs, fixed_targets, records, device)
            balanced = comparison['condition_balanced']
            metrics = {name:balanced['output_'+name] for name in ['loss','l1','ssim']}
            metrics.update({name+'_gain':balanced[name+'_gain'] for name in ['loss','l1','ssim']})
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError('Non-finite diagnostic metrics.')
            step = progress['step']
            progress['history'].append({'step':step, **metrics, 'comparison':comparison})
            improved = metrics['loss'] < progress['best']
            if improved:
                progress['best'] = metrics['loss']
                save('best.pt')
            for image_index in range(min(4,len(paths))):
                start=image_index*10
                filename=output/f'step_{step:06d}_image_{image_index+1:02d}.png'
                save_grid(fixed_inputs[start:start+10], fixed_targets[start:start+10], predictions[start:start+10],
                          records[start:start+10], filename)
                mlflow.log_artifact(str(filename),artifact_path='reconstructions')
            (output/'latest_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            (output/'history.json').write_text(json.dumps(progress['history'],indent=2)+'\n')
            mlflow.log_metrics(metrics,step=step)
            print(f'Step {step}/{args.steps}: loss={metrics["loss"]:.5f}, L1={metrics["l1"]:.5f}, SSIM={metrics["ssim"]:.4f}, SSIM gain={metrics["ssim_gain"]:+.4f}',flush=True)
        print(f'DIAGNOSTIC ONLY: {len(paths)} training images with fresh corruptions; fixed evaluation on those SAME images; dropout=0; device={device}',flush=True)
        if not progress['history']:
            measure()
        save()
        while progress['step'] < args.steps:
            if stop['requested'] or time.monotonic()-started >= args.max_minutes*60:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            inputs = training_inputs(cpu_targets).to(device)
            loss = criterion(model(inputs),targets)['loss'].mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stop and investigate.')
            loss.backward()
            optimizer.step()
            progress['step'] += 1
            if progress['step'] % args.report_every == 0 or progress['step']==args.steps:
                measure()
            if progress['step'] % args.checkpoint_every == 0:
                save()
        if progress['history'][-1]['step'] != progress['step']:
            measure()
        save()
        for name in ['config.json','selected_images.json','provenance.json','environment.json','history.json','fixed_training_cases.json','latest_comparison.json','last.pt']:
            mlflow.log_artifact(str(output/name))
        mlflow.set_tag('completion','finished' if progress['step']==args.steps else 'paused')
    print(f'Saved {output}; completed updates: {progress["step"]}. These are training-fit metrics, not validation results.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO_ROOT/'configs/task1.yaml')
    parser.add_argument('--train-split',type=Path,default=REPO_ROOT/'data/splits/pets_train.json')
    parser.add_argument('--raw-dir',type=Path,default=REPO_ROOT/'data/raw/oxford_pets')
    parser.add_argument('--output-dir',type=Path,default=REPO_ROOT/'artifacts/task1-corruption-diagnostic')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--images',type=int,default=16)
    parser.add_argument('--steps',type=int,default=1000,help='Total full-batch optimizer updates, not additional updates.')
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--report-every',type=int,default=50)
    parser.add_argument('--checkpoint-every',type=int,default=25)
    parser.add_argument('--max-minutes',type=float,default=60)
    parser.add_argument('--cpu-threads',type=int,default=2)
    args=parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.max_minutes):
        parser.error('Learning rate/time budget must be finite and positive.')
    run(args)


if __name__=='__main__':
    main()
