"""Fit fixed clean training images to themselves: a capacity/optimization diagnostic.

No corruption, augmentation, validation/test data, or baseline weights are used.
One step is one full-batch update on the same selected training images.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_images(split_file, raw_dir, count):
    dataset = PetsDataset(split_file, raw_dir)
    if not 1 <= count <= len(dataset):
        raise ValueError('Image count must be between 1 and the training split size.')
    samples = [dataset[i] for i in range(count)]
    return torch.stack([s['image'] for s in samples]), [s['path'] for s in samples]


def save_grid(targets, predictions, paths, destination):
    """Two sample triptychs per row: target / prediction / absolute error."""
    width, height = 400, 165
    canvas = Image.new('RGB', (width * 2, height * math.ceil(len(paths)/2)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        x, y = (i % 2)*width, (i//2)*height
        draw.text((x+2, y), Path(path).stem, fill='black')
        for j, (label, tensor) in enumerate(zip(['Target', 'Output', 'Abs. error'],
                [targets[i], predictions[i], (targets[i]-predictions[i]).abs()])):
            draw.text((x+j*132+2, y+14), label, fill='black')
            pixels = (tensor.detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(pixels), (x+j*132+2, y+29))
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
    config = {'purpose': 'clean_training_reconstruction_diagnostic', 'model': model_config,
              'loss': original['loss'], 'learning_rate': args.learning_rate,
              'seed': args.seed, 'images': args.images}
    targets, paths = prepare_images(args.train_split, args.raw_dir, args.images)
    signature = {'train_split_sha256': sha(args.train_split), 'selected_paths': paths,
                 'preprocessed_tensor_sha256': hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
                 'source': {name: sha(REPO_ROOT/name) for name in [
                     'training/diagnose_task1.py', 'training/train_task1.py',
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
    targets = targets.to(device)
    mlflow.set_tracking_uri((output/'mlruns').as_uri())
    mlflow.set_experiment('task1-clean-diagnostic')
    started = time.monotonic()
    with mlflow.start_run(run_name='clean-image-fit') as tracking, stop_requests() as stop:
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
            model.eval()
            with torch.inference_mode():
                predictions = model(targets)
                values = criterion(predictions, targets)
                metrics = {name: value.mean().item() for name, value in values.items()}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError('Non-finite diagnostic metrics.')
            step = progress['step']
            progress['history'].append({'step':step, **metrics})
            improved = metrics['loss'] < progress['best']
            if improved:
                progress['best'] = metrics['loss']
                save('best.pt')
            save_grid(targets, predictions, paths, output/f'step_{step:06d}.png')
            (output/'history.json').write_text(json.dumps(progress['history'],indent=2)+'\n')
            mlflow.log_metrics(metrics,step=step)
            mlflow.log_artifact(str(output/f'step_{step:06d}.png'),artifact_path='reconstructions')
            print(f'Step {step}/{args.steps}: loss={metrics["loss"]:.5f}, L1={metrics["l1"]:.5f}, SSIM={metrics["ssim"]:.4f}',flush=True)
        print(f'DIAGNOSTIC ONLY: {len(paths)} fixed clean training images; dropout=0; device={device}',flush=True)
        if not progress['history']:
            measure()
        save()
        while progress['step'] < args.steps:
            if stop['requested'] or time.monotonic()-started >= args.max_minutes*60:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(targets),targets)['loss'].mean()
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
        for name in ['config.json','selected_images.json','provenance.json','environment.json','history.json','last.pt']:
            mlflow.log_artifact(str(output/name))
        mlflow.set_tag('completion','finished' if progress['step']==args.steps else 'paused')
    print(f'Saved {output}; completed updates: {progress["step"]}. These are training-fit metrics, not validation results.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO_ROOT/'configs/task1.yaml')
    parser.add_argument('--train-split',type=Path,default=REPO_ROOT/'data/splits/pets_train.json')
    parser.add_argument('--raw-dir',type=Path,default=REPO_ROOT/'data/raw/oxford_pets')
    parser.add_argument('--output-dir',type=Path,default=REPO_ROOT/'artifacts/task1-diagnostic')
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
