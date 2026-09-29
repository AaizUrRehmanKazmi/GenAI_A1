"""Task 1 baseline with local MLflow tracking, validation and resumable batches.

Run from repository root: python -m training.train_task1 --help
Training uses num_workers=0 to make saved RNG/cursor continuation unambiguous.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import mlflow
import numpy as np
from PIL import Image, ImageDraw
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import Subset
from torch.utils.data._utils.collate import default_collate
import yaml

from src.data.pets_dataset import REPO_ROOT, TrainingPetsDataset, EvaluationPetsDataset
from src.models.universal_ae import UniversalAutoencoder
from src.losses.reconstruction import ReconstructionLoss


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, progress, config, signature, device):
    atomic_save({'version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'progress': progress, 'config': config, 'data_signature': signature,
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                 'device_type': device.type}, path)


def restore_checkpoint(path, model, optimizer, config, signature, device):
    # Load only tensor/primitive checkpoint structures, not arbitrary pickled classes.
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['version'] != 1 or saved['config'] != config or saved['data_signature'] != signature:
        raise ValueError('Resume configuration or dataset fingerprints differ from the checkpoint.')
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    if device.type == 'cuda' and saved['device_type'] == 'cuda':
        if len(saved['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed; exact RNG restoration is unavailable.')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    elif saved['device_type'] != device.type:
        print('Resuming on a different device type; numerical trajectory may differ.', flush=True)
    return saved['progress']


@contextmanager
def stop_requests():
    state = {'requested': False}
    previous = {}
    def request(signum, frame):
        state['requested'] = True
        print('Stop requested; saving at the next safe batch boundary.', flush=True)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        previous[sig] = signal.signal(sig, request)
    try:
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def batch_at(dataset, indices):
    return default_collate([dataset[i] for i in indices])


def validation(model, dataset, criterion, device, batch_size, should_stop):
    model.eval()
    groups = {}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if should_stop():
                return None
            batch = batch_at(dataset, range(start, min(start + batch_size, len(dataset))))
            values = criterion(model(batch['input'].to(device)), batch['target'].to(device))
            for i, label in enumerate(batch['label'].tolist()):
                key = f'{label}/{batch["severity"][i]}'
                group = groups.setdefault(key, {'count': 0, 'loss': 0., 'l1': 0., 'ssim': 0.})
                group['count'] += 1
                for metric in ['loss', 'l1', 'ssim']:
                    group[metric] += values[metric][i].item()
    means = {key: {m: row[m] / row['count'] for m in ['loss', 'l1', 'ssim']}
             for key, row in groups.items()}
    # Average severities within each condition, then equally weight four conditions.
    balanced = {}
    for metric in ['loss', 'l1', 'ssim']:
        condition_means = []
        for label in range(4):
            rows = [row[metric] for key, row in means.items() if key.startswith(f'{label}/')]
            if not rows:
                raise ValueError('Validation subset must cover all four conditions.')
            condition_means.append(sum(rows) / len(rows))
        balanced[metric] = sum(condition_means) / 4
    return {'balanced': balanced, 'by_condition_severity': means}


def preview(model, dataset, device, path):
    # First validation image, all ten fixed cases; always the same across epochs.
    model.eval()
    with torch.no_grad():
        batch = batch_at(dataset, range(min(10, len(dataset))))
        predictions = model(batch['input'].to(device)).cpu()
    count = len(predictions)
    canvas = Image.new('RGB', (4 * 128, count * 150), 'white')
    draw = ImageDraw.Draw(canvas)
    for i in range(count):
        panels = [batch['target'][i], batch['input'][i], predictions[i],
                  (predictions[i] - batch['target'][i]).abs()]
        for column, (label, tensor) in enumerate(zip(['Target', 'Input', 'Output', 'Absolute error'], panels)):
            draw.text((column * 128, i * 150), label, fill='black')
            arr = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(arr), (column * 128, i * 150 + 18))
    canvas.save(path)


def run(args):
    started = time.monotonic()
    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    every = config['training']['checkpoint_every_batches']
    if (epochs < 1 or batch_size < 1 or every < 1 or args.max_hours <= 0
            or args.train_limit < 0 or args.val_images < 0):
        raise ValueError('Epochs, batch size, checkpoint interval and time must be positive; limits nonnegative.')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable. Use CPU for smoke checks only.')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    train_path = args.splits_dir / 'pets_train.json'
    val_path = args.splits_dir / 'pets_val.json'
    val_manifest = args.manifests_dir / 'pets_val_corruptions.json'
    train = TrainingPetsDataset(train_path, args.raw_dir)
    val = EvaluationPetsDataset(val_path, val_manifest, args.raw_dir)
    if args.train_limit:
        train = Subset(train, range(min(args.train_limit, len(train))))
    if args.val_images:
        val = Subset(val, range(min(args.val_images * 10, len(val))))
    signature = {
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path, val_path, val_manifest]},
        'train_limit': args.train_limit, 'val_images': args.val_images,
        'preprocessing': 'RGB-exif-bilinear-128-float32-0to1',
        'source': {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [REPO_ROOT / name for name in ['src/data/pets_dataset.py', 'src/data/corruptions.py',
                       'src/models/universal_ae.py', 'src/losses/reconstruction.py', 'training/train_task1.py']]},
    }
    model = UniversalAutoencoder(**config['model']).to(device)
    criterion = ReconstructionLoss(**config['loss']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.resume is None and (output / 'last.pt').exists():
        raise ValueError('Output already has a checkpoint. Use --resume or a new --output-dir.')
    progress = {'epoch': 0, 'order': None, 'cursor': 0, 'train_sum': 0., 'train_count': 0,
                'best': math.inf, 'history': [], 'previous_run_id': None}
    if args.resume:
        progress = restore_checkpoint(args.resume, model, optimizer, config, signature, device)
    (output / 'config.yaml').write_text(yaml.safe_dump(config))
    (output / 'data_signature.json').write_text(json.dumps(signature, indent=2))
    versions = {'torch': str(torch.__version__), 'device': str(device),
                'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                'parameters': sum(p.numel() for p in model.parameters()),
                'debug_subset': bool(args.train_limit or args.val_images)}
    (output / 'environment.json').write_text(json.dumps(versions, indent=2))
    mlflow.set_tracking_uri((output / 'mlruns').as_uri())
    mlflow.set_experiment('task1-baseline')
    with mlflow.start_run(run_name='debug-subset' if versions['debug_subset'] else 'baseline') as tracking, stop_requests() as stop:
        if progress['previous_run_id']:
            mlflow.set_tag('resumes_run_id', progress['previous_run_id'])
        progress['previous_run_id'] = tracking.info.run_id
        mlflow.log_params({'alpha': config['loss']['alpha'], 'batch_size': batch_size,
                           'learning_rate': config['training']['learning_rate'],
                           'latent_dim': config['model']['latent_dim'], 'channels': str(config['model']['channels']),
                           'dropout': config['model']['dropout'], 'seed': config['seed'],
                           'train_count': len(train), 'validation_cases': len(val), **{k: v for k, v in versions.items() if v is not None}})
        def should_stop():
            return stop['requested'] or time.monotonic() - started >= args.max_hours * 3600
        def save():
            save_checkpoint(output / 'last.pt', model, optimizer, progress, config, signature, device)
        print(f'Device: {device}; parameters: {versions["parameters"]:,}; train: {len(train)}; val cases: {len(val)}', flush=True)
        if versions['debug_subset']:
            print('DEBUG SUBSET: this run is not final assignment evidence.', flush=True)
        while progress['epoch'] < epochs:
            if should_stop():
                break
            if progress['order'] is None:
                progress['order'] = torch.randperm(len(train)).tolist()
            model.train()
            while progress['cursor'] < len(train):
                if should_stop():
                    break
                cursor = progress['cursor']
                indices = progress['order'][cursor:cursor + batch_size]
                batch = batch_at(train, indices)
                optimizer.zero_grad(set_to_none=True)
                values = criterion(model(batch['input'].to(device)), batch['target'].to(device))
                loss = values['loss'].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss; inspect inputs/model.')
                loss.backward()
                optimizer.step()
                progress['train_sum'] += loss.item() * len(indices)
                progress['train_count'] += len(indices)
                progress['cursor'] += len(indices)
                if (cursor // batch_size + 1) % every == 0:
                    save()
                    print(f'Epoch {progress["epoch"] + 1}: {progress["cursor"]}/{len(train)} images', flush=True)
            if progress['cursor'] < len(train) or should_stop():
                break
            result = validation(model, val, criterion, device, batch_size, should_stop)
            if result is None:
                break  # Restart incomplete validation after resume; no optimizer steps repeated.
            epoch = progress['epoch'] + 1
            metrics = {'train_loss': progress['train_sum'] / progress['train_count'],
                       **{f'val_{k}': v for k, v in result['balanced'].items()}}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError('Non-finite validation metrics.')
            progress['history'].append({'epoch': epoch, **metrics, 'validation': result})
            improved = metrics['val_loss'] < progress['best']
            if improved:
                progress['best'] = metrics['val_loss']
            progress.update(epoch=epoch, order=None, cursor=0, train_sum=0., train_count=0)
            save()
            if improved:
                save_checkpoint(output / 'best.pt', model, optimizer, progress, config, signature, device)
            (output / 'history.json').write_text(json.dumps(progress['history'], indent=2))
            mlflow.log_metrics(metrics, step=epoch)
            for group, row in result['by_condition_severity'].items():
                mlflow.log_metrics({f'val/{group}/{k}': v for k, v in row.items()}, step=epoch)
            preview(model, val, device, output / 'preview.png')
            mlflow.log_artifact(str(output / 'preview.png'), artifact_path=f'epoch_{epoch}')
            print(f'Epoch {epoch}/{epochs}: train={metrics["train_loss"]:.5f}, '
                  f'val={metrics["val_loss"]:.5f}, SSIM={metrics["val_ssim"]:.4f}', flush=True)
        save()
        (output / 'history.json').write_text(json.dumps(progress['history'], indent=2))
        for name in ['config.yaml', 'data_signature.json', 'environment.json', 'history.json']:
            mlflow.log_artifact(str(output / name))
        # Record actual trained weights in the experiment as well as resumable local checkpoints.
        mlflow.log_artifact(str(output / 'last.pt'), artifact_path='checkpoints')
        mlflow.set_tag('completion', 'finished' if progress['epoch'] >= epochs else 'paused')
    print(f'Saved {output / "last.pt"}; completed epochs: {progress["epoch"]}; next batch cursor: {progress["cursor"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/task1.yaml')
    parser.add_argument('--raw-dir', type=Path, default=REPO_ROOT / 'data/raw/oxford_pets')
    parser.add_argument('--splits-dir', type=Path, default=REPO_ROOT / 'data/splits')
    parser.add_argument('--manifests-dir', type=Path, default=REPO_ROOT / 'data/manifests')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts/task1-baseline')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, help='Total target epochs, not additional epochs; keep unchanged on resume.')
    parser.add_argument('--max-hours', type=float, default=4.5, help='Soft per-session budget; allow extra time for saving/uploading.')
    parser.add_argument('--train-limit', type=int, default=0, help='Debug only; first N training images, 0=all.')
    parser.add_argument('--val-images', type=int, default=0, help='Debug only; first N validation images, 0=all.')
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
