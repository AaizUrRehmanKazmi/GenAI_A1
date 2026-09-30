"""Separate spatial-AE experiment; legacy trainer and checkpoint fingerprints remain intact."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import mlflow
import torch
from torch.utils.data import Subset
import yaml
from src.data.pets_dataset import REPO_ROOT, TrainingPetsDataset, EvaluationPetsDataset
from src.models.spatial_ae import SpatialAutoencoder
from src.losses.reconstruction import ReconstructionLoss
from training.train_task1 import (save_checkpoint, restore_checkpoint, stop_requests,
                                  batch_at, validation, preview)


def run(args):
    started = time.monotonic()
    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if config.get('architecture') != 'spatial_ae_v1':
        raise ValueError('This trainer requires architecture: spatial_ae_v1.')
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
                       'src/models/spatial_ae.py', 'src/losses/reconstruction.py', 'training/train_task1.py', 'training/train_task1_spatial.py']]},
    }
    model = SpatialAutoencoder(**config['model']).to(device)
    criterion = ReconstructionLoss(**config['loss']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.resume is None and any(output.iterdir()):
        raise ValueError('Output is not empty. Use --resume or a new --output-dir.')
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
    mlflow.set_experiment('task1-spatial')
    with mlflow.start_run(run_name='debug-subset' if versions['debug_subset'] else 'spatial') as tracking, stop_requests() as stop:
        if progress['previous_run_id']:
            mlflow.set_tag('resumes_run_id', progress['previous_run_id'])
        progress['previous_run_id'] = tracking.info.run_id
        mlflow.log_params({'alpha': config['loss']['alpha'], 'batch_size': batch_size,
                           'learning_rate': config['training']['learning_rate'],
                           'latent_channels': config['model']['latent_channels'], 'latent_values': config['model']['latent_channels'] * 64, 'channels': str(config['model']['channels']),
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
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/task1_spatial.yaml')
    parser.add_argument('--raw-dir', type=Path, default=REPO_ROOT / 'data/raw/oxford_pets')
    parser.add_argument('--splits-dir', type=Path, default=REPO_ROOT / 'data/splits')
    parser.add_argument('--manifests-dir', type=Path, default=REPO_ROOT / 'data/manifests')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts/task1-spatial')
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
