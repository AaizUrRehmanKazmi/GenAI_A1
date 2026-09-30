"""Validation-only input-versus-restored analysis. Never loads the official test set."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data.corruptions import CLASSES
from src.data.pets_dataset import EvaluationPetsDataset, REPO_ROOT
from src.losses.reconstruction import ReconstructionLoss
from src.models.spatial16_ae import Spatial16Autoencoder


from evaluation.evaluate_task1 import digest, summarize, write_csv, select_examples, image_grid


def run(args):
    torch.set_num_threads(args.cpu_threads)
    output = args.output_dir or args.checkpoint.parent / 'validation_analysis'
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --output-dir to preserve previous analysis.')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('version') != 1:
        raise ValueError('Unsupported checkpoint format.')
    if saved['config'].get('architecture') != 'spatial16_ae_v1':
        raise ValueError('This evaluator requires a spatial16_ae_v1 checkpoint.')
    split = args.splits_dir / 'pets_val.json'
    manifest = args.manifests_dir / 'pets_val_corruptions.json'
    signature = saved['data_signature']
    for path in [split, manifest]:
        if signature['files'].get(path.name) != digest(path):
            raise ValueError(f'Validation file differs from training provenance: {path.name}')
    # Trainer source may evolve for analysis, but model, loss and preprocessing must match.
    for name in ['src/models/spatial16_ae.py', 'src/losses/reconstruction.py',
                 'src/data/pets_dataset.py', 'src/data/corruptions.py']:
        if signature['source'].get(name) != digest(REPO_ROOT / name):
            raise ValueError(f'Inference source differs from checkpoint: {name}')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')
    model = Spatial16Autoencoder(**saved['config']['model']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    criterion = ReconstructionLoss(**saved['config']['loss']).to(device)
    dataset = EvaluationPetsDataset(split, manifest, args.raw_dir)
    if dataset.manifest['split'] != 'val':
        raise ValueError('This tool is for validation only.')
    if args.val_images:
        dataset = Subset(dataset, range(min(len(dataset), args.val_images * 10)))
    rows = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0):
            image, target = batch['input'].to(device), batch['target'].to(device)
            prediction = model(image)
            inputs = {k: v.cpu().tolist() for k, v in criterion(image, target).items()}
            outputs = {k: v.cpu().tolist() for k, v in criterion(prediction, target).items()}
            for i in range(len(image)):
                row = {'index': len(rows), 'image_id': batch['image_id'][i], 'path': batch['path'][i],
                       'condition': CLASSES[batch['label'][i].item()], 'severity': batch['severity'][i]}
                for metric in ['l1', 'ssim', 'loss']:
                    row[f'input_{metric}'] = inputs[metric][i]
                    row[f'output_{metric}'] = outputs[metric][i]
                    row[f'{metric}_gain'] = (outputs[metric][i] - inputs[metric][i]) * (1 if metric == 'ssim' else -1)
                if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise ValueError('Non-finite metrics encountered.')
                rows.append(row)
            if len(rows) % (args.batch_size * 25) == 0:
                print(f'Analyzed {len(rows)}/{len(dataset)} validation cases', flush=True)
    summary = summarize(rows)
    summary['provenance'] = {'checkpoint': args.checkpoint.name, 'checkpoint_sha256': digest(args.checkpoint),
        'checkpoint_completed_epochs': saved['progress']['epoch'], 'alpha': criterion.alpha,
        'torch_version': str(torch.__version__), 'device': str(device), 'split': 'validation',
        'case_count': len(rows), 'analysis_is_subset': bool(args.val_images),
        'checkpoint_was_debug_subset': bool(signature['train_limit'] or signature['val_images']),
        'validation_split_sha256': digest(split), 'validation_manifest_sha256': digest(manifest)}
    representatives, failures = select_examples(rows)
    summary['example_selection'] = {'representatives': representatives, 'worst_ssim_regressions': failures}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_image.csv', rows)
    write_csv(output / 'by_condition.csv', summary['by_condition'])
    write_csv(output / 'by_condition_severity.csv', summary['by_condition_severity'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    image_grid(model, dataset, representatives, device, output / 'representative_grid.png')
    image_grid(model, dataset, failures, device, output / 'ssim_regressions.png')
    report = ['# Task 1 validation analysis', '',
              f"Checkpoint completed epoch: {saved['progress']['epoch']}. Cases: {len(rows)}.", '',
              'Positive gain means improvement: input L1 minus output L1; output SSIM minus input SSIM.', '',
              '| Condition | Input L1 | Output L1 | L1 gain | Input SSIM | Output SSIM | SSIM gain |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['by_condition']:
        report.append('| ' + row['condition'] + ' | ' + ' | '.join(f'{row[k]:.5f}' for k in
                      ['input_l1','output_l1','l1_gain','input_ssim','output_ssim','ssim_gain']) + ' |')
    report += ['', 'The overall summary equally weights four conditions. Clean input is an identity baseline: '
               'any restoration error there is degradation. Small metric differences are not statistical-significance claims.', '',
               f'Selected {len(representatives)} representative cases and {len(failures)} unique-image SSIM regressions. '
               'Regression selection excludes clean cases; these are candidates for interpretation, not explanations of cause.', '',
               'Absolute error panels use a fixed [0,1] display scale, without per-image contrast stretching.', '',
               'No official test images were evaluated.']
    if args.val_images or summary['provenance']['checkpoint_was_debug_subset']:
        report.insert(2, '**DEBUG SUBSET CHECK: not final performance evidence.**\n')
    (output / 'README.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report), flush=True)
    print(f'Analysis saved to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, default=REPO_ROOT / 'data/raw/oxford_pets')
    parser.add_argument('--splits-dir', type=Path, default=REPO_ROOT / 'data/splits')
    parser.add_argument('--manifests-dir', type=Path, default=REPO_ROOT / 'data/manifests')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--val-images', type=int, default=0, help='Debug only; 0 evaluates the full validation split.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or args.val_images < 0:
        parser.error('Batch size/threads must be positive; val-images must be nonnegative.')
    run(args)


if __name__ == '__main__':
    main()
