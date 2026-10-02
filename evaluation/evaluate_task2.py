"""Task 2 validation analysis: HardRouter evaluation comparing Oracle versus Predicted routing.
Never loads the official test set.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from src.data.corruptions import CLASSES
from src.data.pets_dataset import EvaluationPetsDataset, REPO_ROOT
from src.losses.reconstruction import ReconstructionLoss
from src.models.corruption_classifier import CorruptionClassifier
from src.models.specialist_ae import SpecialistAutoencoder
from src.models.hard_router import HardRouter
from evaluation.classifier_metrics import metrics_from_confusion
from evaluation.evaluate_task1 import digest, write_csv


def verify_checkpoint(saved, kind, split, manifest):
    """Reject mismatched data or inference code; allow trainer-only evolution."""
    sig = saved['data_signature']
    if kind == 'classifier':
        hashes = sig['hashes']
        sources = ['src/models/corruption_classifier.py', 'src/data/pets_dataset.py', 'src/data/corruptions.py']
        files = {'data/splits/pets_val.json': split, 'data/manifests/pets_val_corruptions.json': manifest}
    else:
        hashes = sig['source']
        sources = ['src/models/spatial16_ae.py', 'src/losses/reconstruction.py', 'src/data/pets_dataset.py', 'src/data/corruptions.py']
        if kind != 'task1': sources.append('src/models/specialist_ae.py')
        files = {}
        for path in [split, manifest]:
            if sig['files'].get(path.name) != digest(path):
                raise ValueError(f'{kind}: validation fingerprint mismatch: {path.name}')
    for name in sources:
        if hashes.get(name) != digest(REPO_ROOT / name):
            raise ValueError(f'{kind}: inference source mismatch: {name}')
    for name, path in files.items():
        if hashes.get(name) != digest(path):
            raise ValueError(f'{kind}: validation fingerprint mismatch: {name}')
    return bool(sig['train_limit'] or sig['val_images'])


def load_checkpoint_bundle(classifier_path, salt_path, blur_path, occlusion_path, device, split, manifest):
    """Load classifier and 3 specialist models, verifying their configurations."""
    c_saved = torch.load(classifier_path, map_location='cpu', weights_only=True)
    if c_saved.get('version') != 1:
        raise ValueError('Unsupported classifier checkpoint format.')
    c_debug = verify_checkpoint(c_saved, 'classifier', split, manifest)
    classifier = CorruptionClassifier(**c_saved['config']['model'])
    classifier.load_state_dict(c_saved['model'])

    specialists = {}
    expert_metadata = {}
    for name, path in [('salt', salt_path), ('blur', blur_path), ('occlusion', occlusion_path)]:
        s_saved = torch.load(path, map_location='cpu', weights_only=True)
        if s_saved.get('version') != 1:
            raise ValueError(f'Unsupported specialist checkpoint format for {name}.')
        if s_saved['config'].get('architecture') != 'specialist16_ae_v1':
            raise ValueError(f'Checkpoint for {name} is not specialist16_ae_v1.')
        if s_saved['config'].get('condition') != name:
            raise ValueError(f'Checkpoint at {path} is for condition {s_saved["config"].get("condition")}, expected {name}.')
        debug = verify_checkpoint(s_saved, name, split, manifest)
        expert_metadata[name] = {'path': str(path), 'sha256': digest(path), 'epoch': s_saved['progress']['epoch'], 'debug_subset': debug}
        expert = SpecialistAutoencoder(**s_saved['config']['model'])
        expert.load_state_dict(s_saved['model'])
        specialists[name] = expert

    router = HardRouter(classifier, specialists['salt'], specialists['blur'], specialists['occlusion']).to(device)
    router.eval()
    metadata = {'classifier': {'path': str(classifier_path), 'epoch': c_saved['progress']['epoch'],
                'sha256': digest(classifier_path), 'debug_subset': c_debug}, **expert_metadata}
    return router, c_saved, metadata


def summarize_task2(rows, confusion_matrix, cross_entropy):
    """Compute condition, severity, balanced metrics and routing failure taxonomy."""
    metrics = [
        'input_l1', 'input_ssim', 'input_loss',
        'oracle_l1', 'oracle_ssim', 'oracle_loss', 'oracle_l1_gain', 'oracle_ssim_gain', 'oracle_loss_gain',
        'pred_l1', 'pred_ssim', 'pred_loss', 'pred_l1_gain', 'pred_ssim_gain', 'pred_loss_gain',
        'ssim_routing_cost', 'l1_routing_cost'
    ]

    if rows and 'task1_l1' in rows[0]:
        metrics += ['task1_l1', 'task1_ssim', 'task1_loss']

    def aggregate(group):
        count = len(group)
        correct = sum(r['correct_routing'] for r in group)
        base = {
            'count': count,
            'routing_accuracy': correct / count if count else 0.0,
            **{k: sum(r[k] for r in group) / count for k in metrics},
            'oracle_ssim_worse_count': sum(r['oracle_ssim_gain'] < -1e-6 for r in group),
            'pred_ssim_worse_count': sum(r['pred_ssim_gain'] < -1e-6 for r in group),
            'oracle_l1_worse_count': sum(r['oracle_l1_gain'] < -1e-6 for r in group),
            'pred_l1_worse_count': sum(r['pred_l1_gain'] < -1e-6 for r in group),
        }
        return base

    severity = []
    conditions = []
    for condition in CLASSES:
        group = [r for r in rows if r['true_condition'] == condition]
        if not group:
            raise ValueError(f'Analysis requires all four conditions, missing {condition}.')
        conditions.append({'condition': condition, **aggregate(group)})
        for level in ['clean', 'low', 'medium', 'high']:
            subset = [r for r in group if r['severity'] == level]
            if subset:
                severity.append({'condition': condition, 'severity': level, **aggregate(subset)})

    balanced = {key: sum(r[key] for r in conditions) / 4 for key in ['routing_accuracy'] + metrics}

    # Taxonomy of routing failures
    misrouted = [r for r in rows if not r['correct_routing']]
    clean_as_corrupted = [r for r in rows if r['true_condition'] == 'clean' and not r['correct_routing']]
    corrupted_as_clean = [r for r in rows if r['true_condition'] != 'clean' and r['pred_condition'] == 'clean']
    cross_corruption = [r for r in rows if r['true_condition'] != 'clean' and r['pred_condition'] != 'clean' and not r['correct_routing']]
    clean_preserved = [r for r in rows if r['true_condition'] == 'clean' and r['correct_routing']]

    taxonomy = {
        'total_cases': len(rows),
        'correct_routing_count': len(rows) - len(misrouted),
        'misrouted_count': len(misrouted),
        'overall_accuracy': (len(rows) - len(misrouted)) / len(rows) if rows else 0.0,
        'clean_identity_preserved_count': len(clean_preserved),
        'clean_as_corrupted_count': len(clean_as_corrupted),
        'corrupted_as_clean_count': len(corrupted_as_clean),
        'cross_corruption_count': len(cross_corruption),
        'mean_ssim_loss_when_misrouted': (
            sum(r['ssim_routing_cost'] for r in misrouted) / len(misrouted) if misrouted else 0.0
        ),
        'mean_clean_degradation_l1': (
            sum(r['pred_l1'] for r in clean_as_corrupted) / len(clean_as_corrupted) if clean_as_corrupted else 0.0
        ),
        'mean_clean_degradation_ssim': (
            sum(1.0 - r['pred_ssim'] for r in clean_as_corrupted) / len(clean_as_corrupted) if clean_as_corrupted else 0.0
        ),
    }

    clf_metrics = {**metrics_from_confusion(confusion_matrix), 'cross_entropy': cross_entropy, 'class_order': list(CLASSES)}

    return {
        'classifier_metrics': clf_metrics,
        'by_condition': conditions,
        'by_condition_severity': severity,
        'condition_balanced': balanced,
        'routing_taxonomy': taxonomy
    }


def select_examples_task2(rows):
    """Select 12 representatives (3 per condition) and distinct routing failure modes."""
    representatives = []
    for condition in CLASSES:
        group = [r for r in rows if r['true_condition'] == condition]
        seen = set()
        levels = ['clean'] * 3 if condition == 'clean' else ['low', 'medium', 'high']
        for level in levels:
            candidates = [r for r in group if r['severity'] == level and r['image_id'] not in seen]
            if candidates:
                chosen = candidates[0]
                representatives.append(chosen)
                seen.add(chosen['image_id'])

    failures = []
    seen = set()

    # Priority 1: Clean misclassified as corruption (clean detail degradation)
    for r in sorted([r for r in rows if r['true_condition'] == 'clean' and not r['correct_routing']],
                    key=lambda x: (x['pred_ssim'], x['index'])):
        if r['image_id'] not in seen:
            failures.append(r)
            seen.add(r['image_id'])
            if len([x for x in failures if x['true_condition'] == 'clean']) >= 2:
                break

    # Priority 2: Corrupted misclassified as clean (bypass failure)
    for r in sorted([r for r in rows if r['true_condition'] != 'clean' and r['pred_condition'] == 'clean'],
                    key=lambda x: (-x['ssim_routing_cost'], x['index'])):
        if r['image_id'] not in seen:
            failures.append(r)
            seen.add(r['image_id'])
            if len([x for x in failures if x['pred_condition'] == 'clean']) >= 2:
                break

    # Priority 3: Cross-corruption misrouting
    for r in sorted([r for r in rows if r['true_condition'] != 'clean' and r['pred_condition'] != 'clean' and not r['correct_routing']],
                    key=lambda x: (-x['ssim_routing_cost'], x['index'])):
        if r['image_id'] not in seen:
            failures.append(r)
            seen.add(r['image_id'])
            if len(failures) >= 6:
                break

    # Only show actual misroutes; do not label correct-routing restoration errors as routing failures
    return representatives, failures


def image_grid_task2(router, dataset, rows, device, destination):
    """Render 5-column visual comparison: Target, Input, Oracle, Predicted, |Pred - Target|."""
    if not rows:
        return
    col_width = 160
    row_height = 176
    canvas = Image.new('RGB', (col_width * 5 + 32, len(rows) * row_height), 'white')
    draw = ImageDraw.Draw(canvas)

    with torch.inference_mode():
        for i, row in enumerate(rows):
            sample = dataset[row['index']]
            img_tensor = sample['input'][None].to(device)
            target_tensor = sample['target'][None].to(device)
            lbl_tensor = torch.tensor([sample['label']], dtype=torch.long, device=device)

            pred_res = router(img_tensor)
            oracle_res = router(img_tensor, labels=lbl_tensor)

            pred_out = pred_res['output'][0].cpu()
            oracle_out = oracle_res['output'][0].cpu()
            target_out = target_tensor[0].cpu()
            input_out = img_tensor[0].cpu()
            diff_out = (pred_out - target_out).abs()

            status = "CORRECT" if row['correct_routing'] else f"MISROUTED (True:{row['true_condition']}->Pred:{row['pred_condition']})"
            title = (f"[{status}] {row['image_id']} ({row['severity']}) | "
                     f"Oracle SSIM:{row['oracle_ssim']:.3f} | Pred SSIM:{row['pred_ssim']:.3f} (Cost:{row['ssim_routing_cost']:+.3f})")
            draw.text((8, i * row_height + 4), title, fill='black')

            cols = [
                ('Clean target', target_out),
                ('Input', input_out),
                ('Oracle output', oracle_out),
                ('Predicted output', pred_out),
                ('|Pred - Target|', diff_out),
            ]
            for j, (name, tensor) in enumerate(cols):
                draw.text((j * col_width + 16, i * row_height + 20), name, fill='black')
                pixels = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                canvas.paste(Image.fromarray(pixels), (j * col_width + 16, i * row_height + 36))

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def run(args):
    torch.set_num_threads(args.cpu_threads)
    output = args.output_dir or REPO_ROOT / 'artifacts/task2_validation_analysis'
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory {output} is not empty; specify a clean --output-dir.')

    # Determine checkpoint paths
    c_path = args.classifier_checkpoint
    s_path = args.salt_checkpoint
    b_path = args.blur_checkpoint
    o_path = args.occlusion_checkpoint

    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        ckpts = cfg.get('checkpoints', {})
        c_path = c_path or (Path(ckpts['classifier']) if 'classifier' in ckpts else None)
        s_path = s_path or (Path(ckpts['salt']) if 'salt' in ckpts else None)
        b_path = b_path or (Path(ckpts['blur']) if 'blur' in ckpts else None)
        o_path = o_path or (Path(ckpts['occlusion']) if 'occlusion' in ckpts else None)

    for name, p in [('classifier', c_path), ('salt', s_path), ('blur', b_path), ('occlusion', o_path)]:
        if p is None or not Path(p).is_file():
            raise FileNotFoundError(f'Missing checkpoint for {name}: {p}')

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')

    split = args.splits_dir / 'pets_val.json'
    manifest = args.manifests_dir / 'pets_val_corruptions.json'
    if not split.exists() or not manifest.exists():
        raise FileNotFoundError('Validation split or corruptions manifest not found.')

    # Provenance source checks
    source_files = [
        'src/models/hard_router.py', 'src/models/corruption_classifier.py',
        'src/models/specialist_ae.py', 'src/losses/reconstruction.py',
        'src/data/pets_dataset.py', 'src/data/corruptions.py',
        'evaluation/classifier_metrics.py'
    ]
    source_digests = {f: digest(REPO_ROOT / f) for f in source_files}

    router, classifier_saved, ckpt_provenance = load_checkpoint_bundle(
        c_path, s_path, b_path, o_path, device, split, manifest
    )

    universal = None
    if args.task1_checkpoint:
        from src.models.spatial16_ae import Spatial16Autoencoder
        saved = torch.load(args.task1_checkpoint, map_location='cpu', weights_only=True)
        if saved.get('version') != 1 or saved['config'].get('architecture') != 'spatial16_ae_v1':
            raise ValueError('Expected Task1 spatial16 checkpoint')
        debug = verify_checkpoint(saved, 'task1', split, manifest)
        universal = Spatial16Autoencoder(**saved['config']['model']).to(device)
        universal.load_state_dict(saved['model']); universal.eval()
        ckpt_provenance['task1'] = {'sha256': digest(args.task1_checkpoint), 'epoch': saved['progress']['epoch'], 'debug_subset': debug}

    alpha = args.alpha
    criterion = ReconstructionLoss(alpha=alpha).to(device)

    dataset = EvaluationPetsDataset(split, manifest, args.raw_dir)
    if dataset.manifest['split'] != 'val':
        raise ValueError('This evaluator is strictly for validation splits.')
    if args.val_images:
        dataset = Subset(dataset, range(min(len(dataset), args.val_images * 10)))

    rows = []
    confusion_matrix = torch.zeros(4, 4, dtype=torch.int64)
    total_ce = 0.0

    print(f'Starting Task 2 HardRouter evaluation on {len(dataset)} cases using device {device}...', flush=True)

    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0):
            images = batch['input'].to(device)
            targets = batch['target'].to(device)
            true_labels = batch['label'].to(device)

            pred_res = router(images)
            pred_outputs = pred_res['output']
            pred_labels = pred_res['labels']
            pred_probs = pred_res['probabilities']

            oracle_res = router(images, labels=true_labels)
            oracle_outputs = oracle_res['output']

            # Cross entropy on classifier
            logits = router.classifier(images)
            ce = torch.nn.functional.cross_entropy(logits, true_labels, reduction='sum').item()
            total_ce += ce

            confusion_matrix += torch.bincount((true_labels * 4 + pred_labels).cpu(), minlength=16).reshape(4, 4)

            inputs_rec = {k: v.cpu().tolist() for k, v in criterion(images, targets).items()}
            oracle_rec = {k: v.cpu().tolist() for k, v in criterion(oracle_outputs, targets).items()}
            pred_rec = {k: v.cpu().tolist() for k, v in criterion(pred_outputs, targets).items()}

            task1_rec = {k: v.cpu().tolist() for k,v in criterion(universal(images), targets).items()} if universal is not None else None
            for i in range(len(images)):
                t_lbl = true_labels[i].item()
                p_lbl = pred_labels[i].item()
                true_cond = CLASSES[t_lbl]
                pred_cond = CLASSES[p_lbl]
                correct = (t_lbl == p_lbl)

                inp_l1 = inputs_rec['l1'][i]
                inp_ssim = inputs_rec['ssim'][i]
                inp_loss = inputs_rec['loss'][i]

                orc_l1 = oracle_rec['l1'][i]
                orc_ssim = oracle_rec['ssim'][i]
                orc_loss = oracle_rec['loss'][i]

                prd_l1 = pred_rec['l1'][i]
                prd_ssim = pred_rec['ssim'][i]
                prd_loss = pred_rec['loss'][i]

                row = {
                    'index': len(rows),
                    'image_id': batch['image_id'][i],
                    'path': batch['path'][i],
                    'true_condition': true_cond,
                    'severity': batch['severity'][i],
                    'true_label': t_lbl,
                    'pred_condition': pred_cond,
                    'pred_label': p_lbl,
                    'pred_confidence': pred_probs[i, p_lbl].item() if pred_probs is not None else 1.0,
                    'correct_routing': correct,
                    'input_l1': inp_l1,
                    'input_ssim': inp_ssim,
                    'input_loss': inp_loss,
                    'oracle_l1': orc_l1,
                    'oracle_ssim': orc_ssim,
                    'oracle_loss': orc_loss,
                    'oracle_l1_gain': inp_l1 - orc_l1,
                    'oracle_ssim_gain': orc_ssim - inp_ssim,
                    'oracle_loss_gain': inp_loss - orc_loss,
                    'pred_l1': prd_l1,
                    'pred_ssim': prd_ssim,
                    'pred_loss': prd_loss,
                    'pred_l1_gain': inp_l1 - prd_l1,
                    'pred_ssim_gain': prd_ssim - inp_ssim,
                    'pred_loss_gain': inp_loss - prd_loss,
                    'l1_routing_cost': prd_l1 - orc_l1,
                    'ssim_routing_cost': orc_ssim - prd_ssim,
                }
                for label, name in enumerate(CLASSES):
                    row['prob_' + name] = pred_probs[i, label].item()
                if task1_rec is not None:
                    row.update({f'task1_{k}': task1_rec[k][i] for k in ['l1','ssim','loss']})
                if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise ValueError('Non-finite metrics encountered during evaluation.')
                rows.append(row)

            if len(rows) % (args.batch_size * 25) == 0:
                print(f'Processed {len(rows)}/{len(dataset)} validation cases', flush=True)

    summary = summarize_task2(rows, confusion_matrix, total_ce / len(rows) if rows else 0.0)
    summary['provenance'] = {
        'task': 2,
        'architecture': 'HardRouter_with_3_Specialists',
        'alpha': alpha,
        'checkpoints': ckpt_provenance,
        'torch_version': str(torch.__version__),
        'device': str(device),
        'split': 'validation',
        'case_count': len(rows),
        'analysis_is_subset': bool(args.val_images),
        'split_sha256': digest(split),
        'manifest_sha256': digest(manifest),
        'source_hashes': source_digests
    }

    representatives, failures = select_examples_task2(rows)
    summary['example_selection'] = {
        'representative_count': len(representatives),
        'routing_failure_count': len(failures),
        'representatives': representatives,
        'routing_failures': failures
    }

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_image.csv', rows)
    write_csv(output / 'by_condition.csv', summary['by_condition'])
    write_csv(output / 'by_condition_severity.csv', summary['by_condition_severity'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'confusion_matrix.json').write_text(json.dumps(summary['classifier_metrics'], indent=2) + '\n')

    image_grid_task2(router, dataset, representatives, device, output / 'representative_grid.png')
    image_grid_task2(router, dataset, failures, device, output / 'routing_failures.png')

    # Generate Markdown Report
    report = [
        '# Task 2 Hard Routing Validation Analysis',
        '',
        f"Evaluated {len(rows)} validation cases. Device: {device}.",
        f"Reconstruction loss alpha: {alpha}.",
        '',
        '## 1. Classification & Confusion Matrix',
        f"- Overall Classification Accuracy: **{summary['classifier_metrics']['accuracy']*100:.2f}%**",
        f"- Macro F1 Score: **{summary['classifier_metrics']['macro_f1']:.4f}**",
        f"- Cross Entropy: **{summary['classifier_metrics']['cross_entropy']:.4f}**",
        '',
        '| True \\ Pred | Clean | Salt | Blur | Occlusion | Recall | F1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    cm = summary['classifier_metrics']['confusion']
    for idx, cname in enumerate(CLASSES):
        rec = summary['classifier_metrics']['recall'][idx]
        f1 = summary['classifier_metrics']['f1'][idx]
        report.append(f"| **{cname}** | {cm[idx][0]} | {cm[idx][1]} | {cm[idx][2]} | {cm[idx][3]} | {rec:.4f} | {f1:.4f} |")

    report += [
        '',
        '## 2. Condition-Level Comparison: Oracle vs Predicted Routing',
        '',
        'Positive gain indicates improvement over corrupted input (input - output for L1; output - input for SSIM).',
        'Routing cost is (Oracle SSIM - Pred SSIM); positive cost reflects restoration forfeited or degradation caused by routing errors.',
        '',
        '| Condition | Input SSIM | Oracle SSIM | Oracle Gain | Pred SSIM | Pred Gain | SSIM Routing Cost | Accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for row in summary['by_condition']:
        report.append(
            f"| {row['condition']} | {row['input_ssim']:.4f} | {row['oracle_ssim']:.4f} | {row['oracle_ssim_gain']:+.4f} | "
            f"{row['pred_ssim']:.4f} | {row['pred_ssim_gain']:+.4f} | {row['ssim_routing_cost']:.4f} | {row['routing_accuracy']*100:.1f}% |"
        )

    b = summary['condition_balanced']
    report.append(
        f"| **Balanced Average** | **{b['input_ssim']:.4f}** | **{b['oracle_ssim']:.4f}** | **{b['oracle_ssim_gain']:+.4f}** | "
        f"**{b['pred_ssim']:.4f}** | **{b['pred_ssim_gain']:+.4f}** | **{b['ssim_routing_cost']:.4f}** | **{b['routing_accuracy']*100:.1f}%** |"
    )

    tax = summary['routing_taxonomy']
    report += [
        '',
        '## 3. Routing Error Taxonomy',
        f"- Clean identity preserved: {tax['clean_identity_preserved_count']} cases",
        f"- Clean corrupted by false expert trigger: {tax['clean_as_corrupted_count']} cases",
        f"- Corruption bypassed due to false clean prediction: {tax['corrupted_as_clean_count']} cases",
        f"- Cross-corruption mismatch (wrong expert triggered): {tax['cross_corruption_count']} cases",
        f"- Mean SSIM degradation when misrouted: {tax['mean_ssim_loss_when_misrouted']:.4f}",
        '',
        '## 4. Visual Evidence',
        f"- Representative panels: `representative_grid.png` ({len(representatives)} cases)",
        f"- Routing failure panels: `routing_failures.png` ({len(failures)} cases)",
        '',
        'Official test images were not evaluated; analysis restricted to validation split.'
    ]
    if args.val_images or any(v['debug_subset'] for v in ckpt_provenance.values()):
        report.insert(2, '**DEBUG SUBSET CHECK: not final performance evidence.**\n')

    report += ['', '## Fixed-weight overall comparison', '', '| Method | L1 | SSIM | Loss |', '|---|---:|---:|---:|']
    for method in ['input', 'oracle', 'pred'] + (['task1'] if universal is not None else []):
        report.append(f"| {method} | {b[method+'_l1']:.6f} | {b[method+'_ssim']:.6f} | {b[method+'_loss']:.6f} |")
    report += ['', 'All methods use the same evaluation alpha, regardless of training alpha. Negative routing cost means predicted routing scored better than oracle; oracle is not a guaranteed metric upper bound. Error panels use fixed [0,1] scale.']
    (output / 'README.md').write_text('\n'.join(report) + '\n')
    print('\n'.join(report), flush=True)
    print(f'Task 2 validation analysis saved to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=None, help='Optional task2 YAML configuration')
    parser.add_argument('--task1-checkpoint', type=Path, help='Optional selected Task1 spatial16 best.pt for same-case comparison')
    parser.add_argument('--classifier-checkpoint', type=Path, default=None)
    parser.add_argument('--salt-checkpoint', type=Path, default=None)
    parser.add_argument('--blur-checkpoint', type=Path, default=None)
    parser.add_argument('--occlusion-checkpoint', type=Path, default=None)
    parser.add_argument('--raw-dir', type=Path, default=REPO_ROOT / 'data/raw/oxford_pets')
    parser.add_argument('--splits-dir', type=Path, default=REPO_ROOT / 'data/splits')
    parser.add_argument('--manifests-dir', type=Path, default=REPO_ROOT / 'data/manifests')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--val-images', type=int, default=0, help='Debug only; 0 evaluates full validation split.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Fixed comparison (not training) reconstruction loss alpha balance.')
    args = parser.parse_args()

    if args.batch_size < 1 or args.cpu_threads < 1 or args.val_images < 0:
        parser.error('Batch size and cpu-threads must be positive; val-images must be nonnegative.')
    run(args)


if __name__ == '__main__':
    main()
