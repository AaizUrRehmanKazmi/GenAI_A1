"""Sequential Optuna screening for Task 4 style GAN with atomic JSON persistence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import optuna
import torch
import yaml

from src.data.pets_dataset import REPO_ROOT
from training.train_task4_gan import run as train_gan
from exports.task2_bundle import digest


def score(row):
    """Consistent ranking objective across trials with different training alphas."""
    return 0.8 * float(row['val_l1']) + 0.2 * (1.0 - float(row['val_ssim']))


def distributions():
    return {
        'generator_lr': optuna.distributions.FloatDistribution(5e-5, 4e-4, log=True),
        'discriminator_lr': optuna.distributions.FloatDistribution(2e-5, 2e-4, log=True),
        'batch_size': optuna.distributions.CategoricalDistribution([4, 8]),
        'base_channels': optuna.distributions.CategoricalDistribution([32, 48]),
        'dropout': optuna.distributions.CategoricalDistribution([0., .2, .4]),
        'style_dim': optuna.distributions.CategoricalDistribution([8, 16]),
        'l1_weight': optuna.distributions.CategoricalDistribution([50., 100., 150.]),
    }


def baseline_params(base):
    return {**{k: base['training'][k] for k in ('generator_lr','discriminator_lr','batch_size')},
            **base['generator'], 'l1_weight': base['loss']['l1_weight']}


def trial_config(base, params):
    config = copy.deepcopy(base)
    config['training'].update({k: params[k] for k in ('generator_lr','discriminator_lr','batch_size')})
    config['generator'].update({k: params[k] for k in ('base_channels','style_dim','dropout')})
    config['discriminator'].update({k: params[k] for k in ('base_channels','style_dim')})
    config['loss']['l1_weight'] = params['l1_weight']
    return config


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(args):
    started = time.monotonic()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.config.read_text())

    files = ['optimization/optimize_task4.py', 'training/train_task4_gan.py',
        'training/train_task1.py', 'src/models/generator_unet.py',
        'src/models/discriminator_patchgan.py', 'src/losses/gan_loss.py',
        'src/losses/reconstruction.py', 'src/data/fs2k_dataset.py']
    rows = sum([json.loads((args.splits_dir / f'{split}.json').read_text())
                for split in ('train','val')], [])
    protocol = {'base': base, 'screen_epochs': args.screen_epochs,
        'train_limit': args.train_limit, 'val_per_style': args.val_per_style,
        'optuna': optuna.__version__,
        'objective': '0.8*equal-style L1 + 0.2*(1-equal-style SSIM) at final screening epoch',
        'hashes': {f: digest(REPO_ROOT / f) for f in files},
        'splits': {n: digest(args.splits_dir / n) for n in ('train.json','val.json','metadata.json')},
        'images': {r[k]: digest(args.raw_dir / r[k]) for r in rows for k in ('photo','sketch')}}

    state_path = root / 'study.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'protocol': protocol, 'trials': []}
    if state['protocol'] != protocol:
        raise ValueError('Search protocol or source code changed; choose a new output directory.')
    write_json(state_path, state)

    while len([t for t in state['trials'] if t['state'] == 'COMPLETE']) < args.trials:
        remaining = args.max_hours * 3600 - (time.monotonic() - started)
        if remaining < 60:
            break

        pending = next((t for t in state['trials'] if t['state'] == 'RUNNING'), None)
        if pending is None:
            number = len(state['trials'])
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42 + number, n_startup_trials=4))
            for t in state['trials']:
                study.add_trial(optuna.trial.create_trial(params=t['params'], distributions=distributions(), value=t['score']))
            trial = study.ask(distributions())
            params = trial.params
            if number == 0:
                params = baseline_params(base)
            pending = {'number': number, 'state': 'RUNNING', 'params': params}
            state['trials'].append(pending)
            write_json(state_path, state)

        number = pending['number']
        params = pending['params']
        config = trial_config(base, params)

        config_path = root / f'trial_{number:03d}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        out = root / f'trial_{number:03d}'
        last = out / 'last.pt'

        if out.exists() and not last.exists() and any(out.iterdir()):
            raise ValueError(f'Interrupted before first checkpoint: preserve/rename {out} then rerun.')

        print(f'Trial {number+1}/{args.trials}: {params}', flush=True)
        train_gan(SimpleNamespace(config=config_path, raw_dir=args.raw_dir, splits_dir=args.splits_dir,
            output_dir=out, resume=last if last.exists() else None,
            stop_after_epoch=args.screen_epochs, max_steps=0, device=args.device,
            max_hours=remaining / 3600, train_limit=args.train_limit, val_per_style=args.val_per_style))

        history_file = out / 'history.json'
        if not history_file.exists():
            break
        history = json.loads(history_file.read_text())
        if not history or history[-1]['epoch'] < args.screen_epochs:
            break

        row = history[-1]
        pending.update(state='COMPLETE', score=score(row), metrics=row, checkpoint=str(last))
        write_json(state_path, state)

        completed = [t for t in state['trials'] if t['state'] == 'COMPLETE']
        winner = min(completed, key=lambda t: t['score'])
        best_config = yaml.safe_load((root / f"trial_{winner['number']:03d}.yaml").read_text())
        (root / 'selected_config.yaml').write_text(yaml.safe_dump(best_config))
        write_json(root / 'leaderboard.json', {
            'provisional': True,
            'task': 4,
            'debug_subset': bool(args.train_limit or args.val_per_style),
            'ranked_trials': sorted(completed, key=lambda t: t['score'])
        })

    print(f'Search state saved to {state_path}. Rerun to continue; trials is a total target.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir', type=Path, required=True)
    p.add_argument('--splits-dir', type=Path, default=REPO_ROOT / 'data/splits/fs2k')
    p.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/task4.yaml')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--screen-epochs', type=int, default=5)
    p.add_argument('--max-hours', type=float, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--val-per-style', type=int, default=0)
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    if min(a.trials, a.screen_epochs, a.max_hours) <= 0 or min(a.train_limit, a.val_per_style) < 0 or not 1 <= a.screen_epochs <= config['training']['epochs']:
        p.error('Invalid trial, epoch, time or subset limits.')
    run(a)


if __name__ == '__main__':
    main()
