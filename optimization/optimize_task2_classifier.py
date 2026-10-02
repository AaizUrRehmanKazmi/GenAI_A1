"""Sequential Optuna screening for Task 2 corruption classifier with atomic JSON persistence."""
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
from training.train_classifier import run as train_classifier_run


def score(row):
    """Minimize 1 - macro_f1 to maximize balanced classification across all 4 conditions."""
    return 1.0 - float(row['macro_f1'])


def distributions():
    return {
        'learning_rate': optuna.distributions.FloatDistribution(0.0003, 0.003, log=True),
        'batch_size': optuna.distributions.CategoricalDistribution([16, 32, 64]),
        'dropout': optuna.distributions.CategoricalDistribution([0.0, 0.1, 0.2]),
        'weight_decay': optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
        'channels_preset': optuna.distributions.CategoricalDistribution(['small', 'standard', 'wide']),
    }


CHANNELS_MAP = {
    'small': [16, 32, 32],
    'standard': [16, 32, 64],
    'wide': [32, 64, 128],
}


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(args):
    started = time.monotonic()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.config.read_text())

    files = [
        'optimization/optimize_task2_classifier.py', 'training/train_classifier.py',
        'src/models/corruption_classifier.py', 'src/data/classifier_batches.py',
        'src/data/pets_dataset.py', 'src/data/corruptions.py',
        'evaluation/classifier_metrics.py',
        'data/splits/pets_train.json', 'data/splits/pets_val.json',
        'data/manifests/pets_val_corruptions.json'
    ]
    protocol = {
        'base': base, 'screen_epochs': args.screen_epochs, 'train_limit': args.train_limit,
        'val_images': args.val_images, 'optuna': optuna.__version__,
        'objective': '1 - macro_f1 at final screening epoch',
        'hashes': {f: hashlib.sha256((REPO_ROOT / f).read_bytes()).hexdigest() for f in files}
    }

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
                params = dict(learning_rate=0.001, batch_size=32, dropout=0.1, weight_decay=0.0001, channels_preset='standard')
            pending = {'number': number, 'state': 'RUNNING', 'params': params}
            state['trials'].append(pending)
            write_json(state_path, state)

        number = pending['number']
        params = pending['params']
        config = copy.deepcopy(base)
        config['model']['channels'] = CHANNELS_MAP[params['channels_preset']]
        config['model']['dropout'] = params['dropout']
        config['training']['learning_rate'] = params['learning_rate']
        config['training']['batch_size'] = params['batch_size']
        config['training']['weight_decay'] = params['weight_decay']
        config['training']['epochs'] = args.screen_epochs

        config_path = root / f'trial_{number:03d}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        out = root / f'trial_{number:03d}'
        last = out / 'last.pt'

        if out.exists() and not last.exists() and any(out.iterdir()):
            raise ValueError(f'Interrupted before first checkpoint: preserve/rename {out} then rerun.')

        print(f'Trial {number+1}/{args.trials}: {params}', flush=True)
        train_classifier_run(SimpleNamespace(
            config=config_path, raw_dir=args.raw_dir, output_dir=out,
            resume=last if last.exists() else None, device=args.device,
            max_hours=remaining / 3600, train_limit=args.train_limit, val_images=args.val_images
        ))

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
            'debug_subset': bool(args.train_limit or args.val_images),
            'ranked_trials': sorted(completed, key=lambda t: t['score'])
        })

    print(f'Search state saved to {state_path}. Rerun to continue; trials is a total target.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=REPO_ROOT / 'configs/task2_classifier.yaml')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, default=REPO_ROOT / 'data/raw/oxford_pets')
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--screen-epochs', type=int, default=5)
    p.add_argument('--max-hours', type=float, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--val-images', type=int, default=0)
    a = p.parse_args()
    if min(a.trials, a.screen_epochs, a.max_hours) <= 0 or min(a.train_limit, a.val_images) < 0:
        p.error('Invalid trial, epoch, time or subset limits.')
    run(a)


if __name__ == '__main__':
    main()
