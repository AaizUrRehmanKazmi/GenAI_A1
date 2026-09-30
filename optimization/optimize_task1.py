"""Sequential Optuna screening with atomic JSON persistence and resumable trainer runs."""
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
from src.data.pets_dataset import REPO_ROOT, EvaluationPetsDataset
from src.losses.reconstruction import ReconstructionLoss
from training.train_task1 import validation
from training.train_task1_spatial16 import run as train


def score(row):
    # Training alpha varies; ranking must use the same scale for every trial.
    return .8 * row['val_l1'] + .2 * (1 - row['val_ssim'])


def distributions():
    return {
        'learning_rate': optuna.distributions.FloatDistribution(0.0002, 0.002, log=True),
        'batch_size': optuna.distributions.CategoricalDistribution([8, 16]),
        'latent_channels': optuna.distributions.CategoricalDistribution([16, 32, 64]),
        'dropout': optuna.distributions.CategoricalDistribution([0., .05, .1]),
        'alpha': optuna.distributions.CategoricalDistribution([.5, .65, .8, .9]),
    }


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(args):
    started = time.monotonic()
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.config.read_text())
    files = ['optimization/optimize_task1.py', 'training/train_task1_spatial16.py',
             'training/train_task1.py', 'src/models/spatial16_ae.py', 'src/data/pets_dataset.py',
             'src/data/corruptions.py', 'src/losses/reconstruction.py',
             'data/splits/pets_train.json', 'data/splits/pets_val.json',
             'data/manifests/pets_val_corruptions.json']
    protocol = {'base':base, 'screen_epochs':args.screen_epochs, 'train_limit':args.train_limit,
        'val_images':args.val_images, 'optuna':optuna.__version__, 'objective':'0.8*L1+0.2*(1-SSIM) at final screening epoch',
        'hashes':{f:hashlib.sha256((REPO_ROOT/f).read_bytes()).hexdigest() for f in files}}
    state_path = root/'study.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'protocol':protocol,'trials':[]}
    if state['protocol'] != protocol:
        raise ValueError('Search protocol/source changed; use a new output directory.')
    write_json(state_path, state)
    if 'input_baseline' not in state:
        from torch.utils.data import Subset
        dataset = EvaluationPetsDataset(REPO_ROOT/'data/splits/pets_val.json', REPO_ROOT/'data/manifests/pets_val_corruptions.json', args.raw_dir)
        if args.val_images: dataset = Subset(dataset, range(min(len(dataset), args.val_images*10)))
        torch.set_num_threads(2)
        result = validation(torch.nn.Identity(), dataset, ReconstructionLoss(.8), torch.device(args.device), 16,
                            lambda: time.monotonic()-started >= args.max_hours*3600)
        if result is None: return
        state['input_baseline'] = result
        write_json(state_path,state)
    while len([t for t in state['trials'] if t['state']=='COMPLETE']) < args.trials:
        remaining = args.max_hours*3600 - (time.monotonic()-started)
        if remaining < 60: break
        pending = next((t for t in state['trials'] if t['state']=='RUNNING'), None)
        if pending is None:
            number = len(state['trials'])
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42+number, n_startup_trials=4))
            for t in state['trials']:
                study.add_trial(optuna.trial.create_trial(params=t['params'], distributions=distributions(), value=t['score']))
            trial = study.ask(distributions())
            params = trial.params
            if number == 0:
                params = dict(learning_rate=.001,batch_size=16,latent_channels=32,dropout=0.,alpha=.8)
            pending={'number':number, 'state':'RUNNING', 'params':params}
            state['trials'].append(pending); write_json(state_path,state)
        number=pending['number']; params=pending['params']
        config=copy.deepcopy(base)
        for key in ['latent_channels','dropout']:config['model'][key]=params[key]
        for key in ['learning_rate','batch_size']:config['training'][key]=params[key]
        config['loss']['alpha']=params['alpha']
        config_path=root/f'trial_{number:03d}.yaml';config_path.write_text(yaml.safe_dump(config))
        out=root/f'trial_{number:03d}'; last=out/'last.pt'
        # Recover interrupted setup before the first checkpoint without overwriting unrelated artifacts.
        if out.exists() and not last.exists() and any(out.iterdir()):
            raise ValueError(f'Interrupted before first checkpoint: preserve/rename {out} then rerun.')
        print(f'Trial {number+1}/{args.trials}: {params}',flush=True)
        train(SimpleNamespace(config=config_path,epochs=None,stop_after_epoch=args.screen_epochs,
            raw_dir=args.raw_dir,splits_dir=REPO_ROOT/'data/splits',manifests_dir=REPO_ROOT/'data/manifests',
            output_dir=out,resume=last if last.exists() else None,device=args.device,max_hours=remaining/3600,
            train_limit=args.train_limit,val_images=args.val_images,cpu_threads=2))
        history=json.loads((out/'history.json').read_text())
        if not history or history[-1]['epoch'] < args.screen_epochs: break
        row=history[-1]
        pending.update(state='COMPLETE',score=score(row),metrics=row,checkpoint=str(last))
        write_json(state_path,state)
        completed=[t for t in state['trials'] if t['state']=='COMPLETE']
        winner=min(completed,key=lambda t:t['score'])
        best_config=yaml.safe_load((root/f"trial_{winner['number']:03d}.yaml").read_text())
        (root/'selected_config.yaml').write_text(yaml.safe_dump(best_config))
        write_json(root/'leaderboard.json',{'provisional':True,'debug_subset':bool(args.train_limit or args.val_images),
            'input_baseline':state['input_baseline']['balanced'], 'ranked_trials':sorted(completed,key=lambda t:t['score'])})
    print(f'Search state saved to {state_path}. Rerun the same command to continue; trials is a total target.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=REPO_ROOT/'configs/task1_spatial16.yaml')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path,default=REPO_ROOT/'data/raw/oxford_pets')
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--screen-epochs',type=int,default=5)
    p.add_argument('--max-hours',type=float,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--train-limit',type=int,default=0)
    p.add_argument('--val-images',type=int,default=0)
    a=p.parse_args()
    config=yaml.safe_load(a.config.read_text())
    if min(a.trials,a.screen_epochs,a.max_hours)<=0 or min(a.train_limit,a.val_images)<0 or a.screen_epochs>config['training']['epochs']:
        p.error('Invalid trial/epoch/time/subset limits.')
    run(a)

if __name__=='__main__': main()
