"""Small training-only diagnostic of Task 3 initialization and two-stage gradients."""
import argparse
import json
from pathlib import Path
import torch
from src.models.soft_moe import from_task2_bundle
from src.losses.moe_loss import MoELoss
from src.data.pets_dataset import PetsDataset
from src.data.corruptions import corrupt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('artifacts/task2-delivery'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3-diagnostic.json'))
    a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    model=from_task2_bundle(a.bundle)
    clean=PetsDataset()[0]['image']
    inputs=torch.stack([corrupt(clean,seed=42+i,condition=c)[0] for i,c in enumerate(['clean','salt','blur','occlusion'])])
    targets=clean.unsqueeze(0).repeat(4,1,1,1);labels=torch.arange(4)
    criterion=MoELoss()
    report={'debug_only':True,'data':'one training image, four conditions; not validation evidence','stages':{}}
    for stage,lr in [('warmup',1e-4),('joint',2e-5)]:
        model.set_stage(stage);model.train()
        before=[p.detach().clone() for p in model.experts.parameters()]
        gate_before=[p.detach().clone() for p in model.gate.parameters()]
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
        optimizer.zero_grad();result=model(inputs);metrics=criterion(result,targets,labels)
        assert torch.isfinite(metrics['loss'])
        metrics['loss'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        changed=any(not torch.equal(b,p) for b,p in zip(before,model.experts.parameters()))
        gate_changed=any(not torch.equal(b,p) for b,p in zip(gate_before,model.gate.parameters()))
        assert changed == (stage=='joint') and gate_changed
        report['stages'][stage]={'metrics':{k:float(v.detach()) for k,v in metrics.items()},'expert_weights_changed':changed,'gate_changed':gate_changed,'mean_routing_weights':result['weights'].detach().mean(0).tolist()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
