"""Confusion rows are true classes; columns are predicted classes."""
import torch

def metrics_from_confusion(matrix):
    cm=torch.as_tensor(matrix,dtype=torch.float64)
    support=cm.sum(1); predicted=cm.sum(0); tp=cm.diag()
    recall=tp/support.clamp_min(1); precision=tp/predicted.clamp_min(1)
    f1=2*precision*recall/(precision+recall).clamp_min(1e-15)
    return {'accuracy':(tp.sum()/cm.sum().clamp_min(1)).item(),
        'macro_precision':precision.mean().item(),'macro_recall':recall.mean().item(),
        'macro_f1':f1.mean().item(),'precision':precision.tolist(),'recall':recall.tolist(),
        'f1':f1.tolist(),'support':support.tolist(),'confusion':cm.int().tolist(),
        'normalized_confusion':(cm/support[:,None].clamp_min(1)).tolist()}
