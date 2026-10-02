"""Hard routing for mixed batches; class order clean/salt/blur/occlusion."""
import torch
from torch import nn

class HardRouter(nn.Module):
    def __init__(self,classifier,salt,blur,occlusion):
        super().__init__()
        self.classifier=classifier
        self.experts=nn.ModuleList([salt,blur,occlusion])
        ids=[{id(p) for p in m.parameters()} for m in self.experts]
        if len({id(m) for m in self.experts})!=3 or any(ids[i]&ids[j] for i in range(3) for j in range(i)):
            raise ValueError('Specialists must have independent parameters')
    def forward(self,image,labels=None):
        # Pass manifest labels for oracle routing; classifier is not called then.
        probabilities=None
        if labels is None:
            probabilities=self.classifier(image).softmax(1)
            labels=probabilities.argmax(1)
        if labels.shape!=(len(image),) or labels.dtype!=torch.long or labels.device!=image.device:
            raise ValueError('Labels must be int64 [batch] on the input device')
        if ((labels<0)|(labels>3)).any():raise ValueError('Invalid routing label')
        output=image.clone()
        for label,expert in enumerate(self.experts,1):
            mask=labels==label
            if mask.any():output[mask]=expert(image[mask])
        return {'output':output,'labels':labels,'probabilities':probabilities}
