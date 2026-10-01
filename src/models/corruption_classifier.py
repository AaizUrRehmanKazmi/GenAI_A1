"""Provisional four-class CNN; forward returns logits for cross entropy."""
from torch import nn

class CorruptionClassifier(nn.Module):
    def __init__(self, channels=(16,32,64), dropout=.1):
        super().__init__()
        if len(channels)!=3 or any(type(c) is not int or c<1 for c in channels):
            raise ValueError('Three positive channel widths required')
        if not 0<=dropout<1: raise ValueError('Invalid dropout')
        layers=[]; previous=3
        for c in channels:
            layers.extend([nn.Conv2d(previous,c,3,padding=1),nn.ReLU(),nn.AvgPool2d(2)])
            previous=c
        self.features=nn.Sequential(*layers)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(dropout),nn.Linear(previous,4))
    def forward(self,x): return self.head(self.features(x))
