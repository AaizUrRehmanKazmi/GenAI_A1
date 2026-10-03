"""Conditional 70x70 PatchGAN logits for photo/sketch pairs and style."""
import torch
from torch import nn

class StylePatchGAN(nn.Module):
    def __init__(self,base_channels=32,style_dim=8):
        super().__init__()
        if base_channels<1 or style_dim<1:raise ValueError('Invalid model settings')
        self.style_embedding=nn.Embedding(3,style_dim)
        layers=[];previous=6+style_dim
        for width,stride in [(base_channels,2),(base_channels*2,2),(base_channels*4,2),(base_channels*8,1)]:
            layers.extend([nn.Conv2d(previous,width,4,stride,1),nn.LeakyReLU(.2)]);previous=width
        layers.append(nn.Conv2d(previous,1,4,1,1));self.network=nn.Sequential(*layers)
    def forward(self,photo,sketch,style):
        if photo.shape!=sketch.shape or photo.ndim!=4 or tuple(photo.shape[1:])!=(3,128,128):raise ValueError('Expected matching RGB128 pairs')
        if style.dtype!=torch.long or style.shape!=(len(photo),):raise ValueError('Style must be int64 [N]')
        e=self.style_embedding(style)[:,:,None,None].expand(-1,-1,128,128)
        return self.network(torch.cat([photo,sketch,e],1))
