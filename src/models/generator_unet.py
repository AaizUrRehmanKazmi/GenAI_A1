"""128px paired U-Net with learned, spatially broadcast style conditioning."""
import torch
from torch import nn

class StyleUNet(nn.Module):
    def __init__(self, base_channels=32, style_dim=8, dropout=.2):
        super().__init__()
        if base_channels<1 or style_dim<1 or not 0<=dropout<1:raise ValueError('Invalid model settings')
        self.style_embedding=nn.Embedding(3,style_dim)
        widths=[base_channels,base_channels*2,base_channels*4,base_channels*8]
        self.down=nn.ModuleList();previous=3+style_dim
        for w in widths:
            self.down.append(nn.Sequential(nn.Conv2d(previous,w,4,2,1),nn.LeakyReLU(.2)))
            previous=w
        self.up=nn.ModuleList()
        for i,w in enumerate(reversed(widths[:-1])):
            self.up.append(nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(previous,w,3,padding=1),nn.ReLU(),nn.Dropout2d(dropout if i==0 else 0)))
            previous=w*2
        self.output=nn.Sequential(nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(previous,3,3,padding=1),nn.Sigmoid())

    def forward(self,photo,style):
        if photo.ndim!=4 or tuple(photo.shape[1:])!=(3,128,128):raise ValueError('Expected NCHW RGB 128px')
        if style.dtype!=torch.long or style.shape!=(len(photo),):raise ValueError('Style must be int64 [N]')
        embedding=self.style_embedding(style)[:,:,None,None].expand(-1,-1,128,128)
        x=torch.cat([photo,embedding],1);features=[]
        for block in self.down:x=block(x);features.append(x)
        for block,skip in zip(self.up,reversed(features[:-1])):x=torch.cat([block(x),skip],1)
        return self.output(x)
