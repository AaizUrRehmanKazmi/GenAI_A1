"""Style-conditioned U-Net generator and PatchGAN discriminator, plus a differentiable SSIM."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_STYLES = 3


def _style_map(emb, h, w):
    return emb[:, :, None, None].expand(-1, -1, h, w)


class Down(nn.Module):
    def __init__(self, cin, cout, norm=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.net(x), skip], dim=1)


class StyleUNetGenerator(nn.Module):
    """128x128 U-Net. The learned style embedding is (a) broadcast and concatenated to the input
    photo and (b) projected and added at the 2x2 bottleneck, so it conditions the whole decoder."""

    def __init__(self, base=64, style_dim=16, dropout=0.3, in_ch=3, out_ch=3):
        super().__init__()
        c = base
        self.embed = nn.Embedding(N_STYLES, style_dim)
        self.bottleneck_proj = nn.Linear(style_dim, c * 8)
        self.d1 = Down(in_ch + style_dim, c, norm=False)  # 64
        self.d2 = Down(c, c * 2)                          # 32
        self.d3 = Down(c * 2, c * 4)                      # 16
        self.d4 = Down(c * 4, c * 8)                      # 8
        self.d5 = Down(c * 8, c * 8)                      # 4
        self.d6 = Down(c * 8, c * 8)                      # 2
        self.u1 = Up(c * 8, c * 8, dropout)               # 4  -> cat -> 16c
        self.u2 = Up(c * 16, c * 8, dropout)              # 8  -> cat -> 16c
        self.u3 = Up(c * 16, c * 4, dropout)              # 16 -> cat -> 8c
        self.u4 = Up(c * 8, c * 2)                        # 32 -> cat -> 4c
        self.u5 = Up(c * 4, c)                            # 64 -> cat -> 2c
        self.out = nn.Sequential(nn.ConvTranspose2d(c * 2, out_ch, 4, 2, 1), nn.Tanh())  # 128

    def forward(self, photo, style):
        e = self.embed(style)
        x = torch.cat([photo, _style_map(e, photo.shape[2], photo.shape[3])], dim=1)
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        e5 = self.d5(e4)
        e6 = self.d6(e5) + self.bottleneck_proj(e)[:, :, None, None]
        y = self.u1(e6, e5)
        y = self.u2(y, e4)
        y = self.u3(y, e3)
        y = self.u4(y, e2)
        y = self.u5(y, e1)
        return self.out(y)


class StylePatchDiscriminator(nn.Module):
    """PatchGAN on (photo, sketch, style-map). Returns a grid of real/fake logits."""

    def __init__(self, base=64, style_dim=16, in_ch=3, sk_ch=3):
        super().__init__()
        c = base
        self.embed = nn.Embedding(N_STYLES, style_dim)
        cin = in_ch + sk_ch + style_dim
        self.net = nn.Sequential(
            nn.Conv2d(cin, c, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 4, c * 8, 4, 1, 1, bias=False), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c * 8, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch, style):
        e = self.embed(style)
        x = torch.cat([photo, sketch, _style_map(e, photo.shape[2], photo.shape[3])], dim=1)
        return self.net(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)



class CandidateGenerator(StyleUNetGenerator):
    """External [0,1] contract; internal tanh model uses [-1,1]."""
    def __init__(self, base_channels=32, style_dim=8, dropout=.2):
        super().__init__(base=base_channels, style_dim=style_dim, dropout=dropout)
        self.apply(init_weights)

    def forward(self, photo, style):
        return (super().forward(photo * 2 - 1, style) + 1) / 2


class CandidateDiscriminator(StylePatchDiscriminator):
    def __init__(self, base_channels=32, style_dim=8):
        super().__init__(base=base_channels, style_dim=style_dim)
        self.apply(init_weights)

    def forward(self, photo, sketch, style):
        return super().forward(photo * 2 - 1, sketch * 2 - 1, style)
