"""Provisional Task 1 autoencoder: explicit vector bottleneck, no skip connections."""
import torch
from torch import nn


class UniversalAutoencoder(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), latent_dim=256, dropout=0.1):
        super().__init__()
        if len(channels) != 4 or any(type(c) is not int or c <= 0 for c in channels):
            raise ValueError('channels must contain four positive integers.')
        if type(latent_dim) is not int or not 0 < latent_dim < min(channels[-1] * 8 * 8, 3 * 128 * 128):
            raise ValueError('latent_dim must provide genuine compression.')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1).')
        encoder = []
        previous = 3
        for c in channels:
            encoder += [nn.Conv2d(previous, c, 3, stride=2, padding=1), nn.ReLU(), nn.Dropout2d(dropout)]
            previous = c
        self.encoder = nn.Sequential(*encoder)
        self.to_latent = nn.Linear(channels[-1] * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, channels[-1] * 8 * 8)
        self.last_channels = channels[-1]
        decoder = []
        for index, c in enumerate([channels[2], channels[1], channels[0], 3]):
            decoder += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(previous, c, 3, padding=1)]
            decoder += [nn.Sigmoid() if index == 3 else nn.ReLU()]
            previous = c
        self.decoder = nn.Sequential(*decoder)

    def encode(self, image):
        return self.to_latent(self.encoder(image).flatten(1))

    def forward(self, image):
        latent = self.encode(image)
        features = torch.relu(self.from_latent(latent)).reshape(-1, self.last_channels, 8, 8)
        return self.decoder(features)
