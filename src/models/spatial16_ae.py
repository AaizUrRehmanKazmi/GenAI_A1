"""Task 1 convolutional autoencoder with a compressed 16x16 spatial latent and no skips."""
from torch import nn


class Spatial16Autoencoder(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), latent_channels=32, dropout=0.0):
        super().__init__()
        if len(channels) != 4 or any(type(c) is not int or c <= 0 for c in channels):
            raise ValueError('channels must contain four positive integers.')
        if type(latent_channels) is not int or not 0 < latent_channels < min(channels[-1], 192):
            raise ValueError('latent_channels must compress encoder features and input.')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1).')
        encoder = []
        previous = 3
        for index, c in enumerate(channels):
            encoder += [nn.Conv2d(previous, c, 3, stride=2 if index < 3 else 1, padding=1), nn.ReLU(), nn.Dropout2d(dropout)]
            previous = c
        self.encoder = nn.Sequential(*encoder)
        self.to_latent = nn.Conv2d(channels[-1], latent_channels, 1)
        self.from_latent = nn.Sequential(nn.Conv2d(latent_channels, channels[-1], 1), nn.ReLU())
        decoder = []
        for index, c in enumerate([channels[2], channels[1], channels[0], 3]):
            decoder += [nn.Upsample(scale_factor=1 if index == 0 else 2, mode='nearest'), nn.Conv2d(previous, c, 3, padding=1)]
            decoder += [nn.Sigmoid() if index == 3 else nn.ReLU()]
            previous = c
        self.decoder = nn.Sequential(*decoder)

    def encode(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 128, 128):
            raise ValueError('Expected a batch of RGB 128x128 images.')
        return self.to_latent(self.encoder(image))

    def decode(self, latent):
        return self.decoder(self.from_latent(latent))

    def forward(self, image):
        return self.decode(self.encode(image))
