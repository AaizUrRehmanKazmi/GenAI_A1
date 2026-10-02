"""Independent specialist instances using the provisional 16x16 architecture."""
from .spatial16_ae import Spatial16Autoencoder

class SpecialistAutoencoder(Spatial16Autoencoder):
    """No shared weights or pretrained Task1 initialization."""
    pass
