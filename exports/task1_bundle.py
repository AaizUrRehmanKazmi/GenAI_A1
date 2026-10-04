"""Package selected Task 1 universal autoencoder and export to ONNX."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.models.universal_ae import UniversalAutoencoder
from src.models.spatial_ae import SpatialAutoencoder
from src.models.spatial16_ae import Spatial16Autoencoder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(ref, actual):
    diff = np.abs(ref - actual)
    return {'max_abs_error': float(diff.max()), 'mean_abs_error': float(diff.mean())}


def build_model(config):
    model_config = dict(config['model'])
    if 'latent_dim' in model_config:
        return UniversalAutoencoder(**model_config), 'universal_ae'
    if 'latent_channels' not in model_config:
        raise ValueError(f"Unsupported Task 1 model config: {model_config}")
    architecture = config.get('architecture')
    if architecture == 'spatial_ae':
        return SpatialAutoencoder(**model_config), 'spatial_ae'
    return Spatial16Autoencoder(**model_config), 'spatial16_ae'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()

    if a.output_dir.exists():
        raise ValueError('Use a new output directory')

    out = a.output_dir
    out.mkdir(parents=True)
    torch.set_num_threads(2)

    # Load checkpoint
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = ck['config']
    print(f"Loaded checkpoint: epoch {ck['progress']['epoch']}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")

    # Copy checkpoint and config
    shutil.copy2(a.checkpoint, out / 'best.pt')
    if 'history' in ck:
        (out / 'history.json').write_text(json.dumps(ck.get('history', {}), indent=2))

    # Build model
    model, architecture = build_model(config)
    model.load_state_dict(ck['model'])
    model.eval()

    # Export to ONNX
    onnx_path = out / 'model.onnx'
    dummy = torch.zeros(1, 3, 128, 128)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=['image'],
        output_names=['restored'],
        dynamic_axes={'image': {0: 'batch'}, 'restored': {0: 'batch'}},
        opset_version=17,
        dynamo=False
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print("ONNX checker passed")

    # Verify parity
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])

    checks = []
    with torch.inference_mode():
        for name, batch in [('zeros', torch.zeros(1, 3, 128, 128)),
                            ('ones', torch.ones(1, 3, 128, 128)),
                            ('random', torch.rand(4, 3, 128, 128))]:
            ref = model(batch).numpy()
            actual = session.run(['restored'], {'image': batch.numpy()})[0]
            result = compare(ref, actual)
            result['input'] = name
            result['batch'] = len(batch)
            checks.append(result)
            assert np.allclose(ref, actual, rtol=1e-4, atol=1e-5), f"Parity failed for {name}"
            print(f"  {name}: max_error={result['max_abs_error']:.2e}")

    manifest = {
        'status': 'verified',
        'task': 1,
        'architecture': architecture,
        'checkpoint_sha256': digest(out / 'best.pt'),
        'onnx_sha256': digest(onnx_path),
        'epoch': ck['progress']['epoch'],
        'config': config,
        'input': 'RGB EXIF-transposed bilinear128 float32 NCHW [0,1]',
        'checks': checks
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"\nBundle written to {out}")
    print(f"ONNX SHA256: {manifest['onnx_sha256']}")


if __name__ == '__main__':
    main()
