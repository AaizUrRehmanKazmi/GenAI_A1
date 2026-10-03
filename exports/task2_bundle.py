"""Package the selected Task 2 checkpoints; export and verify four ONNX graphs.

Run as a module from the repository root. Training sources remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch
from src.models.corruption_classifier import CorruptionClassifier
from src.models.specialist_ae import SpecialistAutoencoder
from src.models.hard_router import HardRouter
from src.data.pets_dataset import PetsDataset
from src.data.corruptions import corrupt

ROOT = Path(__file__).resolve().parents[1]
SELECTED = {
    'classifier': ('classifier_trial005_continued.zip', 'classifier-trial005-continuation', '387ab888695fca654b37a227725b1a88d8a63a2df4ac73294f4305b4bfeca5ed', 10),
    'salt': ('salt_trial007_continued.zip', 'trial_007', '9dbfabdbfecb10a7ca832b12f4e1ca8553ca62809af94d81609a6a158374a56b', 19),
    'blur': ('blur_trial007_continued.zip', 'trial_007', 'c609d739e50443a411603642b74333d2e2a8b13a9b6fe61e8acdf7b17b50007a', 20),
    'occlusion': ('occlusion_trial007_continued.zip', 'trial_007', '71b4cc936ed799c4a758a0552fb8610d6134c4b4b0c7341f54389e38b37fbb11', 20),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def route(sessions, images, labels=None):
    """Application-side hard routing: only selected experts run; clean is exact identity."""
    images = np.asarray(images)
    if images.dtype != np.float32 or images.ndim != 4 or images.shape[1:] != (3, 128, 128) or not len(images):
        raise ValueError('Expected nonempty float32 [N,3,128,128] input')
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Input must be finite and in [0,1]')
    if labels is None:
        logits = sessions['classifier'].run(None, {'image': images})[0]
        labels = logits.argmax(1)
    labels = np.asarray(labels)
    if labels.shape != (len(images),) or labels.dtype != np.int64 or np.any((labels < 0) | (labels > 3)):
        raise ValueError('Expected int64 [N] labels in [0,3]')
    result = images.copy()
    for index, name in enumerate(('salt', 'blur', 'occlusion'), 1):
        mask = labels == index
        if mask.any():
            result[mask] = sessions[name].run(None, {'image': images[mask]})[0]
    return result, labels


def compare(reference, actual):
    if reference.shape != actual.shape or not np.isfinite(actual).all():
        raise AssertionError('Invalid ONNX output')
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    return {'max_absolute_error': float(np.abs(reference - actual).max()), 'passed': True}


def build(backups, output):
    import onnx
    import onnxruntime as ort
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    models, sessions, records = {}, {}, {}
    for name, (archive, prefix, expected, epoch) in SELECTED.items():
        folder = output / name
        folder.mkdir()
        with zipfile.ZipFile(backups / archive) as z:
            # Read only explicitly named files; never extract arbitrary archive paths.
            for filename in ('best.pt', 'config.yaml', 'history.json'):
                (folder / filename).write_bytes(z.read(f'{prefix}/{filename}'))
        checkpoint = folder / 'best.pt'
        if digest(checkpoint) != expected:
            raise ValueError(f'{name}: checkpoint differs from selected validation run')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['version'] != 1 or saved['progress']['epoch'] != epoch:
            raise ValueError(f'{name}: checkpoint format/epoch mismatch')
        signature = saved['data_signature']
        if signature['train_limit'] or signature['val_images']:
            raise ValueError('Debug checkpoint is not deliverable')
        hashes = signature['hashes'] if name == 'classifier' else signature['source']
        sources = ['src/models/corruption_classifier.py'] if name == 'classifier' else ['src/models/specialist_ae.py', 'src/models/spatial16_ae.py']
        for source in sources:
            if hashes.get(source) != digest(ROOT / source):
                raise ValueError(f'Changed model source: {source}')
        if name != 'classifier' and saved['config'].get('condition') != name:
            raise ValueError('Wrong specialist condition')
        cls = CorruptionClassifier if name == 'classifier' else SpecialistAutoencoder
        model = cls(**saved['config']['model']).eval()
        model.load_state_dict(saved['model'], strict=True)
        models[name] = model
        path = folder / 'model.onnx'
        torch.onnx.export(model, torch.zeros(1, 3, 128, 128), str(path),
                          input_names=['image'], output_names=['logits' if name == 'classifier' else 'restored'],
                          dynamic_axes={'image': {0: 'batch'}, ('logits' if name == 'classifier' else 'restored'): {0: 'batch'}},
                          opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        sessions[name] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        records[name] = {'checkpoint_sha256': expected, 'epoch': epoch,
                         'onnx_sha256': digest(path), 'config': saved['config'], 'parity': []}
        print(f'Exported {name}', flush=True)

    # Validation images only, with seeded examples for every condition. No official test data.
    dataset = PetsDataset(ROOT / 'data/splits/pets_val.json')
    cases = []
    for i in range(4):
        clean = dataset[i]['image']
        for condition in ('clean', 'salt', 'blur', 'occlusion'):
            cases.append(corrupt(clean, seed=1000 + len(cases), condition=condition)[0])
    rng = np.random.default_rng(42)
    batches = [torch.stack(cases[i:i+4]).numpy() for i in range(0, 16, 4)]
    batches += [np.zeros((1,3,128,128), np.float32), np.ones((1,3,128,128), np.float32),
                rng.random((3,3,128,128), dtype=np.float32)]
    router = HardRouter(models['classifier'], models['salt'], models['blur'], models['occlusion']).eval()
    routing = []
    with torch.inference_mode():
        for images in batches:
            tensor = torch.from_numpy(images)
            for name, model in models.items():
                ref = model(tensor).numpy()
                actual = sessions[name].run(None, {'image': images})[0]
                records[name]['parity'].append(compare(ref, actual))
                if name == 'classifier':
                    np.testing.assert_array_equal(ref.argmax(1), actual.argmax(1))
            actual, labels = route(sessions, images)
            ref = router(tensor)
            np.testing.assert_array_equal(labels, ref['labels'].numpy())
            routing.append(compare(ref['output'].numpy(), actual))
            np.testing.assert_array_equal(actual[labels == 0], images[labels == 0])
        # Force all branches, including mixed, repeated and absent expert assignments.
        for labels in (np.array([0,1,2,3]), np.array([0,0,0,0]), np.array([3,1,3,1])):
            images = batches[0]
            actual, _ = route(sessions, images, labels)
            ref = router(torch.from_numpy(images), torch.from_numpy(labels))['output'].numpy()
            routing.append(compare(ref, actual))
            np.testing.assert_array_equal(actual[labels == 0], images[labels == 0])
    manifest = {'status': 'verified', 'class_order': ['clean','salt','blur','occlusion'],
                'input': 'float32 NCHW [N,3,128,128], RGB [0,1], EXIF transpose then PIL bilinear resize',
                'routing': 'classifier argmax; clean identity; other classes dispatched to named ONNX specialist',
                'opset': 17, 'rtol': 1e-4, 'atol': 1e-5, 'models': records, 'routing_parity': routing,
                'verification': '16 seeded validation-image cases plus zeros, ones, random inputs; batches 1,3,4; CPU float32',
                'versions': {'torch': torch.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'README.md').write_text('# Selected Task 2 inference bundle\n\nFour ONNX graphs, selected PyTorch checkpoints, configurations and training histories.\nSee manifest.json for hashes, classes, preprocessing and numerical verification.\nClean images bypass reconstruction exactly. Classifier output is logits.\nThis is an inference package; retain original training ZIPs for full resume history.\nParity checks are not a new quality evaluation or a test-set evaluation.\n')
    shutil.copy2(__file__, output / 'task2_bundle.py')
    archive = shutil.make_archive(str(output), 'zip', root_dir=output.parent, base_dir=output.name)
    print(f'Verified bundle: {archive}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backups-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.backups_dir, args.output_dir)


if __name__ == '__main__':
    main()
