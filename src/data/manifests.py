"""Fixed clean + nine corruption cases per image for validation and final test.

Generation reads path lists only, never image pixels. Seeds are stable hashes of
split, image path and case; Python's process-randomized hash() is not used.
"""
import hashlib
import json
from pathlib import Path

from .corruptions import make_spec

CASES = [('clean', 'clean', {})]
CASES += [('salt', level, {'probability': p}) for level, p in zip(
    ['low', 'medium', 'high'], [.03, .08, .15])]
CASES += [('blur', level, {'kernel_size': k, 'sigma': s}) for level, (k, s) in zip(
    ['low', 'medium', 'high'], [(3, .7), (5, 1.5), (7, 2.5)])]
CASES += [('occlusion', level, {'rectangles': n, 'area_fraction': a}) for level, (n, a) in zip(
    ['low', 'medium', 'high'], [(1, .10), (2, .20), (3, .35)])]
PREPROCESSING = {'size': [128, 128], 'color': 'RGB', 'interpolation': 'bilinear',
                 'exif_transpose': True, 'range': [0, 1]}


def read_paths(split_file):
    paths = json.loads(Path(split_file).read_text())
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        raise ValueError('Expected a non-empty JSON list of paths.')
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate paths in split.')
    for p in paths:
        if Path(p).is_absolute() or '..' in Path(p).parts or not p.startswith('images/'):
            raise ValueError(f'Invalid relative image path: {p}')
    return paths


def paths_digest(paths):
    return hashlib.sha256(json.dumps(paths, separators=(',', ':')).encode()).hexdigest()


def generate_manifest(split_file, *, split, seed=42):
    if split not in ('val', 'test'):
        raise ValueError('Manifests are only for val or test.')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Invalid master seed.')
    paths = read_paths(split_file)
    records = []
    for path in paths:
        for condition, severity, settings in CASES:
            key = json.dumps([seed, split, path, condition, severity]).encode()
            case_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**63)
            records.append({'path': path, 'image_id': Path(path).stem, 'severity': severity,
                            'spec': make_spec(condition, seed=case_seed, **settings)})
    return {'schema_version': 1, 'split': split, 'seed': seed,
            'split_sha256': paths_digest(paths), 'image_count': len(paths),
            'preprocessing': PREPROCESSING.copy(), 'records': records}


def save_manifest(manifest, destination):
    """Accept identical reruns; never silently replace established evaluation data."""
    destination = Path(destination)
    content = json.dumps(manifest, indent=2) + '\n'
    if destination.exists():
        if destination.read_text() != content:
            raise ValueError(f'Conflicting manifest: {destination}; review before replacing it.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(content)
    temporary.replace(destination)


def load_manifest(manifest_file, split_file):
    manifest = json.loads(Path(manifest_file).read_text())
    paths = read_paths(split_file)
    if (manifest.get('schema_version') != 1 or manifest.get('split') not in ('val', 'test')
            or manifest.get('preprocessing') != PREPROCESSING
            or manifest.get('split_sha256') != paths_digest(paths)
            or manifest.get('image_count') != len(paths)):
        raise ValueError('Manifest schema, preprocessing or split does not match.')
    records = manifest.get('records', [])
    if len(records) != 10 * len(paths):
        raise ValueError('Expected exactly ten evaluation cases per image.')
    # Reconstruct the specification, not image pixels, to reject modified settings,
    # duplicate/missing cases or path substitutions before evaluation starts.
    expected = generate_manifest(split_file, split=manifest['split'], seed=manifest['seed'])
    if manifest != expected:
        raise ValueError('Manifest records differ from the documented fixed case specification.')
    return manifest
