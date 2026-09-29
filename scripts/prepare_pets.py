"""Create the shared Tasks 1–3 split from Oxford-IIIT Pet's official lists.

Only filenames are checked here. Images are not decoded, resized or modified.
Default paths are relative to this repository, independent of your working directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXPECTED_TRAINVAL = 3680
EXPECTED_TEST = 3669
VALIDATION_COUNT = 736


def read_ids(path: Path) -> list[str]:
    """Read the first column, ignoring blank lines and annotation comments."""
    ids, seen = [], set()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        image_id = line.split()[0]
        if not re.fullmatch(r'[A-Za-z0-9_]+', image_id):
            raise ValueError(f'{path}:{number}: invalid image ID {image_id!r}')
        if image_id in seen:
            raise ValueError(f'{path}:{number}: duplicate image ID {image_id!r}')
        seen.add(image_id)
        ids.append(image_id)
    return ids


def split_trainval(ids: list[str]) -> tuple[list[str], list[str]]:
    """Shuffle a copy with a local RNG; reserve the first 736 for validation."""
    shuffled = list(ids)
    random.Random(SEED).shuffle(shuffled)
    return shuffled[VALIDATION_COUNT:], shuffled[:VALIDATION_COUNT]


def prepare(raw_dir: Path, output_dir: Path) -> dict:
    trainval_path = raw_dir / 'annotations/trainval.txt'
    test_path = raw_dir / 'annotations/test.txt'
    trainval, test = read_ids(trainval_path), read_ids(test_path)
    if len(trainval) != EXPECTED_TRAINVAL or len(test) != EXPECTED_TEST:
        raise ValueError(
            f'Expected {EXPECTED_TRAINVAL} trainval and {EXPECTED_TEST} test IDs; '
            f'found {len(trainval)} and {len(test)}. Check the official annotations.'
        )
    if set(trainval) & set(test):
        raise ValueError('Official trainval and test lists overlap.')
    missing = [image_id for image_id in trainval + test
               if not (raw_dir / 'images' / f'{image_id}.jpg').is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} referenced images; examples: {missing[:10]}')

    train, val = split_trainval(trainval)
    if (set(train) & set(val)) or (set(train) | set(val)) != set(trainval):
        raise ValueError('Training/validation partition failed validation.')
    groups = {'train': train, 'val': val, 'test': test}
    # Portable paths relative to the dataset root, not this machine.
    documents = {f'pets_{name}.json': [f'images/{image_id}.jpg' for image_id in ids]
                 for name, ids in groups.items()}
    referenced = set(trainval) | set(test)
    extra_count = sum(p.stem not in referenced for p in (raw_dir / 'images').glob('*.jpg')
                      if p.is_file())
    metadata = {
        'schema_version': 1,
        'dataset': 'Oxford-IIIT Pet',
        'source_url': 'https://www.robots.ox.ac.uk/~vgg/data/pets/',
        'seed': SEED,
        'validation_fraction': 0.2,
        'algorithm': 'Python random.Random(42).shuffle on official trainval order; first 736 IDs are validation',
        'path_base': 'Dataset root containing images/ and annotations/',
        'counts': {name: len(ids) for name, ids in groups.items()},
        'unlisted_jpeg_files_excluded': extra_count,
        'annotation_sha256': {
            'trainval.txt': hashlib.sha256(trainval_path.read_bytes()).hexdigest(),
            'test.txt': hashlib.sha256(test_path.read_bytes()).hexdigest(),
        },
        'checks': {
            'unique_ids': True, 'disjoint_splits': True,
            'all_referenced_files_exist': True, 'official_test_order_preserved': True,
            'image_decoding_checked': False,
        },
    }
    documents['pets_split_metadata.json'] = metadata
    serialized = {name: json.dumps(value, indent=2) + '\n' for name, value in documents.items()}
    # Never silently replace an established experiment's split or provenance.
    for name, content in serialized.items():
        target = output_dir / name
        if target.exists() and target.read_text() != content:
            raise ValueError(f'Existing output differs: {target}. Review it before regenerating; '
                             'use --output-dir for a separate candidate split.')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in serialized.items():
        target = output_dir / name
        if not target.exists():
            temporary = target.with_suffix('.json.tmp')
            temporary.write_text(content)
            temporary.replace(target)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=REPO_ROOT / 'data/raw/oxford_pets')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'data/splits')
    args = parser.parse_args()
    try:
        metadata = prepare(args.raw_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Dataset preparation failed: {exc}\n')
    for name, count in metadata['counts'].items():
        print(f'{name}: {count}')
    print(f"Unlisted JPEGs excluded: {metadata['unlisted_jpeg_files_excluded']}")
    print(f'Split files verified/saved in: {args.output_dir.resolve()}')
    print('Image contents have not been decoded or modified.')


if __name__ == '__main__':
    main()
