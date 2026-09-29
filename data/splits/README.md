# Shared Oxford-IIIT Pet split for Tasks 1–3

From the repository root, run:
```sh
python3 scripts/prepare_pets.py
```
No additional Python packages are required. Default paths work even when the command is launched from another directory. For a different dataset location use `--raw-dir /path/to/oxford_pets`; use `--output-dir /path/to/splits` for a separate output location.

Files:
- `pets_train.json`: 2,944 relative image paths.
- `pets_val.json`: 736 relative image paths.
- `pets_test.json`: 3,669 relative image paths in official test-list order.
- `pets_split_metadata.json`: seed, algorithm, counts, annotation hashes and checks.

Each list contains strings such as `images/Abyssinian_1.jpg`. Resolve paths against the dataset root (`data/raw/oxford_pets` by default), not against this splits directory. Load JSON with `json.load`; all three restoration tasks must reuse these same lists.

The script copies the official trainval ID order, shuffles with a local Python `random.Random(42)` instance, assigns the first 736 IDs to validation and the remainder to training. The official test membership and order are preserved. Seed 42 alone does not specify a split across different libraries: the shuffle method and initial ordering matter. Commit these generated lists to make membership explicit across machines.

Existing identical outputs are accepted without rewriting. Conflicting outputs cause an error before any output is written; review the discrepancy rather than deleting an established split casually. The metadata includes SHA-256 hashes of the source annotation files (not image-file integrity hashes).

This stage checks unique IDs, official counts, split disjointness and referenced-file presence. It excludes the 41 unlisted JPEG files found in the current download. It does not decode images, resize them, create corruptions, evaluate test images or select model settings.

Verify the script with:
```sh
python3 -m unittest discover -s tests -p test_prepare_pets.py -v
```
