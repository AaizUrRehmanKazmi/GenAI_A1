# Task 4 data preparation

Official source: https://github.com/DengPingFan/FS2K
Download photo+sketch+annotations using the linked Google Drive archive.
Extract so the chosen raw directory directly contains photo/, sketch/,
anno_train.json and anno_test.json. Photo directory numbers identify source
collections, not sketch styles; use annotation style 0/1/2 (UI Style1/2/3).
Pair naming follows the official tools/split_train_test.py mapping.

```sh
python -m scripts.prepare_fs2k --raw-dir data/raw/fs2k/FS2K
```

The command validates IDs, style range, pair paths, official counts and split
non-overlap; writes deterministic style-stratified seed42 training/validation
manifests and preserves the official test list. Per-style nearest-integer 15%
rounding gives 160 validation and 898 training entries for official counts
357/351/350. Test1046 remains held out. Annotation hashes are recorded. Existing
different manifests are rejected. Test pixels are not decoded. Pair paths are
checked for existence only; image decoding/visual alignment audit is next.

Two synthetic tests passed. Actual archive is not yet present locally; real-data
verification and dataset loader/GAN training are pending. Do not infer three
paired styles per individual photograph: train from the annotated pair/style.

## Actual dataset verification

Downloaded archive includes one uppercase JPG extension; preparation now resolves
.jpg/.jpeg/.png case-insensitively while rejecting ambiguous duplicates.
Actual preparation completed: train898/val160/test1046. FS2KDataset decodes RGB
pairs, EXIF-transposes, resizes both to 128 square with bilinear and returns CHW
float32 [0,1]. Optional horizontal flip samples once for BOTH images; no unrelated
crop/rotation/color changes. RGB sketch representation preserves the source and
matches a provisional 3-channel generator; grayscale is a later research option.

`python -m scripts.check_fs2k_dataset` decoded all 898 training and 160 validation
pairs, checked shape/range/finiteness and equal original pair dimensions. Official
test pixels were not decoded. Preview artifacts/fs2k-data-checks/pairs.png contains
three training pairs per style; visual inspection found corresponding subjects
and broadly aligned poses. This sampled inspection cannot prove perfect artistic
pixel correspondence across every pair. Two loader tests passed (shared flip,
path escape rejection). Generator/discriminator and training remain next steps.
