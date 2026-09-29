# Oxford-IIIT Pet development split

Status: split implementation complete; image-content checks pending.

The assignment requires the official trainval collection to be divided into 80% training and 20% validation using seed 42. It requires the same split across Tasks 1–3 and preserves official test data for final evaluation.

Implementation choice: use the standard-library `random.Random(42).shuffle` on a copy of the official trainval order. Take the first 736 IDs for validation and the remaining 2,944 for training. Store the actual lists rather than relying only on a seed. This operation requires no ML environment or third-party dependency.

Alternatives considered structurally, not through model experiments:
- A library split utility can also produce an appropriate random split, but may produce different memberships with the same numeric seed. It adds a dependency unnecessary for this operation.
- Breed-stratified splitting could preserve breed proportions more closely. It has not been tested here; the assignment requests a random split and does not require breed stratification for these restoration tasks. This is not a claim that random splitting performs better.
- Splitting every JPEG in the folder would include images outside the official lists and could mix official test images into development. That approach conflicts with the required data separation.

Evidence: real-data preparation found 3,680 official trainval IDs and 3,669 test IDs, no duplicates, no overlap, and no missing referenced paths. Outputs contain 2,944 training and 736 validation paths; all 41 unlisted JPEGs are excluded. Source annotation hashes are recorded in the metadata. Six automated tests cover repeatability, membership, preserved test order, duplicates, missing files, overlap, unexpected counts and overwrite protection.

Source: supplied assignment PDF, dataset preparation section. Official dataset provenance: https://www.robots.ox.ac.uk/~vgg/data/pets/

Next: verify image decoding and RGB/resizing behavior before corruption implementation. No restoration quality claims have been made.
