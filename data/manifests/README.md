# Fixed Oxford-IIIT Pet corruption manifests

Generate once from the saved split lists:
```sh
python -m scripts.generate_manifests
```

- pets_val_corruptions.json: 736 images × 10 cases = 7,360 records.
- pets_test_corruptions.json: 3,669 images × 10 cases = 36,690 records.

Each image has one clean case, three salt probabilities (0.03/0.08/0.15), three blur pairs (3,0.7)/(5,1.5)/(7,2.5), and three occlusion cases (one rectangle/~10%, two/~20%, three/~35%). Validation uses the same severity grid as a documented comparison convention; its seeds are distinct from test seeds. Do not use test results to select models.

Records store relative path, image ID, severity name, corruption label, seed and applicable parameters, including exact rectangle coordinates and requested/actual mask area. Seeds use SHA-256 of master seed 42, split name, path, condition and severity. Metadata stores the split-list digest and preprocessing convention.

Generation reads path lists only; it does not decode official test images or save corrupted image copies. Commit these manifests together with their source split lists. Existing conflicting files are rejected, never silently replaced. Identical reruns are safe.

EvaluationPetsDataset checks manifest provenance and reconstructs the expected settings to detect changed parameters or incomplete coverage. It then applies each saved spec on access. This strict check intentionally ties version 1 manifests to the current corruption sampler; future algorithm changes require an explicit version/migration decision.

Because there are three severities per corruption but one clean case, an unweighted mean over all rows gives clean images only 10% of the weight. Report condition/severity results separately and explicitly choose condition-balanced aggregation when defining the validation objective. The classifier's exactly balanced training batches are a separate future requirement.
