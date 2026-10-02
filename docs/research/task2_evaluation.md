# Task 2 Hard Routing Evaluation Pipeline

## Purpose
`evaluation/evaluate_task2.py` implements the comparative validation analysis between Oracle routing and Predicted routing under the HardRouter architecture. It isolates the empirical degradation caused by classifier mistakes versus architectural capacity limitations of individual specialist autoencoders.

## Routing Mechanisms
1. **Predicted Routing**:
   The input image $x$ is classified by `CorruptionClassifier(x) \to p \in [0,1]^4$.
   The predicted label $\hat{y} = \arg\max(p)$ routes the image:
   - If $\hat{y} = 0$ (clean): Output is $x$ verbatim (exact clean identity preservation, no specialist invoked).
   - If $\hat{y} \in \{1, 2, 3\}$: Only the corresponding expert (`salt`, `blur`, or `occlusion`) processes the image.
2. **Oracle Routing**:
   The router receives ground truth labels $y \in \{0, 1, 2, 3\}$ from the validation manifest.
   The classifier is never called. Perfect condition routing establishes the upper-bound specialist performance.

## Routing Degradation & Taxonomy
Routing errors are quantified via:
$$\text{SSIM Routing Cost} = \text{SSIM}_{\text{oracle}} - \text{SSIM}_{\text{predicted}}$$
$$\text{L1 Routing Cost} = \text{L1}_{\text{predicted}} - \text{L1}_{\text{oracle}}$$

Three distinct failure failure modes are tracked:
- **Clean as Corrupted**: Clean input misclassified as a corruption condition. Triggers an expert on clean details, resulting in loss of sharp features.
- **Corrupted as Clean**: Corrupted image misclassified as clean. Bypasses restoration completely, producing zero gain.
- **Cross-Corruption**: Corrupted image routed to the wrong specialist (e.g. salt image processed by blur specialist), introducing inappropriate filtering artifacts.

## Outputs and Evidence
- `summary.json`: Detailed quantitative metrics, condition-balanced averages, and provenance hashes.
- `confusion_matrix.json`: Classification accuracy, per-class recall/precision/F1, and normalized confusion matrix.
- `by_condition.csv`: Direct tabular comparison between Oracle and Predicted metrics.
- `by_condition_severity.csv`: Disaggregated breakdown across 10 condition-severity combinations.
- `per_image.csv`: Per-case record with probabilities, predicted conditions, and routing costs.
- `representative_grid.png`: 12 representative cases across all four conditions (clean, salt, blur, occlusion).
- `routing_failures.png`: Visual panels isolating misrouted cases to inspect failure dynamics.
- `README.md`: Rendered markdown report with summary tables.

Official test images are strictly excluded; all analysis runs on the 2,944 fixed cases of `pets_val_corruptions.json`.
