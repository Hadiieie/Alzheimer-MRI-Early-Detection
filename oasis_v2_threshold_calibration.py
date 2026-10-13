"""OASIS V2 threshold selection on the already inspected development set.

Place this file in PROJECT_ROOT/Scripts and run from PROJECT_ROOT:
    python Scripts/oasis_v2_threshold_calibration.py

Only the BRAIN_MGZ PASS decision scores from the volume-source ablation are
used. This script reads predictions, never loads or changes the frozen model.
The selected threshold and all apparent metrics are DEVELOPMENT/CALIBRATION
ONLY, not external validation.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


LABEL = "DEVELOPMENT/CALIBRATION ONLY - NOT EXTERNAL VALIDATION"
SOURCE = "BRAIN_MGZ"
INPUT_RELATIVE = Path("results/OASIS_VOLUME_SOURCE_ABLATION/volume_source_predictions.csv")
OUTPUT_RELATIVE = Path("results/OASIS_THRESHOLD_CALIBRATION")


def load_cases(input_path):
    if not input_path.is_file():
        raise FileNotFoundError(f"Ablation predictions not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "source", "status", "ground_truth_binary", "decision_score", "threshold"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
        source_rows = [row for row in reader if row["source"].strip() == SOURCE]

    if not source_rows:
        raise ValueError(f"No {SOURCE} rows found in {input_path}")

    status_counts = Counter(row["status"].strip() for row in source_rows)
    cases = []
    seen = set()
    original_thresholds = set()
    for row in source_rows:
        if row["status"].strip() != "PASS":
            continue
        subject_id = row["subject_id"].strip()
        if not subject_id or subject_id in seen:
            raise ValueError(f"Missing or duplicate PASS subject ID: {subject_id!r}")
        seen.add(subject_id)
        label_text = row["ground_truth_binary"].strip()
        if label_text not in {"0", "1"}:
            raise ValueError(f"Invalid binary label for {subject_id}: {label_text!r}")
        try:
            score = float(row["decision_score"])
            original_threshold = float(row["threshold"])
        except ValueError as exc:
            raise ValueError(f"Invalid score or original threshold for {subject_id}") from exc
        if not math.isfinite(score) or not math.isfinite(original_threshold):
            raise ValueError(f"Non-finite score or original threshold for {subject_id}")
        cases.append((subject_id, int(label_text), score))
        original_thresholds.add(original_threshold)

    if len({label for _, label, _ in cases}) != 2:
        raise ValueError("Calibration needs PASS cases from both classes (0 and 1).")
    if len(original_thresholds) != 1:
        raise ValueError("PASS rows have inconsistent original frozen thresholds.")
    return cases, status_counts, original_thresholds.pop()


def metrics(cases, threshold):
    tn = fp = fn = tp = 0
    for _, actual, score in cases:
        predicted = int(score >= threshold)
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0:
            fp += 1
        elif predicted == 0:
            fn += 1
        else:
            tp += 1

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(cases)
    balanced_accuracy = (sensitivity + specificity) / 2
    return {
        "threshold": threshold,
        "balanced_accuracy": balanced_accuracy,
        "youden_j": sensitivity + specificity - 1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def candidate_thresholds(cases):
    """One threshold for every distinct prediction pattern under score >= threshold."""
    scores = sorted({score for _, _, score in cases})
    above_max = math.nextafter(scores[-1], math.inf)
    if not math.isfinite(above_max):
        raise ValueError("Cannot construct a finite threshold above the maximum score.")
    return scores + [above_max]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Folder containing Scripts and results (defaults to parent of this script's folder).",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    input_path = root / INPUT_RELATIVE
    cases, status_counts, original_threshold = load_cases(input_path)

    sweep = [metrics(cases, threshold) for threshold in candidate_thresholds(cases)]
    # For binary classes Youden J = 2 * balanced accuracy - 1. Ties favor
    # sensitivity, then specificity, then the lower threshold, deterministically.
    selected = max(
        sweep,
        key=lambda row: (
            row["balanced_accuracy"], row["youden_j"],
            row["sensitivity"], row["specificity"], -row["threshold"],
        ),
    )
    output_dir = root / OUTPUT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "threshold_sweep.csv"
    candidate_path = output_dir / "candidate_threshold.json"

    with sweep_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)

    report = {
        "analysis_label": LABEL,
        "model_version": "OASIS V2 candidate threshold for the unchanged frozen model",
        "source": SOURCE,
        "input_predictions": str(input_path),
        "selection_set": "Previously inspected OASIS development/calibration subjects; PASS only",
        "n_pass": len(cases),
        "class_counts": {
            "Normal_0": sum(label == 0 for _, label, _ in cases),
            "Early_Demented_1": sum(label == 1 for _, label, _ in cases),
        },
        "source_status_counts": dict(sorted(status_counts.items())),
        "n_excluded_non_pass": sum(count for status, count in status_counts.items() if status != "PASS"),
        "original_frozen_threshold_reference_only": original_threshold,
        "threshold_rule": "Predict Early_Demented (1) when decision_score >= threshold",
        "optimization": "Maximum balanced accuracy / Youden J; ties: sensitivity, specificity, lower threshold",
        "candidate_threshold": selected["threshold"],
        "development_metrics_at_candidate": {key: value for key, value in selected.items() if key != "threshold"},
        "confusion_matrix_labels_0_1": [
            [selected["TN"], selected["FP"]],
            [selected["FN"], selected["TP"]],
        ],
        "n_thresholds_swept": len(sweep),
        "threshold_sweep_csv": str(sweep_path),
        "interpretation": "Apparent performance on the same cases used to choose the threshold; an untouched new OASIS set is required for external validation.",
        "frozen_model_changed": False,
        "original_frozen_threshold_changed": False,
    }
    with candidate_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(LABEL)
    print(f"Input: {input_path}")
    print(f"{SOURCE} PASS cases: {len(cases)}; excluded non-PASS: {report['n_excluded_non_pass']}")
    print(f"Original frozen threshold (reference only): {original_threshold:.12g}")
    print(f"OASIS V2 candidate threshold: {selected['threshold']:.17g}")
    for name in ("balanced_accuracy", "youden_j", "sensitivity", "specificity", "precision", "f1", "accuracy"):
        print(f"{name}: {selected[name]:.6f}")
    print(f"Confusion matrix TN/FP/FN/TP: {selected['TN']}/{selected['FP']}/{selected['FN']}/{selected['TP']}")
    print(f"Threshold sweep: {sweep_path}")
    print(f"Candidate JSON: {candidate_path}")
    print("These selected-set metrics are not external validation.")


if __name__ == "__main__":
    main()
