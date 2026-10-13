from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from predict_mri import predict_single_mri

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = PROJECT_ROOT / "NEW MRI" / "OASIS_BATCH"
MANIFEST_FILE = BATCH_DIR / "oasis_validation_manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "OASIS_BATCH_VALIDATION"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"]


def find_image(subject_id):
    for ext in SUPPORTED_EXTENSIONS:
        candidate = BATCH_DIR / f"{subject_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def label_to_int(label):
    return 1 if label == "Early_Demented" else 0


print("\n" + "=" * 72)
print("OASIS BATCH PILOT VALIDATION")
print("=" * 72)

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_FILE}\n\n"
        "Place oasis_validation_manifest.csv inside NEW MRI\\OASIS_BATCH."
    )

manifest = pd.read_csv(MANIFEST_FILE)
required = ["subject_id", "CDR", "ground_truth"]
missing = [c for c in required if c not in manifest.columns]
if missing:
    raise KeyError("Manifest missing columns: " + ", ".join(missing))

print(f"\nLocked manifest cases: {len(manifest)}")
print(manifest["ground_truth"].value_counts())

records = []

for i, row in manifest.iterrows():
    subject_id = str(row["subject_id"])
    ground_truth = str(row["ground_truth"])
    cdr = float(row["CDR"])

    print("\n" + "-" * 72)
    print(f"[{i+1:02d}/{len(manifest):02d}] {subject_id} | CDR={cdr} | GT={ground_truth}")

    image_path = find_image(subject_id)
    if image_path is None:
        print("MISSING IMAGE")
        records.append({
            "subject_id": subject_id,
            "CDR": cdr,
            "ground_truth": ground_truth,
            "image_path": "",
            "prediction": "",
            "decision_score": np.nan,
            "margin": np.nan,
            "correct": np.nan,
            "status": "MISSING_IMAGE",
        })
        continue

    try:
        report, case_dir = predict_single_mri(image_path)
        pred = report["prediction"]
        score = float(report["decision_score"])
        margin = float(report["margin_from_threshold"])
        correct = bool(pred == ground_truth)

        print(f"Prediction: {pred}")
        print(f"Decision score: {score:.6f}")
        print(f"Margin: {margin:+.6f}")
        print("Correct:", correct)

        records.append({
            "subject_id": subject_id,
            "CDR": cdr,
            "ground_truth": ground_truth,
            "image_path": str(image_path),
            "prediction": pred,
            "decision_score": score,
            "margin": margin,
            "correct": correct,
            "status": "PASS",
        })

    except Exception as e:
        print("ERROR:", e)
        records.append({
            "subject_id": subject_id,
            "CDR": cdr,
            "ground_truth": ground_truth,
            "image_path": str(image_path),
            "prediction": "",
            "decision_score": np.nan,
            "margin": np.nan,
            "correct": np.nan,
            "status": "ERROR",
            "error": str(e),
        })

results = pd.DataFrame(records)
results_file = RESULTS_DIR / "oasis_batch_results.csv"
results.to_csv(results_file, index=False)

print("\n" + "=" * 72)
print("BATCH COMPLETION")
print("=" * 72)
print(results["status"].value_counts(dropna=False))

complete = bool((results["status"] == "PASS").all() and len(results) == len(manifest))

if not complete:
    print("\nBATCH IS INCOMPLETE.")
    print("No final validation metrics will be reported until all locked cases run successfully.")
    print("\nResults saved to:")
    print(results_file)
    raise SystemExit(0)

# Final locked-set metrics
true_labels = np.array([label_to_int(x) for x in results["ground_truth"]], dtype=int)
pred_labels = np.array([label_to_int(x) for x in results["prediction"]], dtype=int)
scores = results["decision_score"].to_numpy(dtype=float)

tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()

sensitivity = recall_score(true_labels, pred_labels, pos_label=1)
specificity = tn / (tn + fp) if (tn + fp) else np.nan

metrics = {
    "n_cases": int(len(results)),
    "n_normal": int((true_labels == 0).sum()),
    "n_early_demented": int((true_labels == 1).sum()),
    "accuracy": float(accuracy_score(true_labels, pred_labels)),
    "balanced_accuracy": float(balanced_accuracy_score(true_labels, pred_labels)),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "precision": float(precision_score(true_labels, pred_labels, zero_division=0)),
    "f1": float(f1_score(true_labels, pred_labels, zero_division=0)),
    "roc_auc": float(roc_auc_score(true_labels, scores)),
    "pr_auc": float(average_precision_score(true_labels, scores)),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "note": (
        "Pilot OASIS challenge-set evaluation. Treat as fully independent external validation "
        "only if independence from the original training-data provenance is established."
    ),
}

metrics_file = RESULTS_DIR / "oasis_batch_metrics.json"
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\n" + "=" * 72)
print("OASIS PILOT METRICS")
print("=" * 72)
for key in [
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "precision", "f1", "roc_auc", "pr_auc"
]:
    print(f"{key}: {metrics[key]:.6f}")

print("\nConfusion Matrix")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print("\nResults saved to:")
print(results_file)
print(metrics_file)
