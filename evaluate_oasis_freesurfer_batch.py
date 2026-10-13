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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from predict_mri import predict_single_mri


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    PROJECT_ROOT
    / "NEW MRI"
    / "OASIS_BATCH"
    / "oasis_validation_manifest.csv"
)

EXTRACTION_SUMMARY = (
    PROJECT_ROOT
    / "NEW MRI"
    / "OASIS_FREESURFER_BATCH"
    / "freesurfer_slice_extraction_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "OASIS_FREESURFER_EVALUATION"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_CSV = OUTPUT_DIR / "oasis_freesurfer_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "oasis_freesurfer_metrics.json"


# ============================================================
# LOAD INPUT TABLES
# ============================================================

print("\n" + "=" * 72)
print("OASIS FREESURFER CHALLENGE-SET EVALUATION")
print("=" * 72)

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST}"
    )

if not EXTRACTION_SUMMARY.exists():
    raise FileNotFoundError(
        f"Extraction summary not found:\n{EXTRACTION_SUMMARY}"
    )

manifest = pd.read_csv(MANIFEST)
extraction = pd.read_csv(EXTRACTION_SUMMARY)

required_manifest = {"subject_id"}
if not required_manifest.issubset(manifest.columns):
    raise KeyError(
        f"Manifest must contain subject_id. "
        f"Columns found: {list(manifest.columns)}"
    )

# Flexible ground-truth column support.
if "ground_truth" in manifest.columns:
    gt_col = "ground_truth"
elif "target_name" in manifest.columns:
    gt_col = "target_name"
else:
    raise KeyError(
        "Manifest must contain either 'ground_truth' or 'target_name'. "
        f"Columns found: {list(manifest.columns)}"
    )

required_extraction = {
    "subject_id",
    "status",
    "output_png",
}
if not required_extraction.issubset(extraction.columns):
    raise KeyError(
        "Extraction summary is missing required columns. "
        f"Columns found: {list(extraction.columns)}"
    )

merged = manifest.merge(
    extraction,
    on="subject_id",
    how="left",
    suffixes=("", "_extract"),
)

print("\nManifest cases:")
print(len(manifest))

print("\nExtraction status:")
print(
    merged["status"]
    .fillna("MISSING")
    .value_counts()
)

runnable = merged[
    merged["status"].astype(str).str.upper() == "PASS"
].copy()

print("\nCases available for frozen inference:")
print(len(runnable))

if len(runnable) == 0:
    raise RuntimeError(
        "No successfully extracted FreeSurfer PNG cases are available."
    )


# ============================================================
# LABEL MAPPING
# ============================================================

def normalize_ground_truth(value):
    text = str(value).strip().lower()

    if text in {
        "normal",
        "nondemented",
        "non_demented",
        "0",
    }:
        return "Normal", 0

    if text in {
        "early_demented",
        "early-demented",
        "verymilddemented",
        "milddemented",
        "very_mild_demented",
        "mild_demented",
        "1",
    }:
        return "Early_Demented", 1

    raise ValueError(
        f"Unsupported ground-truth label: {value}"
    )


# ============================================================
# RUN FROZEN INFERENCE
# ============================================================

records = []

print("\n" + "=" * 72)
print("RUNNING FROZEN MODEL")
print("=" * 72)

for i, (_, row) in enumerate(
    runnable.iterrows(),
    start=1,
):
    subject_id = str(row["subject_id"]).strip()
    image_path = Path(str(row["output_png"]))

    gt_name, gt_binary = normalize_ground_truth(
        row[gt_col]
    )

    cdr = (
        row["CDR"]
        if "CDR" in row.index
        else np.nan
    )

    print(
        f"\n[{i:02d}/{len(runnable):02d}] "
        f"{subject_id} | CDR={cdr} | GT={gt_name}"
    )

    try:
        report, case_dir = predict_single_mri(
            image_path
        )

        prediction = str(
            report["prediction"]
        )

        pred_binary = int(
            report["prediction_binary"]
        )

        decision_score = float(
            report["decision_score"]
        )

        threshold = float(
            report["frozen_threshold"]
        )

        margin = float(
            report["margin_from_threshold"]
        )

        correct = bool(
            pred_binary == gt_binary
        )

        print("Prediction:", prediction)
        print("Decision score:", f"{decision_score:.6f}")
        print("Margin:", f"{margin:+.6f}")
        print("Correct:", correct)

        records.append({
            "subject_id":
                subject_id,

            "CDR":
                cdr,

            "ground_truth":
                gt_name,

            "ground_truth_binary":
                gt_binary,

            "image_path":
                str(image_path),

            "prediction":
                prediction,

            "prediction_binary":
                pred_binary,

            "decision_score":
                decision_score,

            "threshold":
                threshold,

            "margin":
                margin,

            "correct":
                correct,

            "status":
                "PASS",

            "error":
                "",
        })

    except Exception as e:
        print("STATUS: ERROR")
        print("Reason:", e)

        records.append({
            "subject_id":
                subject_id,

            "CDR":
                cdr,

            "ground_truth":
                gt_name,

            "ground_truth_binary":
                gt_binary,

            "image_path":
                str(image_path),

            "prediction":
                "",

            "prediction_binary":
                np.nan,

            "decision_score":
                np.nan,

            "threshold":
                np.nan,

            "margin":
                np.nan,

            "correct":
                False,

            "status":
                "ERROR",

            "error":
                str(e),
        })


results = pd.DataFrame(
    records
)

results.to_csv(
    RESULTS_CSV,
    index=False
)


# ============================================================
# METRICS
# ============================================================

successful = results[
    results["status"] == "PASS"
].copy()

print("\n" + "=" * 72)
print("BATCH COMPLETION")
print("=" * 72)

print(
    results["status"]
    .value_counts(
        dropna=False
    )
)

if len(successful) == 0:
    raise RuntimeError(
        "No successful inference cases."
    )

y_true = successful[
    "ground_truth_binary"
].astype(int).to_numpy()

y_pred = successful[
    "prediction_binary"
].astype(int).to_numpy()

scores = successful[
    "decision_score"
].astype(float).to_numpy()

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()

accuracy = accuracy_score(
    y_true,
    y_pred,
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred,
)

sensitivity = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

roc_auc = (
    roc_auc_score(
        y_true,
        scores,
    )
    if len(np.unique(y_true)) == 2
    else np.nan
)

pr_auc = (
    average_precision_score(
        y_true,
        scores,
    )
    if len(np.unique(y_true)) == 2
    else np.nan
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 72)
print("OASIS FREESURFER RESULTS")
print("=" * 72)

print("\nEvaluated cases:")
print(len(successful))

print("\nGround-truth distribution:")
print(
    successful[
        "ground_truth"
    ].value_counts()
)

print("\nAccuracy:")
print(f"{accuracy:.6f}")

print("\nBalanced Accuracy:")
print(f"{balanced_accuracy:.6f}")

print("\nSensitivity / Recall:")
print(f"{sensitivity:.6f}")

print("\nSpecificity:")
print(f"{specificity:.6f}")

print("\nPrecision:")
print(f"{precision:.6f}")

print("\nF1:")
print(f"{f1:.6f}")

print("\nROC-AUC:")
print(f"{roc_auc:.6f}")

print("\nPR-AUC:")
print(f"{pr_auc:.6f}")

print("\nConfusion Matrix:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")


# ============================================================
# OPTIONAL BREAKDOWN BY ORIGINAL CDR
# ============================================================

if "CDR" in successful.columns:
    print("\n" + "=" * 72)
    print("RESULTS BY CDR")
    print("=" * 72)

    cdr_summary = (
        successful
        .groupby("CDR", dropna=False)
        .agg(
            n=("subject_id", "count"),
            correct=("correct", "sum"),
            mean_score=("decision_score", "mean"),
            median_score=("decision_score", "median"),
        )
    )

    cdr_summary["accuracy"] = (
        cdr_summary["correct"]
        /
        cdr_summary["n"]
    )

    print(cdr_summary)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "evaluation_name":
        "OASIS FreeSurfer challenge-set evaluation",

    "requested_manifest_cases":
        int(len(manifest)),

    "successfully_extracted_cases":
        int(len(runnable)),

    "successfully_inferred_cases":
        int(len(successful)),

    "inference_failures":
        int(
            (results["status"] != "PASS").sum()
        ),

    "class_distribution": {
        str(k): int(v)
        for k, v in successful[
            "ground_truth"
        ].value_counts().to_dict().items()
    },

    "accuracy":
        float(accuracy),

    "balanced_accuracy":
        float(balanced_accuracy),

    "sensitivity":
        float(sensitivity),

    "specificity":
        float(specificity),

    "precision":
        float(precision),

    "f1":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "pr_auc":
        float(pr_auc),

    "confusion_matrix": {
        "TN":
            int(tn),
        "FP":
            int(fp),
        "FN":
            int(fn),
        "TP":
            int(tp),
    },

    "note":
        (
            "Challenge-set/pilot evaluation using OASIS-1 FreeSurfer "
            "brain-extracted volumes and a frozen deterministic 50%-of-"
            "occupied-brain-height axial-slice rule. One requested subject "
            "may be missing from the downloaded archive. This evaluation "
            "does not modify the frozen model, feature set, preprocessing "
            "pipeline, Brain Mask V4, or threshold."
        ),
}

with open(
    METRICS_JSON,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        indent=4,
    )

print("\n" + "=" * 72)
print("FILES SAVED")
print("=" * 72)

print("\nPredictions:")
print(RESULTS_CSV)

print("\nMetrics:")
print(METRICS_JSON)

print("\nIMPORTANT:")
print(
    "The frozen model, frozen threshold, Brain Mask V4, "
    "and 30-feature pipeline were not changed."
)

print("=" * 72)
