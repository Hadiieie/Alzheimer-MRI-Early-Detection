from pathlib import Path
import json
import hashlib
from datetime import datetime
import joblib


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\97338\Desktop\RAW DATA")

RESULTS_DIR = PROJECT_ROOT / "results"
FINAL_MODEL_DIR = RESULTS_DIR / "FINAL_MODEL"
FINAL_PROJECT_DIR = RESULTS_DIR / "FINAL_PROJECT"

FINAL_PROJECT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FROZEN ARTIFACTS
# ============================================================

MODEL_FILE = FINAL_MODEL_DIR / "final_model.joblib"
SCALER_FILE = FINAL_MODEL_DIR / "final_scaler.joblib"
FEATURES_FILE = FINAL_MODEL_DIR / "final_features.txt"
CONFIG_FILE = FINAL_MODEL_DIR / "final_model_config.json"
FREEZE_MANIFEST_FILE = FINAL_MODEL_DIR / "freeze_manifest.json"

FROZEN_FILES = [
    MODEL_FILE,
    SCALER_FILE,
    FEATURES_FILE,
    CONFIG_FILE,
    FREEZE_MANIFEST_FILE,
]


# ============================================================
# 3. OFFICIALLY FROZEN PROJECT CONSTANTS
# ============================================================

PROJECT_NAME = "Alzheimer MRI Early-Detection System"

TASK_TYPE = "Binary classification"

POSITIVE_CLASS = "Early_Demented"
NEGATIVE_CLASS = "Normal"

EARLY_DEMENTED_CLASSES = [
    "VeryMildDemented",
    "MildDemented",
]

EXCLUDED_FROM_PRIMARY_TASK = [
    "ModerateDemented",
]

FULL_CLEAN_DATASET_SIZE = 39538
PRIMARY_DATASET_SIZE = 29853

TRAIN_SIZE = 20891
VALIDATION_SIZE = 4482
TEST_SIZE = 4480

FROZEN_THRESHOLD = 0.10872472921801979

EXPECTED_MODEL_NAME = "SVM RBF"

EXPECTED_NUM_FEATURES = 30


# ============================================================
# 4. FINAL TEST RESULTS
# ============================================================

FINAL_TEST_METRICS = {
    "test_images": 4480,
    "threshold": FROZEN_THRESHOLD,

    "accuracy": 0.812054,
    "balanced_accuracy": 0.810185,

    "sensitivity_recall": 0.815834,
    "specificity": 0.804536,

    "precision": 0.892477,
    "f1": 0.852436,

    "roc_auc": 0.887921,
    "pr_auc": 0.929306,
}

FINAL_CONFUSION_MATRIX = {
    "TN": 1206,
    "FP": 293,
    "FN": 549,
    "TP": 2432,
}


# ============================================================
# 5. GROUP-AWARE BOOTSTRAP 95% CONFIDENCE INTERVALS
# ============================================================

BOOTSTRAP_RESULTS = {
    "bootstrap_samples": 2000,
    "bootstrap_type": "GROUP-AWARE",

    "accuracy": {
        "point_estimate": 0.812054,
        "ci_95_lower": 0.799774,
        "ci_95_upper": 0.823596,
    },

    "balanced_accuracy": {
        "point_estimate": 0.810185,
        "ci_95_lower": 0.797359,
        "ci_95_upper": 0.822235,
    },

    "sensitivity": {
        "point_estimate": 0.815834,
        "ci_95_lower": 0.801290,
        "ci_95_upper": 0.830681,
    },

    "specificity": {
        "point_estimate": 0.804536,
        "ci_95_lower": 0.783018,
        "ci_95_upper": 0.823897,
    },

    "precision": {
        "point_estimate": 0.892477,
        "ci_95_lower": 0.879939,
        "ci_95_upper": 0.904361,
    },

    "f1": {
        "point_estimate": 0.852436,
        "ci_95_lower": 0.842068,
        "ci_95_upper": 0.862555,
    },

    "roc_auc": {
        "point_estimate": 0.887921,
        "ci_95_lower": 0.877116,
        "ci_95_upper": 0.898177,
    },

    "pr_auc": {
        "point_estimate": 0.929306,
        "ci_95_lower": 0.918980,
        "ci_95_upper": 0.938399,
    },
}


# ============================================================
# 6. SUBGROUP TEST RESULTS
# ============================================================

SUBGROUP_RESULTS = {
    "VeryMildDemented": {
        "test_images": 1537,
        "correct": 1110,
        "errors": 427,
        "false_negatives": 427,
        "sensitivity": 0.722186,
        "false_negative_rate": 0.27781392,
    },

    "MildDemented": {
        "test_images": 1444,
        "correct": 1322,
        "errors": 122,
        "false_negatives": 122,
        "sensitivity": 0.915512,
        "false_negative_rate": 0.08448753,
    },

    "NonDemented": {
        "test_images": 1499,
        "correct": 1206,
        "errors": 293,
        "false_positives": 293,
        "specificity": 0.804536,
        "false_positive_rate": 0.195464,
    },
}


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def sha256_file(file_path: Path):
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def load_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_threshold(config):
    """
    Search several possible keys because previous configuration
    versions used different threshold names.
    """

    candidate_keys = [
        "threshold",
        "selected_threshold",
        "frozen_threshold",
        "final_threshold",
        "decision_threshold",
    ]

    for key in candidate_keys:
        if key in config:
            try:
                return float(config[key]), key
            except (TypeError, ValueError):
                pass

    return None, None


# ============================================================
# 8. VERIFY FROZEN ARTIFACTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT FREEZE")
print("=" * 70)

print("\n1. VERIFYING FROZEN MODEL ARTIFACTS")
print("-" * 70)

missing_files = []

for path in FROZEN_FILES:

    if path.exists():
        print(f"FOUND: {path.name}")
    else:
        print(f"MISSING: {path}")
        missing_files.append(path)

if missing_files:
    print("\nERROR:")
    print("One or more required frozen artifacts are missing.")
    print("No final project freeze will be created.")

    for path in missing_files:
        print(" -", path)

    raise FileNotFoundError(
        "Required frozen model artifacts are missing."
    )

print("\nAll required frozen artifacts are present.")


# ============================================================
# 9. VERIFY MODEL CAN BE LOADED
# ============================================================

print("\n" + "=" * 70)
print("2. MODEL LOAD CHECK")
print("=" * 70)

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Object type:")
print(type(model))

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")

    for step_name in model.named_steps.keys():
        print(" -", step_name)

print("\ndecision_function():",
      "AVAILABLE" if hasattr(model, "decision_function") else "NOT AVAILABLE")

print("predict():",
      "AVAILABLE" if hasattr(model, "predict") else "NOT AVAILABLE")


# ============================================================
# 10. LOAD AND VERIFY FEATURE LIST
# ============================================================

print("\n" + "=" * 70)
print("3. FEATURE LIST CHECK")
print("=" * 70)

with open(FEATURES_FILE, "r", encoding="utf-8") as f:
    features = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

print("Number of frozen features:")
print(len(features))

for i, feature in enumerate(features, start=1):
    print(f"{i:02d}. {feature}")

if len(features) != EXPECTED_NUM_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_NUM_FEATURES} frozen features, "
        f"but found {len(features)}."
    )

print("\nFEATURE COUNT: PASS")


# ============================================================
# 11. CONFIGURATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("4. CONFIGURATION CHECK")
print("=" * 70)

config = load_json(CONFIG_FILE)

config_threshold, threshold_key = find_threshold(config)

print("Configuration file loaded.")

if config_threshold is not None:

    print("\nThreshold key found:")
    print(threshold_key)

    print("Threshold stored in config:")
    print(config_threshold)

    print("Expected frozen threshold:")
    print(FROZEN_THRESHOLD)

    difference = abs(config_threshold - FROZEN_THRESHOLD)

    print("Absolute difference:")
    print(difference)

    if difference <= 1e-10:
        print("\nTHRESHOLD CONSISTENCY: PASS")
    else:
        raise ValueError(
            "Threshold in final_model_config.json does not match "
            "the officially frozen threshold."
        )

else:

    print("\nNOTE:")
    print(
        "No direct threshold field was found in final_model_config.json."
    )

    print(
        "The officially frozen threshold will therefore be recorded "
        "from the validated project result:"
    )

    print(FROZEN_THRESHOLD)


# ============================================================
# 12. COMPUTE SHA256 CHECKSUMS
# ============================================================

print("\n" + "=" * 70)
print("5. SHA256 CHECKSUMS")
print("=" * 70)

checksums = {}

for path in FROZEN_FILES:

    checksum = sha256_file(path)
    checksums[path.name] = checksum

    print(f"{path.name}")
    print(checksum)
    print()


# ============================================================
# 13. SAVE FINAL METRICS JSON
# ============================================================

metrics_output = {
    "project": PROJECT_NAME,
    "task": TASK_TYPE,

    "final_test_metrics": FINAL_TEST_METRICS,
    "confusion_matrix": FINAL_CONFUSION_MATRIX,

    "bootstrap_95_ci": BOOTSTRAP_RESULTS,
    "subgroup_results": SUBGROUP_RESULTS,
}

FINAL_METRICS_FILE = FINAL_PROJECT_DIR / "final_metrics.json"

with open(FINAL_METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics_output, f, indent=4)

print("Saved:")
print(FINAL_METRICS_FILE)


# ============================================================
# 14. SAVE CONFUSION MATRIX JSON
# ============================================================

CONFUSION_FILE = (
    FINAL_PROJECT_DIR /
    "final_test_confusion_matrix.json"
)

with open(CONFUSION_FILE, "w", encoding="utf-8") as f:
    json.dump(
        {
            "test_images": TEST_SIZE,
            "threshold": FROZEN_THRESHOLD,
            "matrix": FINAL_CONFUSION_MATRIX,
        },
        f,
        indent=4,
    )

print("Saved:")
print(CONFUSION_FILE)


# ============================================================
# 15. FINAL PROJECT MANIFEST
# ============================================================

project_manifest = {

    "project_name": PROJECT_NAME,

    "project_status": "FROZEN",

    "freeze_timestamp_local": datetime.now().isoformat(),

    "task": {
        "type": TASK_TYPE,
        "negative_class": NEGATIVE_CLASS,
        "positive_class": POSITIVE_CLASS,
        "positive_original_classes": EARLY_DEMENTED_CLASSES,
        "excluded_from_primary_task": EXCLUDED_FROM_PRIMARY_TASK,
    },

    "dataset": {
        "original_images": 40000,
        "clean_images": FULL_CLEAN_DATASET_SIZE,
        "primary_dataset_images": PRIMARY_DATASET_SIZE,

        "train_images": TRAIN_SIZE,
        "validation_images": VALIDATION_SIZE,
        "test_images": TEST_SIZE,

        "group_aware_split": True,
        "train_validation_overlap": 0,
        "train_test_overlap": 0,
        "validation_test_overlap": 0,
    },

    "preprocessing": {
        "approach": "Whole-brain",
        "roi_v1_v2_status": "Rejected scientifically",
        "brain_mask_version": "V4",
        "brain_mask_full_audit_pass": FULL_CLEAN_DATASET_SIZE,
    },

    "features": {
        "number_of_features": len(features),
        "feature_names": features,
    },

    "model": {
        "model_name": EXPECTED_MODEL_NAME,
        "model_file": str(MODEL_FILE),
        "scaler_file": str(SCALER_FILE),
        "feature_file": str(FEATURES_FILE),
        "configuration_file": str(CONFIG_FILE),

        "frozen_threshold": FROZEN_THRESHOLD,

        "model_selection_used_test_set": False,
        "threshold_selection_used_test_set": False,
        "test_set_opened_after_freeze": True,
    },

    "final_test": FINAL_TEST_METRICS,

    "confusion_matrix": FINAL_CONFUSION_MATRIX,

    "checksums_sha256": checksums,

    "scientific_limitations": [
        (
            "Verified patient identifiers were not available. "
            "Leakage control therefore relied on exact-duplicate and "
            "near-duplicate/group-based image analysis rather than "
            "confirmed subject-level splitting."
        ),
        (
            "External validation on an independent MRI dataset has "
            "not yet been performed."
        ),
        (
            "The primary task is binary early-detection classification "
            "rather than full four-stage dementia classification."
        ),
        (
            "The model is intended for research and educational use "
            "and is not a clinical diagnostic medical device."
        ),
    ],

    "freeze_policy": {
        "model_must_not_change": True,
        "feature_set_must_not_change": True,
        "threshold_must_not_change": True,
        "test_set_must_not_be_used_for_tuning": True,
    },
}

MANIFEST_FILE = FINAL_PROJECT_DIR / "project_manifest.json"

with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(project_manifest, f, indent=4)

print("Saved:")
print(MANIFEST_FILE)


# ============================================================
# 16. SHA256 CHECKSUM TEXT FILE
# ============================================================

CHECKSUM_FILE = FINAL_PROJECT_DIR / "SHA256_CHECKSUMS.txt"

with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:

    f.write("FINAL FROZEN MODEL ARTIFACT CHECKSUMS\n")
    f.write("=" * 70 + "\n\n")

    for filename, checksum in checksums.items():
        f.write(f"{checksum}  {filename}\n")

print("Saved:")
print(CHECKSUM_FILE)


# ============================================================
# 17. MODEL CARD
# ============================================================

model_card = f"""# Alzheimer MRI Early-Detection System

## Project status

FINAL FROZEN MODEL

This document describes the final frozen machine-learning system.
The model, feature set, scaler and decision threshold must not be
modified based on final-test results.

---

## 1. Task

Binary MRI classification:

- Normal
- Early_Demented

Early_Demented consists of:

- VeryMildDemented
- MildDemented

ModerateDemented is retained in the wider cleaned dataset but is not
part of the primary early-detection binary task.

---

## 2. Dataset

Original dataset:

40,000 MRI images

After exact duplicate cleaning:

{FULL_CLEAN_DATASET_SIZE:,} unique images

Primary early-detection dataset:

{PRIMARY_DATASET_SIZE:,} images

Group-aware data split:

- Train: {TRAIN_SIZE:,}
- Validation: {VALIDATION_SIZE:,}
- Test: {TEST_SIZE:,}

Leakage-group overlap:

- Train vs Validation: 0
- Train vs Test: 0
- Validation vs Test: 0

The final test set remained locked during model selection,
hyperparameter tuning and threshold selection.

---

## 3. Image-processing methodology

The final system uses the whole-brain approach.

ROI V1/V2 experiments were rejected during methodological development.

Final brain-mask pipeline:

Brain Mask V4

Full mask audit:

{FULL_CLEAN_DATASET_SIZE:,} / {FULL_CLEAN_DATASET_SIZE:,} PASS

---

## 4. Feature engineering

Number of numerical features:

{len(features)}

Frozen feature set:

"""

for i, feature in enumerate(features, start=1):
    model_card += f"{i}. {feature}\n"


model_card += f"""

---

## 5. Final classifier

Model:

{EXPECTED_MODEL_NAME}

Frozen decision threshold:

{FROZEN_THRESHOLD:.17f}

The threshold was selected using the validation set only.

The test set was not used for:

- feature selection
- hyperparameter tuning
- model selection
- threshold selection

---

## 6. Final one-time test results

Test images:

{TEST_SIZE:,}

Accuracy:

{FINAL_TEST_METRICS["accuracy"]:.6f}

Balanced Accuracy:

{FINAL_TEST_METRICS["balanced_accuracy"]:.6f}

Sensitivity / Recall:

{FINAL_TEST_METRICS["sensitivity_recall"]:.6f}

Specificity:

{FINAL_TEST_METRICS["specificity"]:.6f}

Precision:

{FINAL_TEST_METRICS["precision"]:.6f}

F1 Score:

{FINAL_TEST_METRICS["f1"]:.6f}

ROC-AUC:

{FINAL_TEST_METRICS["roc_auc"]:.6f}

PR-AUC:

{FINAL_TEST_METRICS["pr_auc"]:.6f}

Confusion Matrix:

- TN = {FINAL_CONFUSION_MATRIX["TN"]}
- FP = {FINAL_CONFUSION_MATRIX["FP"]}
- FN = {FINAL_CONFUSION_MATRIX["FN"]}
- TP = {FINAL_CONFUSION_MATRIX["TP"]}

---

## 7. Group-aware bootstrap robustness

Bootstrap repetitions:

{BOOTSTRAP_RESULTS["bootstrap_samples"]}

Bootstrap type:

GROUP-AWARE

95% Confidence Intervals:

Accuracy:
[{BOOTSTRAP_RESULTS["accuracy"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["accuracy"]["ci_95_upper"]:.6f}]

Balanced Accuracy:
[{BOOTSTRAP_RESULTS["balanced_accuracy"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["balanced_accuracy"]["ci_95_upper"]:.6f}]

Sensitivity:
[{BOOTSTRAP_RESULTS["sensitivity"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["sensitivity"]["ci_95_upper"]:.6f}]

Specificity:
[{BOOTSTRAP_RESULTS["specificity"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["specificity"]["ci_95_upper"]:.6f}]

Precision:
[{BOOTSTRAP_RESULTS["precision"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["precision"]["ci_95_upper"]:.6f}]

F1:
[{BOOTSTRAP_RESULTS["f1"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["f1"]["ci_95_upper"]:.6f}]

ROC-AUC:
[{BOOTSTRAP_RESULTS["roc_auc"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["roc_auc"]["ci_95_upper"]:.6f}]

PR-AUC:
[{BOOTSTRAP_RESULTS["pr_auc"]["ci_95_lower"]:.6f},
 {BOOTSTRAP_RESULTS["pr_auc"]["ci_95_upper"]:.6f}]

---

## 8. Original-class subgroup performance

### MildDemented

Test images: 1444

False negatives: 122

Sensitivity:

0.915512

### VeryMildDemented

Test images: 1537

False negatives: 427

Sensitivity:

0.722186

This subgroup was substantially more difficult for the model than
MildDemented.

### NonDemented

Test images: 1499

False positives: 293

Specificity:

0.804536

---

## 9. Interpretability and error analysis

The completed project includes:

- permutation feature importance
- representative local feature-impact analysis
- true-positive analysis
- true-negative analysis
- false-positive analysis
- false-negative analysis
- borderline-case analysis
- error distance from the frozen threshold
- original-class error analysis
- threshold robustness analysis

The final model was not altered after these analyses.

---

## 10. Important scientific limitations

1. Verified patient identifiers were not available in the source
   dataset.

   Therefore, leakage prevention used exact-duplicate,
   near-duplicate and image-group analysis rather than confirmed
   patient-level splitting.

2. External validation on an independent patient-level MRI dataset
   has not yet been performed.

3. The primary model performs binary early-detection classification,
   not complete four-stage dementia classification.

4. Hand-engineered MRI features are used rather than an end-to-end
   deep-learning representation.

---

## 11. Intended use

This project is intended for:

- machine-learning research
- biomedical-engineering research
- educational experimentation

It is NOT a clinically validated diagnostic medical device and must
not be used independently to diagnose Alzheimer disease.

---

## 12. Frozen-model policy

After the final model freeze:

- DO NOT retrain using the test set.
- DO NOT tune the decision threshold using the test set.
- DO NOT select features using the test set.
- DO NOT modify model hyperparameters because of test results.
- DO NOT replace the frozen artifacts without creating a new
  explicitly versioned experiment.

Future independent datasets may be used for external validation.

"""

MODEL_CARD_FILE = FINAL_PROJECT_DIR / "model_card.md"

with open(MODEL_CARD_FILE, "w", encoding="utf-8") as f:
    f.write(model_card)

print("Saved:")
print(MODEL_CARD_FILE)


# ============================================================
# 18. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("6. FINAL PROJECT PACKAGE")
print("=" * 70)

expected_outputs = [
    FINAL_METRICS_FILE,
    CONFUSION_FILE,
    MANIFEST_FILE,
    CHECKSUM_FILE,
    MODEL_CARD_FILE,
]

all_ok = True

for path in expected_outputs:

    status = "PASS" if path.exists() else "FAIL"

    print(f"{status}: {path.name}")

    if not path.exists():
        all_ok = False


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

if all_ok:

    print("FINAL PROJECT FREEZE: PASS")

    print("\nFrozen model:")
    print(MODEL_FILE)

    print("\nFrozen threshold:")
    print(FROZEN_THRESHOLD)

    print("\nFrozen features:")
    print(len(features))

    print("\nFinal project documentation:")
    print(FINAL_PROJECT_DIR)

    print("\nIMPORTANT:")
    print(
        "The model, scaler, feature set and threshold are now "
        "scientifically frozen."
    )

    print(
        "Do not modify them based on final-test results."
    )

    print("\nNEXT STEP:")
    print("BUILD FINAL SINGLE-MRI INFERENCE PIPELINE")

else:

    print("FINAL PROJECT FREEZE: FAIL")
    print("One or more output files were not created.")

print("=" * 70)