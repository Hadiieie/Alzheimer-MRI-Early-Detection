# ============================================================
# ONE-TIME FINAL TEST EVALUATION
# Alzheimer MRI Early Detection Project
# ============================================================
#
# FINAL frozen system:
#   Model      : Tuned SVM RBF
#   Features   : FULL_30
#   Threshold  : frozen from VALIDATION only
#   Scaling    : INSIDE sklearn Pipeline
#
# IMPORTANT:
#   - NO hyperparameter tuning
#   - NO threshold optimization
#   - NO feature selection
#   - NO model comparison
#   - TEST is evaluated only with the frozen system
#
# ============================================================

from pathlib import Path
import hashlib
import json
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

FINAL_MODEL_DIR = (
    RESULTS_DIR
    / "FINAL_MODEL"
)

GROUP_SPLIT_DIR = (
    RESULTS_DIR
    / "GROUP_AWARE_SPLIT"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "FINAL_TEST_EVALUATION"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

MODEL_FILE = (
    FINAL_MODEL_DIR
    / "final_model.joblib"
)

FEATURE_FILE = (
    FINAL_MODEL_DIR
    / "final_features.txt"
)

CONFIG_FILE = (
    FINAL_MODEL_DIR
    / "final_model_config.json"
)

MANIFEST_FILE = (
    FINAL_MODEL_DIR
    / "freeze_manifest.json"
)

TEST_FILE = (
    GROUP_SPLIT_DIR
    / "test.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

METRICS_FILE = (
    OUTPUT_DIR
    / "final_test_metrics.json"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "final_test_predictions.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "final_test_report.txt"
)

CONFUSION_MATRIX_FILE = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)

ROC_CURVE_FILE = (
    OUTPUT_DIR
    / "roc_curve.png"
)

PR_CURVE_FILE = (
    OUTPUT_DIR
    / "precision_recall_curve.png"
)

TEST_HASH_FILE = (
    OUTPUT_DIR
    / "test_file_sha256.txt"
)


# ============================================================
# 4. EXPECTED TEST SPLIT
#    From the already-created group-aware split
# ============================================================

EXPECTED_TEST_ROWS = 4480

EXPECTED_BINARY_COUNTS = {
    "Early_Demented": 2981,
    "Normal": 1499,
}

EXPECTED_ORIGINAL_COUNTS = {
    "VeryMildDemented": 1537,
    "MildDemented": 1444,
    "NonDemented": 1499,
}


# ============================================================
# 5. UTILITY: SHA256
# ============================================================

def sha256_file(file_path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 6. HEADER
# ============================================================

print("\n")
print("=" * 78)
print("ONE-TIME FINAL TEST EVALUATION")
print("=" * 78)

print("\nProject:")
print("Alzheimer MRI Early Detection")

print("\nIMPORTANT:")
print("This script does NOT tune anything.")

print("\nFrozen model:")
print(MODEL_FILE)

print("\nTest split:")
print(TEST_FILE)


# ============================================================
# 7. VERIFY FROZEN FILES BEFORE OPENING TEST
# ============================================================

print("\n")
print("=" * 78)
print("PRE-TEST INTEGRITY CHECK")
print("=" * 78)


required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    CONFIG_FILE,
    MANIFEST_FILE,
    TEST_FILE,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print(f"FOUND: {path.name}")


# ============================================================
# 8. LOAD FROZEN CONFIGURATION
# ============================================================

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8"
) as file:

    config = json.load(file)


if config.get("status") != "FROZEN":

    raise ValueError(
        "Final model configuration is not marked FROZEN."
    )


if config.get(
    "test_used_for_model_selection",
    False
):

    raise ValueError(
        "Configuration indicates test data was used "
        "for model selection."
    )


if config.get(
    "test_used_for_threshold_selection",
    False
):

    raise ValueError(
        "Configuration indicates test data was used "
        "for threshold selection."
    )


threshold = float(
    config["selected_threshold"]
)


print("\nFrozen threshold:")
print(f"{threshold:.12f}")


print("\nFrozen feature set:")
print(
    config.get(
        "feature_set",
        "UNKNOWN"
    )
)


# ============================================================
# 9. VERIFY FROZEN MODEL HASH
# ============================================================

with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8"
) as file:

    manifest = json.load(file)


expected_model_hash = (
    manifest
    ["final_model"]
    ["sha256"]
)

current_model_hash = sha256_file(
    MODEL_FILE
)


print("\nExpected model SHA256:")
print(expected_model_hash)

print("\nCurrent model SHA256:")
print(current_model_hash)


if (
    expected_model_hash
    !=
    current_model_hash
):

    raise RuntimeError(
        "FINAL MODEL HASH MISMATCH. "
        "Evaluation aborted."
    )


print("\nMODEL HASH: PASS")


# ============================================================
# 10. LOAD FEATURE ORDER
# ============================================================

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as file:

    features = [
        line.strip()
        for line in file
        if line.strip()
    ]


print("\nNumber of frozen features:")
print(len(features))


if len(features) != 30:

    raise ValueError(
        f"Expected 30 frozen features, "
        f"found {len(features)}."
    )


if len(set(features)) != len(features):

    raise ValueError(
        "Duplicate feature names found."
    )


# ============================================================
# 11. LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)


print("\nModel object:")
print(type(model))


# ============================================================
# 12. PIPELINE SAFETY CHECK
# ============================================================

if not hasattr(
    model,
    "named_steps"
):

    raise ValueError(
        "Frozen model is not an sklearn Pipeline."
    )


print("\nPipeline steps:")

for step_name in model.named_steps:

    print(
        f" - {step_name}"
    )


if "scaler" not in model.named_steps:

    raise ValueError(
        "Frozen pipeline does not contain scaler."
    )


if "clf" not in model.named_steps:

    raise ValueError(
        "Frozen pipeline does not contain classifier."
    )


print("\nScaling strategy:")
print("INTERNAL PIPELINE SCALER")

print("\nExternal scaling:")
print("NOT USED")


# ============================================================
# 13. MODEL FEATURE COUNT CHECK
# ============================================================

if hasattr(
    model,
    "n_features_in_"
):

    print("\nModel n_features_in_:")
    print(
        model.n_features_in_
    )

    if (
        int(model.n_features_in_)
        != 30
    ):

        raise ValueError(
            "Model does not expect 30 features."
        )


# ============================================================
# 14. FEATURE NAME ORDER CHECK
# ============================================================

if hasattr(
    model,
    "feature_names_in_"
):

    model_feature_names = list(
        model.feature_names_in_
    )

    if (
        model_feature_names
        !=
        features
    ):

        raise ValueError(
            "Frozen feature order does not match "
            "model.feature_names_in_."
        )

    print("\nFeature order check:")
    print("PASS")


# ============================================================
# 15. CLASS / SCORE ORIENTATION CHECK
# ============================================================

if not hasattr(
    model,
    "classes_"
):

    raise ValueError(
        "Frozen model does not expose classes_."
    )


model_classes = list(
    model.classes_
)


print("\nModel classes:")
print(model_classes)


# The frozen threshold was optimized with:
# Normal = 0
# Early_Demented = 1
#
# Therefore positive decision_function scores must correspond
# to class 1.

if (
    len(model_classes) != 2
    or model_classes[0] != 0
    or model_classes[1] != 1
):

    raise ValueError(
        "\nUnexpected model class encoding.\n"
        "Expected model.classes_ == [0, 1].\n"
        "STOP before opening/evaluating the test set."
    )


print("\nPositive class:")
print("1 = Early_Demented")

print("\nScore orientation check:")
print("PASS")


# ============================================================
# 16. PRE-TEST CHECK COMPLETE
# ============================================================

print("\n")
print("=" * 78)
print("ALL PRE-TEST CHECKS PASSED")
print("=" * 78)

print("\nFrozen system is internally consistent.")

print("\nThe TEST set will now be opened.")
print("No model or threshold changes are allowed after this point.")


# ============================================================
# 17. OPEN TEST SET
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)


print("\n")
print("=" * 78)
print("TEST SET OPENED")
print("=" * 78)


print("\nRows:")
print(
    len(test_df)
)


if (
    len(test_df)
    !=
    EXPECTED_TEST_ROWS
):

    raise ValueError(
        f"Expected {EXPECTED_TEST_ROWS} test rows, "
        f"found {len(test_df)}."
    )


# ============================================================
# 18. TEST SPLIT INTEGRITY CHECK
# ============================================================

if "target_name" not in test_df.columns:

    raise ValueError(
        "target_name column missing from test.csv."
    )


binary_counts = (
    test_df["target_name"]
    .value_counts()
    .to_dict()
)


print("\nBinary distribution:")
print(
    test_df["target_name"]
    .value_counts()
)


for label, expected_count in (
    EXPECTED_BINARY_COUNTS.items()
):

    actual_count = int(
        binary_counts.get(
            label,
            0
        )
    )

    if actual_count != expected_count:

        raise ValueError(
            f"Unexpected number of {label} images. "
            f"Expected {expected_count}, "
            f"found {actual_count}."
        )


print("\nBinary split integrity:")
print("PASS")


if "original_class" in test_df.columns:

    original_counts = (
        test_df["original_class"]
        .value_counts()
        .to_dict()
    )

    print("\nOriginal-class distribution:")
    print(
        test_df["original_class"]
        .value_counts()
    )

    for label, expected_count in (
        EXPECTED_ORIGINAL_COUNTS.items()
    ):

        actual_count = int(
            original_counts.get(
                label,
                0
            )
        )

        if actual_count != expected_count:

            raise ValueError(
                f"Unexpected number of {label} images. "
                f"Expected {expected_count}, "
                f"found {actual_count}."
            )

    print("\nOriginal-class split integrity:")
    print("PASS")


# ============================================================
# 19. FIND FEATURE VALUES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in test_df.columns
]


if not missing_features:

    print("\nAll 30 features are present directly in test.csv.")

    evaluation_df = test_df.copy()


else:

    print("\nThe split file does not contain all feature columns.")

    print("\nMissing features:")
    for feature in missing_features:
        print(
            f" - {feature}"
        )

    print("\nSearching for master feature table...")


    # --------------------------------------------------------
    # Search CSV files for one containing:
    #   filename + all frozen features
    # --------------------------------------------------------

    candidate_files = []

    for csv_file in RESULTS_DIR.rglob(
        "*.csv"
    ):

        if (
            csv_file.resolve()
            ==
            TEST_FILE.resolve()
        ):
            continue

        # Never use pre-scaled training-preparation files here.
        if (
            "TRAINING_PREPARATION"
            in str(csv_file).upper()
        ):
            continue

        try:

            header = pd.read_csv(
                csv_file,
                nrows=0
            ).columns.tolist()

        except Exception:
            continue

        if (
            "filename" in header
            and
            all(
                feature in header
                for feature in features
            )
        ):

            candidate_files.append(
                csv_file
            )


    if len(candidate_files) == 0:

        raise FileNotFoundError(
            "\nCould not locate a master unscaled "
            "feature table containing all 30 features."
        )


    print("\nCandidate master feature files:")

    for candidate in candidate_files:

        print(
            candidate
        )


    # Prefer files whose path mentions FULL_FEATURE
    preferred_candidates = [
        path
        for path in candidate_files
        if "FULL_FEATURE" in str(path).upper()
    ]


    if preferred_candidates:

        master_file = preferred_candidates[0]

    else:

        master_file = candidate_files[0]


    print("\nSelected master feature file:")
    print(
        master_file
    )


    master_df = pd.read_csv(
        master_file
    )


    if (
        master_df["filename"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate filenames found in master "
            "feature table."
        )


    required_master_columns = [
        "filename"
    ] + features


    feature_table = master_df[
        required_master_columns
    ].copy()


    evaluation_df = test_df.merge(
        feature_table,
        on="filename",
        how="left",
        validate="one_to_one"
    )


    if (
        len(evaluation_df)
        !=
        EXPECTED_TEST_ROWS
    ):

        raise ValueError(
            "Row count changed after feature merge."
        )


    if (
        evaluation_df[features]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Some test images could not be matched "
            "to feature values."
        )


    print("\nFeature merge:")
    print("PASS")


# ============================================================
# 20. CREATE TARGET
# ============================================================

target_mapping = {
    "Normal": 0,
    "Early_Demented": 1,
}


unknown_targets = set(
    evaluation_df["target_name"]
    .dropna()
    .unique()
) - set(
    target_mapping.keys()
)


if unknown_targets:

    raise ValueError(
        f"Unexpected target labels: "
        f"{unknown_targets}"
    )


y_true = (
    evaluation_df["target_name"]
    .map(
        target_mapping
    )
    .astype(int)
    .to_numpy()
)


# ============================================================
# 21. CREATE TEST MATRIX
# ============================================================

X_test = evaluation_df[
    features
].copy()


# ============================================================
# 22. FINAL NUMERICAL QC
# ============================================================

nan_count = int(
    X_test
    .isna()
    .sum()
    .sum()
)


inf_count = int(
    np.isinf(
        X_test.to_numpy(
            dtype=float
        )
    )
    .sum()
)


print("\n")
print("=" * 78)
print("FINAL TEST NUMERICAL QC")
print("=" * 78)

print("\nShape:")
print(
    X_test.shape
)

print("\nNaN values:")
print(
    nan_count
)

print("\nInfinite values:")
print(
    inf_count
)


if nan_count != 0:

    raise ValueError(
        "NaN values detected in final test features."
    )


if inf_count != 0:

    raise ValueError(
        "Infinite values detected in final test features."
    )


if (
    X_test.shape
    !=
    (4480, 30)
):

    raise ValueError(
        f"Expected X_test shape (4480, 30), "
        f"found {X_test.shape}."
    )


print("\nTEST MATRIX QC:")
print("PASS")


# ============================================================
# 23. FINAL MODEL INFERENCE
# ============================================================

print("\n")
print("=" * 78)
print("RUNNING FROZEN MODEL ON TEST SET")
print("=" * 78)


decision_scores = model.decision_function(
    X_test
)


decision_scores = np.asarray(
    decision_scores
).reshape(-1)


if (
    len(decision_scores)
    !=
    EXPECTED_TEST_ROWS
):

    raise ValueError(
        "Unexpected number of decision scores."
    )


print("\nDecision score range:")

print(
    "Minimum:",
    float(
        decision_scores.min()
    )
)

print(
    "Maximum:",
    float(
        decision_scores.max()
    )
)

print(
    "Mean:",
    float(
        decision_scores.mean()
    )
)

print(
    "Median:",
    float(
        np.median(
            decision_scores
        )
    )
)


# ============================================================
# 24. APPLY THE FROZEN THRESHOLD
# ============================================================

y_pred = (
    decision_scores
    >=
    threshold
).astype(int)


print("\nFrozen threshold used:")
print(
    f"{threshold:.12f}"
)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1
    ]
)


TN, FP, FN, TP = (
    cm.ravel()
)


TN = int(TN)
FP = int(FP)
FN = int(FN)
TP = int(TP)


# ============================================================
# 26. FINAL TEST METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        y_pred
    )
)

sensitivity = recall_score(
    y_true,
    y_pred,
    pos_label=1
)

specificity = (
    TN
    /
    (TN + FP)
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1
)

roc_auc = roc_auc_score(
    y_true,
    decision_scores
)

pr_auc = average_precision_score(
    y_true,
    decision_scores
)


# ============================================================
# 27. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 78)
print("FINAL TEST RESULTS")
print("=" * 78)

print(
    f"\nTest images:          {len(y_true)}"
)

print(
    f"Threshold:            {threshold:.12f}"
)

print(
    f"\nAccuracy:             {accuracy:.6f}"
)

print(
    f"Balanced Accuracy:    {balanced_accuracy:.6f}"
)

print(
    f"Sensitivity / Recall: {sensitivity:.6f}"
)

print(
    f"Specificity:          {specificity:.6f}"
)

print(
    f"Precision:            {precision:.6f}"
)

print(
    f"F1 Score:             {f1:.6f}"
)

print(
    f"ROC-AUC:              {roc_auc:.6f}"
)

print(
    f"PR-AUC:               {pr_auc:.6f}"
)


print("\nConfusion Matrix:")

print(
    f"TN = {TN}"
)

print(
    f"FP = {FP}"
)

print(
    f"FN = {FN}"
)

print(
    f"TP = {TP}"
)


# ============================================================
# 28. SAVE PREDICTIONS
# ============================================================

prediction_output = pd.DataFrame()


if "filename" in evaluation_df.columns:

    prediction_output[
        "filename"
    ] = evaluation_df[
        "filename"
    ].values


if "original_class" in evaluation_df.columns:

    prediction_output[
        "original_class"
    ] = evaluation_df[
        "original_class"
    ].values


prediction_output[
    "target_name"
] = evaluation_df[
    "target_name"
].values


prediction_output[
    "y_true"
] = y_true


prediction_output[
    "decision_score"
] = decision_scores


prediction_output[
    "threshold"
] = threshold


prediction_output[
    "y_pred"
] = y_pred


prediction_output[
    "predicted_class"
] = np.where(
    y_pred == 1,
    "Early_Demented",
    "Normal"
)


prediction_output[
    "correct"
] = (
    y_true
    ==
    y_pred
)


prediction_output.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 29. SAVE METRICS JSON
# ============================================================

metrics = {

    "evaluation_status":
        "FINAL_TEST_EVALUATION",

    "test_images":
        int(
            len(y_true)
        ),

    "model":
        "FULL_30__SVM_RBF__TUNED",

    "number_of_features":
        30,

    "threshold":
        float(
            threshold
        ),

    "threshold_source":
        "VALIDATION_ONLY",

    "accuracy":
        float(
            accuracy
        ),

    "balanced_accuracy":
        float(
            balanced_accuracy
        ),

    "sensitivity_recall":
        float(
            sensitivity
        ),

    "specificity":
        float(
            specificity
        ),

    "precision":
        float(
            precision
        ),

    "f1":
        float(
            f1
        ),

    "roc_auc":
        float(
            roc_auc
        ),

    "pr_auc":
        float(
            pr_auc
        ),

    "confusion_matrix": {

        "TN":
            TN,

        "FP":
            FP,

        "FN":
            FN,

        "TP":
            TP
    },

    "test_target_distribution": {

        "Early_Demented":
            int(
                np.sum(
                    y_true == 1
                )
            ),

        "Normal":
            int(
                np.sum(
                    y_true == 0
                )
            )
    },

    "model_sha256":
        current_model_hash
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 30. SAVE TEST FILE HASH
# ============================================================

test_file_hash = sha256_file(
    TEST_FILE
)


with open(
    TEST_HASH_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        test_file_hash
    )


# ============================================================
# 31. CONFUSION MATRIX FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        6,
        5
    )
)


image = ax.imshow(
    cm
)


ax.set_title(
    "Final Test Confusion Matrix"
)


ax.set_xlabel(
    "Predicted Class"
)


ax.set_ylabel(
    "True Class"
)


ax.set_xticks(
    [
        0,
        1
    ]
)


ax.set_yticks(
    [
        0,
        1
    ]
)


ax.set_xticklabels(
    [
        "Normal",
        "Early Demented"
    ]
)


ax.set_yticklabels(
    [
        "Normal",
        "Early Demented"
    ]
)


for i in range(
    2
):

    for j in range(
        2
    ):

        ax.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center"
        )


fig.tight_layout()


fig.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 32. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_true,
    decision_scores
)


fig, ax = plt.subplots(
    figsize=(
        7,
        6
    )
)


ax.plot(
    fpr,
    tpr,
    label=(
        f"ROC-AUC = "
        f"{roc_auc:.4f}"
    )
)


ax.plot(
    [
        0,
        1
    ],
    [
        0,
        1
    ],
    linestyle="--"
)


ax.set_xlabel(
    "False Positive Rate"
)


ax.set_ylabel(
    "True Positive Rate"
)


ax.set_title(
    "Final Test ROC Curve"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    ROC_CURVE_FILE,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 33. PRECISION-RECALL CURVE
# ============================================================

pr_precision, pr_recall, _ = (
    precision_recall_curve(
        y_true,
        decision_scores
    )
)


fig, ax = plt.subplots(
    figsize=(
        7,
        6
    )
)


ax.plot(
    pr_recall,
    pr_precision,
    label=(
        f"PR-AUC = "
        f"{pr_auc:.4f}"
    )
)


ax.set_xlabel(
    "Recall"
)


ax.set_ylabel(
    "Precision"
)


ax.set_title(
    "Final Test Precision-Recall Curve"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    PR_CURVE_FILE,
    dpi=200,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# 34. HUMAN-READABLE REPORT
# ============================================================

report = f"""
==============================================================================
FINAL TEST EVALUATION REPORT
==============================================================================

Project:
Alzheimer MRI Early Detection

FINAL FROZEN SYSTEM
-------------------
Model:
FULL_30 + Tuned SVM RBF

Features:
30

Scaling:
Internal StandardScaler inside frozen sklearn Pipeline

Frozen threshold:
{threshold:.12f}

Threshold selection:
VALIDATION ONLY

TEST SET
--------
Images:
{len(y_true)}

Normal:
{int(np.sum(y_true == 0))}

Early_Demented:
{int(np.sum(y_true == 1))}

FINAL TEST PERFORMANCE
----------------------
Accuracy:
{accuracy:.6f}

Balanced Accuracy:
{balanced_accuracy:.6f}

Sensitivity / Recall:
{sensitivity:.6f}

Specificity:
{specificity:.6f}

Precision:
{precision:.6f}

F1:
{f1:.6f}

ROC-AUC:
{roc_auc:.6f}

PR-AUC:
{pr_auc:.6f}

CONFUSION MATRIX
----------------
TN:
{TN}

FP:
{FP}

FN:
{FN}

TP:
{TP}

MODEL SHA256
------------
{current_model_hash}

TEST CSV SHA256
---------------
{test_file_hash}

IMPORTANT
---------
The test set was NOT used for:
- model selection
- feature selection
- hyperparameter tuning
- threshold selection

The frozen model and threshold must not be changed in response
to these test results.

==============================================================================
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        report.strip()
    )


# ============================================================
# 35. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 78)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 78)


print("\nResults saved to:")
print(
    OUTPUT_DIR
)


print("\nMetrics:")
print(
    METRICS_FILE
)


print("\nPredictions:")
print(
    PREDICTIONS_FILE
)


print("\nConfusion matrix:")
print(
    CONFUSION_MATRIX_FILE
)


print("\nROC curve:")
print(
    ROC_CURVE_FILE
)


print("\nPrecision-Recall curve:")
print(
    PR_CURVE_FILE
)


print("\nReport:")
print(
    REPORT_FILE
)


print("\n")
print("=" * 78)
print("IMPORTANT")
print("=" * 78)

print(
    "The FINAL TEST has now been evaluated."
)

print(
    "Do NOT modify the model, features, "
    "hyperparameters, or threshold based on "
    "these results."
)

print("\nNEXT PROJECT STAGE:")
print(
    "Explainability + Final AI System"
)

print("=" * 78)