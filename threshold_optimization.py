# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
#
# Project:
# Alzheimer MRI Early Detection
#
# Primary model:
# FULL_30 + Tuned SVM RBF
#
# IMPORTANT:
# - Uses VALIDATION only for threshold selection.
# - TRAIN is not needed here.
# - TEST is NEVER loaded.
# - Model parameters are already frozen from tuning.
#
# Outputs:
# - All threshold results
# - Best balanced-accuracy threshold
# - Sensitivity-focused candidate
# - Balanced sensitivity/specificity candidate
# - ROC curve
# - Precision-Recall curve
# - Sensitivity/Specificity vs threshold
# - Balanced Accuracy vs threshold
#
# ============================================================


from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SPLIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "GROUP_AWARE_SPLIT"
)


PREP_DIR = (
    PROJECT_ROOT
    / "results"
    / "TRAINING_PREPARATION"
)


TUNING_DIR = (
    PROJECT_ROOT
    / "results"
    / "HYPERPARAMETER_TUNING"
)


MODEL_DIR = (
    TUNING_DIR
    / "models"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "THRESHOLD_OPTIMIZATION"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

VALIDATION_FILE = (
    SPLIT_DIR
    / "validation.csv"
)


FULL_FEATURE_FILE = (
    PREP_DIR
    / "full_feature_set.txt"
)


MODEL_FILE = (
    MODEL_DIR
    / "FULL_30__SVM_RBF__TUNED.joblib"
)


# ============================================================
# 3. VERIFY FILES
# ============================================================

required_files = [
    VALIDATION_FILE,
    FULL_FEATURE_FILE,
    MODEL_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )


# ============================================================
# 4. HEADER
# ============================================================

print("\n")
print("=" * 75)
print("THRESHOLD OPTIMIZATION")
print("=" * 75)

print("\nPrimary model:")
print("FULL_30 + Tuned SVM RBF")

print("\nThreshold-selection dataset:")
print("VALIDATION ONLY")

print("\nTEST SET:")
print("LOCKED / NOT LOADED")


# ============================================================
# 5. LOAD VALIDATION DATA
# ============================================================

val_df = pd.read_csv(
    VALIDATION_FILE
)


print("\nValidation images:")
print(
    len(val_df)
)


if "target" not in val_df.columns:

    raise ValueError(
        "Column 'target' not found in validation.csv"
    )


# ============================================================
# 6. LOAD FEATURE LIST
# ============================================================

with open(
    FULL_FEATURE_FILE,
    "r",
    encoding="utf-8"
) as file:

    features = [
        line.strip()
        for line in file
        if line.strip()
    ]


print("\nNumber of features:")
print(
    len(features)
)


missing_features = [
    feature
    for feature in features
    if feature not in val_df.columns
]


if missing_features:

    raise ValueError(
        "\nMissing validation features:\n"
        +
        "\n".join(
            missing_features
        )
    )


# ============================================================
# 7. BUILD X / y
# ============================================================

X_val = (
    val_df[
        features
    ]
    .to_numpy(
        dtype=float
    )
)


y_val = (
    val_df[
        "target"
    ]
    .to_numpy(
        dtype=int
    )
)


# ============================================================
# 8. NUMERICAL QC
# ============================================================

nan_count = int(
    np.isnan(
        X_val
    ).sum()
)


inf_count = int(
    np.isinf(
        X_val
    ).sum()
)


print("\nNumerical QC:")

print(
    "NaN values:",
    nan_count
)

print(
    "Infinite values:",
    inf_count
)


if (
    nan_count != 0
    or
    inf_count != 0
):

    raise ValueError(
        "NaN or Inf found in validation features."
    )


# ============================================================
# 9. LOAD TUNED SVM
# ============================================================

model = joblib.load(
    MODEL_FILE
)


print("\nModel loaded successfully:")
print(
    MODEL_FILE
)


# ============================================================
# 10. GET RAW SVM DECISION SCORES
# ============================================================

if not hasattr(
    model,
    "decision_function"
):

    raise ValueError(
        "Loaded model does not provide decision_function()."
    )


scores = model.decision_function(
    X_val
)


scores = np.asarray(
    scores
).ravel()


print("\nDecision score range:")

print(
    "Minimum:",
    float(
        scores.min()
    )
)

print(
    "Maximum:",
    float(
        scores.max()
    )
)

print(
    "Mean:",
    float(
        scores.mean()
    )
)

print(
    "Median:",
    float(
        np.median(
            scores
        )
    )
)


# ============================================================
# 11. GLOBAL THRESHOLD-INDEPENDENT METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_val,
    scores
)


pr_auc = average_precision_score(
    y_val,
    scores
)


print("\nThreshold-independent metrics:")

print(
    f"ROC-AUC: {roc_auc:.6f}"
)

print(
    f"PR-AUC : {pr_auc:.6f}"
)


# ============================================================
# 12. FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_threshold_metrics(
    y_true,
    score_values,
    threshold
):

    y_pred = (
        score_values
        >=
        threshold
    ).astype(int)


    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    )


    tn, fp, fn, tp = (
        matrix.ravel()
    )


    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )


    specificity = (
        tn
        /
        (tn + fp)
        if
        (tn + fp) > 0
        else
        np.nan
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
        pos_label=1,
        zero_division=0
    )


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


    youden_j = (
        sensitivity
        +
        specificity
        -
        1
    )


    return {

        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "sensitivity":
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

        "youden_j":
            float(
                youden_j
            ),

        "TN":
            int(
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TP":
            int(
                tp
            )
    }


# ============================================================
# 13. DEFAULT THRESHOLD = 0
# ============================================================

default_metrics = (
    calculate_threshold_metrics(
        y_val,
        scores,
        0.0
    )
)


print("\n")
print("=" * 75)
print("DEFAULT SVM THRESHOLD = 0.0")
print("=" * 75)


for (
    key,
    value
) in default_metrics.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 14. GENERATE THRESHOLD GRID
# ============================================================
#
# We do NOT assume the useful range is -1 to +1.
# Instead, the threshold search is based on the actual
# validation decision scores.
#
# 2001 candidate thresholds are enough for a smooth,
# precise search while remaining computationally cheap.
# ============================================================

thresholds = np.linspace(
    scores.min(),
    scores.max(),
    2001
)


# Ensure the default SVM threshold 0 is included exactly.

thresholds = np.unique(
    np.append(
        thresholds,
        0.0
    )
)


print("\nThreshold candidates:")
print(
    len(thresholds)
)


# ============================================================
# 15. EVALUATE ALL THRESHOLDS
# ============================================================

rows = []


for threshold in thresholds:

    metrics = (
        calculate_threshold_metrics(
            y_val,
            scores,
            threshold
        )
    )


    rows.append(
        metrics
    )


results_df = pd.DataFrame(
    rows
)


# ============================================================
# 16. SAVE COMPLETE THRESHOLD TABLE
# ============================================================

all_thresholds_file = (
    OUTPUT_DIR
    /
    "all_threshold_results.csv"
)


results_df.to_csv(
    all_thresholds_file,
    index=False
)


# ============================================================
# 17. BEST BALANCED ACCURACY THRESHOLD
# ============================================================

best_balanced_index = (
    results_df[
        "balanced_accuracy"
    ]
    .idxmax()
)


best_balanced = (
    results_df
    .loc[
        best_balanced_index
    ]
)


# ============================================================
# 18. BEST YOUDEN J THRESHOLD
# ============================================================

best_youden_index = (
    results_df[
        "youden_j"
    ]
    .idxmax()
)


best_youden = (
    results_df
    .loc[
        best_youden_index
    ]
)


# ============================================================
# 19. BALANCED SENSITIVITY / SPECIFICITY THRESHOLD
# ============================================================

results_df[
    "sens_spec_gap"
] = (

    results_df[
        "sensitivity"
    ]

    -

    results_df[
        "specificity"
    ]

).abs()


balanced_tradeoff_index = (
    results_df[
        "sens_spec_gap"
    ]
    .idxmin()
)


balanced_tradeoff = (
    results_df
    .loc[
        balanced_tradeoff_index
    ]
)


# ============================================================
# 20. HIGH-SENSITIVITY CANDIDATE
# ============================================================
#
# Find thresholds that achieve >= 90% sensitivity.
#
# Among those, select the threshold with the highest
# specificity.
# ============================================================

high_sensitivity_candidates = (

    results_df[
        results_df[
            "sensitivity"
        ]
        >=
        0.90
    ]
)


if not high_sensitivity_candidates.empty:

    high_sensitivity_index = (

        high_sensitivity_candidates[
            "specificity"
        ]
        .idxmax()
    )


    high_sensitivity = (
        results_df
        .loc[
            high_sensitivity_index
        ]
    )

else:

    high_sensitivity = None


# ============================================================
# 21. DISPLAY HELPER
# ============================================================

def print_candidate(
    title,
    row
):

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


    columns_to_show = [

        "threshold",

        "accuracy",

        "balanced_accuracy",

        "sensitivity",

        "specificity",

        "precision",

        "f1",

        "youden_j",

        "TN",

        "FP",

        "FN",

        "TP"
    ]


    for column in columns_to_show:

        value = row[
            column
        ]


        if column in [
            "TN",
            "FP",
            "FN",
            "TP"
        ]:

            print(
                f"{column}: "
                f"{int(value)}"
            )

        else:

            print(
                f"{column}: "
                f"{float(value):.6f}"
            )


# ============================================================
# 22. PRINT CANDIDATES
# ============================================================

print_candidate(
    "BEST BALANCED ACCURACY THRESHOLD",
    best_balanced
)


print_candidate(
    "BEST YOUDEN-J THRESHOLD",
    best_youden
)


print_candidate(
    "MOST BALANCED SENSITIVITY / SPECIFICITY",
    balanced_tradeoff
)


if high_sensitivity is not None:

    print_candidate(
        "HIGH-SENSITIVITY CANDIDATE (>= 90%)",
        high_sensitivity
    )

else:

    print("\n")
    print("=" * 75)
    print("HIGH-SENSITIVITY CANDIDATE")
    print("=" * 75)

    print(
        "No tested threshold achieved "
        "sensitivity >= 90%."
    )


# ============================================================
# 23. SAVE CANDIDATE SUMMARY
# ============================================================

candidate_rows = []


default_row = dict(
    default_metrics
)

default_row[
    "criterion"
] = "DEFAULT_THRESHOLD_0"


candidate_rows.append(
    default_row
)


row = best_balanced.to_dict()

row[
    "criterion"
] = "BEST_BALANCED_ACCURACY"

candidate_rows.append(
    row
)


row = best_youden.to_dict()

row[
    "criterion"
] = "BEST_YOUDEN_J"

candidate_rows.append(
    row
)


row = balanced_tradeoff.to_dict()

row[
    "criterion"
] = "BALANCED_SENS_SPEC"

candidate_rows.append(
    row
)


if high_sensitivity is not None:

    row = high_sensitivity.to_dict()

    row[
        "criterion"
    ] = "SENSITIVITY_AT_LEAST_90"

    candidate_rows.append(
        row
    )


candidate_df = pd.DataFrame(
    candidate_rows
)


candidate_file = (
    OUTPUT_DIR
    /
    "threshold_candidates.csv"
)


candidate_df.to_csv(
    candidate_file,
    index=False
)


# ============================================================
# 24. SAVE PRIMARY THRESHOLD
# ============================================================
#
# Primary optimization criterion:
# Balanced Accuracy
#
# This is consistent with the model-selection metric used
# during previous project stages.
# ============================================================

primary_threshold = float(
    best_balanced[
        "threshold"
    ]
)


primary_summary = {

    "model":
        "FULL_30__SVM_RBF__TUNED",

    "feature_set":
        "FULL_30",

    "selection_dataset":
        "VALIDATION",

    "criterion":
        "MAX_BALANCED_ACCURACY",

    "threshold":
        primary_threshold,

    "accuracy":
        float(
            best_balanced[
                "accuracy"
            ]
        ),

    "balanced_accuracy":
        float(
            best_balanced[
                "balanced_accuracy"
            ]
        ),

    "sensitivity":
        float(
            best_balanced[
                "sensitivity"
            ]
        ),

    "specificity":
        float(
            best_balanced[
                "specificity"
            ]
        ),

    "precision":
        float(
            best_balanced[
                "precision"
            ]
        ),

    "f1":
        float(
            best_balanced[
                "f1"
            ]
        ),

    "roc_auc":
        float(
            roc_auc
        ),

    "pr_auc":
        float(
            pr_auc
        ),

    "TN":
        int(
            best_balanced[
                "TN"
            ]
        ),

    "FP":
        int(
            best_balanced[
                "FP"
            ]
        ),

    "FN":
        int(
            best_balanced[
                "FN"
            ]
        ),

    "TP":
        int(
            best_balanced[
                "TP"
            ]
        ),

    "test_set_used":
        False
}


primary_threshold_file = (
    OUTPUT_DIR
    /
    "selected_threshold.json"
)


with open(
    primary_threshold_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        primary_summary,
        file,
        indent=4
    )


# ============================================================
# 25. ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = (
    roc_curve(
        y_val,
        scores
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr,
    tpr,
    label=
        f"SVM ROC-AUC = "
        f"{roc_auc:.4f}"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "Validation ROC Curve - Tuned SVM RBF"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    /
    "roc_curve.png",
    dpi=300
)


plt.close()


# ============================================================
# 26. PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_val,
        scores
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    recall_curve,
    precision_curve,
    label=
        f"PR-AUC = "
        f"{pr_auc:.4f}"
)


plt.xlabel(
    "Recall / Sensitivity"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Validation Precision-Recall Curve - Tuned SVM RBF"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    /
    "precision_recall_curve.png",
    dpi=300
)


plt.close()


# ============================================================
# 27. SENSITIVITY / SPECIFICITY VS THRESHOLD
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    results_df[
        "threshold"
    ],
    results_df[
        "sensitivity"
    ],
    label="Sensitivity"
)


plt.plot(
    results_df[
        "threshold"
    ],
    results_df[
        "specificity"
    ],
    label="Specificity"
)


plt.axvline(
    primary_threshold,
    linestyle="--",
    label=
        "Selected Threshold"
)


plt.axvline(
    0.0,
    linestyle=":",
    label=
        "Default Threshold = 0"
)


plt.xlabel(
    "Decision Threshold"
)


plt.ylabel(
    "Metric Value"
)


plt.title(
    "Sensitivity and Specificity vs Decision Threshold"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    /
    "sensitivity_specificity_vs_threshold.png",
    dpi=300
)


plt.close()


# ============================================================
# 28. BALANCED ACCURACY VS THRESHOLD
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    results_df[
        "threshold"
    ],
    results_df[
        "balanced_accuracy"
    ]
)


plt.axvline(
    primary_threshold,
    linestyle="--",
    label=
        f"Selected = "
        f"{primary_threshold:.4f}"
)


plt.axvline(
    0.0,
    linestyle=":",
    label=
        "Default = 0"
)


plt.xlabel(
    "Decision Threshold"
)


plt.ylabel(
    "Balanced Accuracy"
)


plt.title(
    "Balanced Accuracy vs Decision Threshold"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    /
    "balanced_accuracy_vs_threshold.png",
    dpi=300
)


plt.close()


# ============================================================
# 29. DEFAULT VS OPTIMIZED COMPARISON
# ============================================================

comparison = pd.DataFrame([

    {
        "setting":
            "Default threshold",

        **default_metrics
    },

    {
        "setting":
            "Optimized threshold",

        **{
            key:
                (
                    float(value)
                    if
                    isinstance(
                        value,
                        (
                            np.floating,
                            float
                        )
                    )
                    else
                    int(value)
                    if
                    isinstance(
                        value,
                        (
                            np.integer,
                            int
                        )
                    )
                    else
                    value
                )

            for (
                key,
                value
            )

            in
            best_balanced
            .to_dict()
            .items()

            if key
            !=
            "sens_spec_gap"
        }
    }
])


comparison_file = (
    OUTPUT_DIR
    /
    "default_vs_optimized_threshold.csv"
)


comparison.to_csv(
    comparison_file,
    index=False
)


print("\n")
print("=" * 75)
print("DEFAULT VS OPTIMIZED")
print("=" * 75)


columns_to_print = [

    "setting",

    "threshold",

    "balanced_accuracy",

    "sensitivity",

    "specificity",

    "precision",

    "f1",

    "TN",

    "FP",

    "FN",

    "TP"
]


print(
    comparison[
        columns_to_print
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 30. FINAL MESSAGE
# ============================================================

print("\n\n")
print("=" * 75)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 75)


print("\nSelected validation threshold:")

print(
    f"{primary_threshold:.6f}"
)


print("\nValidation Balanced Accuracy:")

print(
    f"{best_balanced['balanced_accuracy']:.6f}"
)


print("\nValidation Sensitivity:")

print(
    f"{best_balanced['sensitivity']:.6f}"
)


print("\nValidation Specificity:")

print(
    f"{best_balanced['specificity']:.6f}"
)


print("\nROC-AUC:")

print(
    f"{roc_auc:.6f}"
)


print("\nPR-AUC:")

print(
    f"{pr_auc:.6f}"
)


print("\nOutputs saved to:")

print(
    OUTPUT_DIR
)


print("\nTEST SET STATUS:")

print(
    "LOCKED / NOT LOADED"
)


print("\nNEXT STEP:")

print(
    "Review threshold candidates, "
    "freeze the final model + threshold, "
    "then prepare the one-time final TEST evaluation."
)


print("=" * 75)