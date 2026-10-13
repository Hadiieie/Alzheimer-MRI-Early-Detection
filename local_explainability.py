# ============================================================
# LOCAL EXPLAINABILITY + CASE-LEVEL ERROR ANALYSIS
# Frozen Final RBF-SVM
#
# IMPORTANT:
#   - NO retraining
#   - NO threshold modification
#   - NO feature selection
#   - Test set is used descriptively only
#
# Method:
#   Local single-feature replacement analysis.
#
# For every selected test case:
#   1. Compute original decision score.
#   2. Replace ONE feature with its VALIDATION median.
#   3. Recompute decision score.
#   4. Measure how much that feature changes the decision.
#
# Positive local impact:
#   feature pushes prediction toward Early_Demented
#
# Negative local impact:
#   feature pushes prediction toward Normal
# ============================================================

from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\97338\Desktop\RAW DATA")

FINAL_MODEL_DIR = ROOT / "results" / "FINAL_MODEL"
SPLIT_DIR = ROOT / "results" / "GROUP_AWARE_SPLIT"

OUTPUT_DIR = ROOT / "results" / "LOCAL_EXPLAINABILITY"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = FINAL_MODEL_DIR / "final_model.joblib"
FEATURE_FILE = FINAL_MODEL_DIR / "final_features.txt"
CONFIG_FILE = FINAL_MODEL_DIR / "final_model_config.json"

VALIDATION_FILE = SPLIT_DIR / "validation.csv"
TEST_FILE = SPLIT_DIR / "test.csv"


# ============================================================
# SETTINGS
# ============================================================

TOP_FEATURES_PER_CASE = 12

NORMAL_LABEL = 0
DEMENTED_LABEL = 1

LABEL_MAP = {
    "Normal": 0,
    "Early_Demented": 1
}


# ============================================================
# HELPERS
# ============================================================

def clean_feature_line(line):

    line = line.strip()

    if not line:
        return None

    line = re.sub(
        r"^\d+\s*[\.\)]\s*",
        "",
        line
    )

    line = re.sub(
        r"^(KEEP|FEATURE)\s*:\s*",
        "",
        line,
        flags=re.I
    )

    return line.strip()


def load_feature_names():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    features = []

    for line in lines:

        feature = clean_feature_line(line)

        if feature:
            features.append(feature)

    features = list(dict.fromkeys(features))

    return features


def load_threshold():

    fallback_threshold = 0.108724729218

    if not CONFIG_FILE.exists():

        print(
            "WARNING: final_model_config.json not found."
        )

        print(
            f"Using frozen fallback threshold: "
            f"{fallback_threshold}"
        )

        return fallback_threshold

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    possible_keys = [
        "threshold",
        "final_threshold",
        "decision_threshold",
        "frozen_threshold"
    ]

    for key in possible_keys:

        if key in config:

            threshold = float(config[key])

            return threshold

    print(
        "WARNING: threshold field not found."
    )

    print(
        f"Using fallback threshold: "
        f"{fallback_threshold}"
    )

    return fallback_threshold


def safe_filename(text):

    text = str(text)

    text = re.sub(
        r"[^A-Za-z0-9_\-\.]",
        "_",
        text
    )

    return text


# ============================================================
# LOAD FROZEN ARTIFACTS
# ============================================================

print()
print("=" * 80)
print("LOCAL EXPLAINABILITY")
print("=" * 80)

print()
print("IMPORTANT:")
print("Frozen model will NOT be retrained.")
print("Frozen threshold will NOT be changed.")
print("Feature set will NOT be changed.")

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test file not found:\n{TEST_FILE}"
    )

model = joblib.load(MODEL_FILE)

features = load_feature_names()

threshold = load_threshold()

print()
print("Model:")
print(MODEL_FILE)

print()
print("Frozen threshold:")
print(f"{threshold:.12f}")

print()
print("Number of features:")
print(len(features))


# ============================================================
# LOAD VALIDATION + TEST
# ============================================================

validation_df = pd.read_csv(
    VALIDATION_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

print()
print("=" * 80)
print("DATASETS")
print("=" * 80)

print()
print("Validation rows:")
print(len(validation_df))

print()
print("Test rows:")
print(len(test_df))


# ============================================================
# BASIC CHECKS
# ============================================================

for dataset_name, df in [
    ("validation", validation_df),
    ("test", test_df)
]:

    if "target_name" not in df.columns:

        raise ValueError(
            f"target_name missing from "
            f"{dataset_name}.csv"
        )

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features in {dataset_name}:\n"
            f"{missing_features}"
        )


# ============================================================
# VALIDATION REFERENCE VALUES
# ============================================================

validation_X_df = validation_df[
    features
].copy()

validation_X = validation_X_df.to_numpy(
    dtype=float
)

if np.isnan(validation_X).sum() != 0:
    raise ValueError(
        "NaN values found in validation features."
    )

if np.isinf(validation_X).sum() != 0:
    raise ValueError(
        "Infinite values found in validation features."
    )


# Reference median for each raw feature
validation_medians = (
    validation_X_df
    .median(axis=0)
)

validation_means = (
    validation_X_df
    .mean(axis=0)
)

validation_stds = (
    validation_X_df
    .std(axis=0)
    .replace(0, np.nan)
)


# ============================================================
# TEST MATRIX
# ============================================================

X_test_df = test_df[
    features
].copy()

X_test = X_test_df.to_numpy(
    dtype=float
)

y_test = (
    test_df["target_name"]
    .map(LABEL_MAP)
    .to_numpy(dtype=int)
)

if np.isnan(X_test).sum() != 0:
    raise ValueError(
        "NaN values found in test features."
    )

if np.isinf(X_test).sum() != 0:
    raise ValueError(
        "Infinite values found in test features."
    )


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

print()
print("=" * 80)
print("RECONSTRUCTING FROZEN TEST PREDICTIONS")
print("=" * 80)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore"
    )

    test_scores = model.decision_function(
        X_test
    )

test_predictions = (
    test_scores >= threshold
).astype(int)

distance_to_threshold = (
    test_scores - threshold
)

absolute_distance = np.abs(
    distance_to_threshold
)


# ============================================================
# CONFUSION CATEGORY
# ============================================================

def confusion_category(
    true_label,
    predicted_label
):

    if (
        true_label == 1
        and predicted_label == 1
    ):
        return "TP"

    if (
        true_label == 0
        and predicted_label == 0
    ):
        return "TN"

    if (
        true_label == 0
        and predicted_label == 1
    ):
        return "FP"

    if (
        true_label == 1
        and predicted_label == 0
    ):
        return "FN"

    return "UNKNOWN"


categories = [
    confusion_category(y_true, y_pred)
    for y_true, y_pred
    in zip(
        y_test,
        test_predictions
    )
]


# ============================================================
# TEST PREDICTION TABLE
# ============================================================

prediction_df = test_df.copy()

prediction_df[
    "true_binary"
] = y_test

prediction_df[
    "predicted_binary"
] = test_predictions

prediction_df[
    "decision_score"
] = test_scores

prediction_df[
    "frozen_threshold"
] = threshold

prediction_df[
    "distance_from_threshold"
] = distance_to_threshold

prediction_df[
    "absolute_distance_from_threshold"
] = absolute_distance

prediction_df[
    "confusion_category"
] = categories


# ============================================================
# CONFUSION MATRIX CHECK
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

print()
print("Confusion Matrix:")

print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print()
print("Expected from final test:")

print("TN = 1206")
print("FP = 293")
print("FN = 549")
print("TP = 2432")

if (
    tn != 1206
    or fp != 293
    or fn != 549
    or tp != 2432
):

    print()
    print(
        "WARNING:"
    )

    print(
        "Reconstructed confusion matrix does not "
        "exactly match the frozen final-test report."
    )

else:

    print()
    print(
        "Final test reconstruction: PASS"
    )


# ============================================================
# SELECT EXPLANATION CASES
# ============================================================

print()
print("=" * 80)
print("SELECTING REPRESENTATIVE CASES")
print("=" * 80)

selected_cases = []

for category in [
    "TP",
    "TN",
    "FP",
    "FN"
]:

    category_df = prediction_df[
        prediction_df[
            "confusion_category"
        ] == category
    ].copy()

    print()
    print(
        f"{category}: {len(category_df)} cases"
    )

    if len(category_df) == 0:
        continue

    # High-confidence:
    # furthest away from threshold
    high_confidence_index = (
        category_df[
            "absolute_distance_from_threshold"
        ]
        .idxmax()
    )

    # Borderline:
    # closest to frozen threshold
    borderline_index = (
        category_df[
            "absolute_distance_from_threshold"
        ]
        .idxmin()
    )

    selected_cases.append(
        {
            "index": high_confidence_index,
            "confusion_category": category,
            "case_type": "HIGH_CONFIDENCE"
        }
    )

    # Avoid duplicate if category has only 1 row
    if (
        borderline_index
        != high_confidence_index
    ):

        selected_cases.append(
            {
                "index": borderline_index,
                "confusion_category": category,
                "case_type": "BORDERLINE"
            }
        )


# ============================================================
# LOCAL SINGLE-FEATURE REPLACEMENT ANALYSIS
# ============================================================

local_results = []

case_summary_rows = []


for case_number, selected in enumerate(
    selected_cases,
    start=1
):

    row_index = selected["index"]

    category = selected[
        "confusion_category"
    ]

    case_type = selected[
        "case_type"
    ]

    row = prediction_df.loc[
        row_index
    ]

    x_original = (
        X_test_df.loc[
            row_index,
            features
        ]
        .to_numpy(
            dtype=float
        )
    )

    original_score = float(
        row[
            "decision_score"
        ]
    )

    predicted_binary = int(
        row[
            "predicted_binary"
        ]
    )

    true_binary = int(
        row[
            "true_binary"
        ]
    )

    filename = (
        row["filename"]
        if "filename" in row.index
        else f"row_{row_index}"
    )

    original_class = (
        row["original_class"]
        if "original_class" in row.index
        else ""
    )

    print()
    print("=" * 80)

    print(
        f"CASE {case_number}: "
        f"{category} | {case_type}"
    )

    print("=" * 80)

    print()
    print("Filename:")
    print(filename)

    print()
    print("Original class:")
    print(original_class)

    print()
    print("True target:")
    print(
        "Early_Demented"
        if true_binary == 1
        else "Normal"
    )

    print()
    print("Predicted target:")
    print(
        "Early_Demented"
        if predicted_binary == 1
        else "Normal"
    )

    print()
    print("Decision score:")
    print(
        f"{original_score:.6f}"
    )

    print()
    print("Threshold:")
    print(
        f"{threshold:.6f}"
    )

    print()
    print("Distance from threshold:")
    print(
        f"{original_score - threshold:.6f}"
    )


    # --------------------------------------------------------
    # FEATURE-BY-FEATURE LOCAL ANALYSIS
    # --------------------------------------------------------

    case_feature_rows = []

    for feature_index, feature in enumerate(
        features
    ):

        x_modified = (
            x_original.copy()
        )

        reference_value = float(
            validation_medians[
                feature
            ]
        )

        original_feature_value = float(
            x_original[
                feature_index
            ]
        )

        # Replace only ONE feature
        # with validation population median
        x_modified[
            feature_index
        ] = reference_value

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            modified_score = float(
                model.decision_function(
                    x_modified.reshape(
                        1,
                        -1
                    )
                )[0]
            )

        # Positive:
        # current feature pushes toward Early_Demented
        #
        # Negative:
        # current feature pushes toward Normal
        local_impact = (
            original_score
            - modified_score
        )

        feature_std = (
            validation_stds[
                feature
            ]
        )

        if (
            pd.isna(feature_std)
            or feature_std == 0
        ):

            standardized_value = np.nan

        else:

            standardized_value = (
                original_feature_value
                - validation_means[
                    feature
                ]
            ) / feature_std


        case_feature_rows.append(
            {
                "case_number": case_number,
                "filename": filename,
                "original_class":
                    original_class,
                "confusion_category":
                    category,
                "case_type":
                    case_type,
                "true_binary":
                    true_binary,
                "predicted_binary":
                    predicted_binary,
                "original_score":
                    original_score,
                "threshold":
                    threshold,
                "distance_from_threshold":
                    original_score - threshold,
                "feature":
                    feature,
                "feature_value":
                    original_feature_value,
                "validation_median":
                    reference_value,
                "validation_mean":
                    validation_means[
                        feature
                    ],
                "standardized_feature_value":
                    standardized_value,
                "score_after_median_replacement":
                    modified_score,
                "local_impact":
                    local_impact,
                "absolute_local_impact":
                    abs(
                        local_impact
                    )
            }
        )


    case_feature_df = pd.DataFrame(
        case_feature_rows
    )

    case_feature_df = (
        case_feature_df
        .sort_values(
            "absolute_local_impact",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    local_results.extend(
        case_feature_df.to_dict(
            orient="records"
        )
    )


    # --------------------------------------------------------
    # PRINT TOP LOCAL DRIVERS
    # --------------------------------------------------------

    print()
    print(
        "Top local feature impacts:"
    )

    print(
        case_feature_df[
            [
                "feature",
                "feature_value",
                "validation_median",
                "local_impact",
                "absolute_local_impact"
            ]
        ]
        .head(
            TOP_FEATURES_PER_CASE
        )
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # CASE SUMMARY
    # --------------------------------------------------------

    case_summary_rows.append(
        {
            "case_number":
                case_number,
            "filename":
                filename,
            "original_class":
                original_class,
            "confusion_category":
                category,
            "case_type":
                case_type,
            "true_target":
                (
                    "Early_Demented"
                    if true_binary == 1
                    else "Normal"
                ),
            "predicted_target":
                (
                    "Early_Demented"
                    if predicted_binary == 1
                    else "Normal"
                ),
            "decision_score":
                original_score,
            "threshold":
                threshold,
            "distance_from_threshold":
                original_score
                - threshold
        }
    )


    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plot_df = (
        case_feature_df
        .head(
            TOP_FEATURES_PER_CASE
        )
        .sort_values(
            "local_impact",
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df[
            "feature"
        ],
        plot_df[
            "local_impact"
        ]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Local decision-score impact"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{category} | {case_type}\n"
        f"{filename}\n"
        f"Score={original_score:.3f} | "
        f"Threshold={threshold:.3f}"
    )

    plt.tight_layout()

    plot_name = (
        f"{case_number:02d}_"
        f"{category}_"
        f"{case_type}_"
        f"{safe_filename(filename)}.png"
    )

    plot_path = (
        PLOTS_DIR
        / plot_name
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

local_results_df = pd.DataFrame(
    local_results
)

case_summary_df = pd.DataFrame(
    case_summary_rows
)

all_predictions_file = (
    OUTPUT_DIR
    / "final_test_predictions_with_categories.csv"
)

selected_cases_file = (
    OUTPUT_DIR
    / "selected_local_cases.csv"
)

local_impacts_file = (
    OUTPUT_DIR
    / "local_feature_impacts.csv"
)

prediction_df.to_csv(
    all_predictions_file,
    index=False
)

case_summary_df.to_csv(
    selected_cases_file,
    index=False
)

local_results_df.to_csv(
    local_impacts_file,
    index=False
)


# ============================================================
# SAVE READABLE REPORT
# ============================================================

report_file = (
    OUTPUT_DIR
    / "local_explainability_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "LOCAL EXPLAINABILITY REPORT\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        "Frozen model:\n"
        f"{MODEL_FILE}\n\n"
    )

    f.write(
        f"Frozen threshold: "
        f"{threshold:.12f}\n\n"
    )

    f.write(
        "Final test confusion matrix:\n"
    )

    f.write(
        f"TN = {tn}\n"
        f"FP = {fp}\n"
        f"FN = {fn}\n"
        f"TP = {tp}\n\n"
    )

    f.write(
        "Selected cases:\n"
    )

    f.write(
        case_summary_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    for case_number in (
        case_summary_df[
            "case_number"
        ]
        .tolist()
    ):

        case_info = (
            case_summary_df[
                case_summary_df[
                    "case_number"
                ] == case_number
            ]
            .iloc[0]
        )

        feature_info = (
            local_results_df[
                local_results_df[
                    "case_number"
                ] == case_number
            ]
            .sort_values(
                "absolute_local_impact",
                ascending=False
            )
            .head(
                TOP_FEATURES_PER_CASE
            )
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            f"CASE {case_number}\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            f"Filename: "
            f"{case_info['filename']}\n"
        )

        f.write(
            f"Category: "
            f"{case_info['confusion_category']}\n"
        )

        f.write(
            f"Type: "
            f"{case_info['case_type']}\n"
        )

        f.write(
            f"True: "
            f"{case_info['true_target']}\n"
        )

        f.write(
            f"Predicted: "
            f"{case_info['predicted_target']}\n"
        )

        f.write(
            f"Decision score: "
            f"{case_info['decision_score']:.6f}\n"
        )

        f.write(
            f"Threshold: "
            f"{threshold:.6f}\n\n"
        )

        f.write(
            "Top local drivers:\n"
        )

        f.write(
            feature_info[
                [
                    "feature",
                    "feature_value",
                    "validation_median",
                    "local_impact"
                ]
            ]
            .to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("LOCAL EXPLAINABILITY COMPLETE")
print("=" * 80)

print()
print("Selected cases:")
print(
    case_summary_df[
        [
            "case_number",
            "filename",
            "confusion_category",
            "case_type",
            "true_target",
            "predicted_target",
            "decision_score",
            "distance_from_threshold"
        ]
    ].to_string(
        index=False
    )
)

print()
print("Files saved:")

print(
    all_predictions_file
)

print(
    selected_cases_file
)

print(
    local_impacts_file
)

print(
    report_file
)

print()
print("Plots saved inside:")
print(
    PLOTS_DIR
)

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

print()
print(
    "Positive local impact:"
)

print(
    "The current feature value pushes the case "
    "toward Early_Demented."
)

print()
print(
    "Negative local impact:"
)

print(
    "The current feature value pushes the case "
    "toward Normal."
)

print()
print(
    "Magnitude:"
)

print(
    "Larger absolute value means stronger local "
    "influence on this specific prediction."
)

print()
print("IMPORTANT:")
print(
    "This is a local perturbation explanation, "
    "not a causal interpretation."
)

print()
print("No model parameters were modified.")
print("No threshold was modified.")
print("No retraining was performed.")

print()
print("NEXT STEP:")
print(
    "Compare TP / TN / FP / FN local explanations "
    "and perform structured final error analysis."
)