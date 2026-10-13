from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

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

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\97338\Desktop\RAW DATA")

TEST_FILE = ROOT / "results" / "GROUP_AWARE_SPLIT" / "test.csv"

FINAL_MODEL_DIR = ROOT / "results" / "FINAL_MODEL"

MODEL_FILE = FINAL_MODEL_DIR / "final_model.joblib"
FEATURE_FILE = FINAL_MODEL_DIR / "final_features.txt"
CONFIG_FILE = FINAL_MODEL_DIR / "final_model_config.json"

OUTPUT_DIR = ROOT / "results" / "FINAL_ROBUSTNESS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Number of bootstrap repetitions
N_BOOTSTRAPS = 2000

# Frozen final test size
EXPECTED_TEST_ROWS = 4480

# Frozen final-test confusion matrix
EXPECTED_TN = 1206
EXPECTED_FP = 293
EXPECTED_FN = 549
EXPECTED_TP = 2432


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_feature_list(path):
    """
    Read final feature list from text file.

    Supports:
        brain_area
        p10
        ...

    Also supports numbered format:
        01. brain_area
        02. p10
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    features = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove possible numbering
        if ". " in line:
            left, right = line.split(". ", 1)

            if left.strip().isdigit():
                line = right.strip()

        low = line.lower()

        # Ignore headers if present
        if low.startswith("number of features"):
            continue

        if low in {
            "features",
            "feature list",
            "final features",
        }:
            continue

        features.append(line)

    return features


def find_target_column(df):
    """
    Try to identify binary target column.
    """

    candidates = [
        "target",
        "target_binary",
        "binary_target",
        "label",
        "target_name",
        "class_binary",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    raise KeyError(
        "Could not identify target column.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def convert_target_to_binary(series):
    """
    Convert binary target to:

        Early_Demented = 1
        Normal         = 0
    """

    # If already numeric 0/1
    if pd.api.types.is_numeric_dtype(series):

        unique_values = sorted(
            pd.Series(series)
            .dropna()
            .unique()
            .tolist()
        )

        if set(unique_values).issubset({0, 1}):

            return series.astype(int)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    positive_names = {
        "early_demented",
        "earlydemented",
        "demented",
        "1",
        "positive",
    }

    negative_names = {
        "normal",
        "non_demented",
        "nondemented",
        "0",
        "negative",
    }

    output = []

    for value in normalized:

        if value in positive_names:

            output.append(1)

        elif value in negative_names:

            output.append(0)

        else:

            raise ValueError(
                f"Unknown target value encountered: {value}"
            )

    return pd.Series(
        output,
        index=series.index,
        dtype=int,
    )


def get_classifier_classes(model):
    """
    Extract classifier classes if available.
    """

    if hasattr(model, "named_steps"):

        if "clf" in model.named_steps:

            clf = model.named_steps["clf"]

            if hasattr(clf, "classes_"):
                return clf.classes_

    if hasattr(model, "classes_"):
        return model.classes_

    return None


def get_decision_scores(model, X):
    """
    Get SVM decision scores.
    """

    if not hasattr(model, "decision_function"):

        raise AttributeError(
            "Frozen model does not provide decision_function()."
        )

    scores = model.decision_function(X)

    scores = np.asarray(scores)

    if scores.ndim != 1:

        raise ValueError(
            f"Expected 1D binary decision scores, "
            f"got shape {scores.shape}"
        )

    return scores.astype(float)


def calculate_metrics(
    y_true,
    scores,
    threshold,
):
    """
    Calculate final binary classification metrics.
    """

    y_pred = (
        scores >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    result = {

        "threshold": float(threshold),

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                scores,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                scores,
            ),

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return result


def percentile_ci(
    values,
    confidence=0.95,
):
    """
    Percentile bootstrap confidence interval.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    alpha = 1.0 - confidence

    lower = np.quantile(
        values,
        alpha / 2,
    )

    upper = np.quantile(
        values,
        1 - alpha / 2,
    )

    return lower, upper


def detect_group_column(df):
    """
    Detect leakage-group column if present.
    """

    candidates = [
        "leakage_group",
        "leakage_group_id",
        "group_id",
        "group",
        "duplicate_group",
        "near_duplicate_group",
        "near_duplicate_group_id",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def load_frozen_threshold(config):
    """
    Read the frozen threshold from final_model_config.json.

    Your configuration currently uses:
        selected_threshold
    """

    threshold_keys = [
        "selected_threshold",
        "threshold",
        "final_threshold",
        "decision_threshold",
        "threshold_selected",
    ]

    for key in threshold_keys:

        if key in config:

            threshold = float(
                config[key]
            )

            print(
                "Threshold loaded from config key:"
            )
            print(key)

            return threshold

    raise KeyError(
        "Could not find frozen threshold inside "
        "final_model_config.json.\n"
        f"Available keys:\n{list(config.keys())}"
    )


# ============================================================
# 1. LOAD FROZEN ARTIFACTS
# ============================================================

section(
    "1. LOADING FROZEN FINAL ARTIFACTS"
)

required_files = [
    TEST_FILE,
    MODEL_FILE,
    FEATURE_FILE,
    CONFIG_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{file_path}"
        )


test_df = pd.read_csv(
    TEST_FILE
)

model = joblib.load(
    MODEL_FILE
)

features = load_feature_list(
    FEATURE_FILE
)

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8",
) as f:

    config = json.load(f)


FROZEN_THRESHOLD = (
    load_frozen_threshold(
        config
    )
)


print("\nFrozen model:")
print(MODEL_FILE)

print("\nFrozen threshold:")
print(FROZEN_THRESHOLD)

print("\nNumber of frozen features:")
print(len(features))

print("\nTest rows:")
print(len(test_df))


# ============================================================
# 2. BASIC TEST INTEGRITY
# ============================================================

section(
    "2. TEST SET INTEGRITY"
)

if len(test_df) != EXPECTED_TEST_ROWS:

    raise RuntimeError(
        "\nTEST ROW COUNT CHANGED!\n"
        f"Expected: {EXPECTED_TEST_ROWS}\n"
        f"Found:    {len(test_df)}"
    )


missing_features = [
    feature
    for feature in features
    if feature not in test_df.columns
]

if missing_features:

    raise RuntimeError(
        "\nMissing frozen model features "
        "from test.csv:\n"
        + "\n".join(
            missing_features
        )
    )


target_col = find_target_column(
    test_df
)


y = convert_target_to_binary(
    test_df[target_col]
).to_numpy()


X = test_df[
    features
].copy()


nan_count = (
    X.isna()
    .sum()
    .sum()
)


inf_count = np.isinf(
    X.to_numpy(
        dtype=float
    )
).sum()


if nan_count != 0:

    raise RuntimeError(
        f"NaN values detected: {nan_count}"
    )


if inf_count != 0:

    raise RuntimeError(
        f"Infinite values detected: {inf_count}"
    )


print("Test set shape:")
print(X.shape)

print("\nTarget column:")
print(target_col)

print("\nBinary distribution:")
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print("\nNumerical QC:")
print(
    "NaN values:",
    nan_count,
)

print(
    "Infinite values:",
    inf_count,
)


# ============================================================
# 3. MODEL STRUCTURE CHECK
# ============================================================

section(
    "3. FROZEN MODEL STRUCTURE"
)


print("Model type:")
print(type(model))


classes = get_classifier_classes(
    model
)


print("\nClassifier classes:")
print(classes)


print("\nDecision function:")

if hasattr(
    model,
    "decision_function",
):

    print("AVAILABLE")

else:

    print("NOT AVAILABLE")


# ============================================================
# 4. FINAL TEST EXACT REPRODUCTION
# ============================================================

section(
    "4. FINAL TEST REPRODUCTION"
)


scores = get_decision_scores(
    model,
    X,
)


print("Decision score range:")

print(
    "Minimum:",
    scores.min(),
)

print(
    "Maximum:",
    scores.max(),
)

print(
    "Mean:",
    scores.mean(),
)

print(
    "Median:",
    np.median(scores),
)


final_metrics = calculate_metrics(
    y,
    scores,
    FROZEN_THRESHOLD,
)


print("\nFrozen threshold:")
print(
    FROZEN_THRESHOLD
)


print("\nFinal metrics:")

for key, value in final_metrics.items():

    print(
        f"{key}: {value}"
    )


actual_cm = (
    final_metrics["TN"],
    final_metrics["FP"],
    final_metrics["FN"],
    final_metrics["TP"],
)


expected_cm = (
    EXPECTED_TN,
    EXPECTED_FP,
    EXPECTED_FN,
    EXPECTED_TP,
)


print("\nExpected confusion matrix:")

print(
    "TN =",
    EXPECTED_TN,
)

print(
    "FP =",
    EXPECTED_FP,
)

print(
    "FN =",
    EXPECTED_FN,
)

print(
    "TP =",
    EXPECTED_TP,
)


print("\nActual confusion matrix:")

print(
    "TN =",
    actual_cm[0],
)

print(
    "FP =",
    actual_cm[1],
)

print(
    "FN =",
    actual_cm[2],
)

print(
    "TP =",
    actual_cm[3],
)


if actual_cm != expected_cm:

    raise RuntimeError(
        "\nFINAL TEST REPRODUCTION FAILED.\n"
        "Do NOT continue robustness analysis.\n"
        "The frozen final-test result was not reproduced."
    )


print(
    "\nFINAL TEST REPRODUCTION: PASS"
)


# ============================================================
# 5. BOOTSTRAP DESIGN
# ============================================================

section(
    "5. BOOTSTRAP DESIGN"
)


group_col = detect_group_column(
    test_df
)


if group_col is not None:

    bootstrap_type = (
        "GROUP-AWARE"
    )

    print(
        "Bootstrap type:"
    )

    print(
        bootstrap_type
    )

    print(
        "\nGrouping column:"
    )

    print(
        group_col
    )

    print(
        "\nUnique groups:"
    )

    print(
        test_df[
            group_col
        ].nunique()
    )

else:

    bootstrap_type = (
        "IMAGE-LEVEL"
    )

    print(
        "Bootstrap type:"
    )

    print(
        bootstrap_type
    )

    print(
        "\nWARNING:"
        "\nNo leakage-group column detected "
        "inside test.csv."
        "\nBootstrap will therefore be "
        "performed at image level."
    )


# ============================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

section(
    "6. BOOTSTRAP 95% CONFIDENCE INTERVALS"
)


rng = np.random.default_rng(
    RANDOM_STATE
)


metric_names = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "pr_auc",
]


bootstrap_values = {

    metric: []

    for metric
    in metric_names
}


if group_col is not None:

    groups = (
        test_df[
            group_col
        ]
        .astype(str)
        .to_numpy()
    )

    unique_groups = (
        np.unique(
            groups
        )
    )

    group_to_indices = {

        group:
            np.where(
                groups == group
            )[0]

        for group
        in unique_groups
    }


    for b in range(
        N_BOOTSTRAPS
    ):

        sampled_groups = (
            rng.choice(
                unique_groups,
                size=len(
                    unique_groups
                ),
                replace=True,
            )
        )

        idx_parts = [

            group_to_indices[
                group
            ]

            for group
            in sampled_groups
        ]

        idx = np.concatenate(
            idx_parts
        )

        y_b = y[idx]
        score_b = scores[idx]

        if len(
            np.unique(
                y_b
            )
        ) < 2:

            continue


        metrics_b = calculate_metrics(
            y_b,
            score_b,
            FROZEN_THRESHOLD,
        )


        for metric in metric_names:

            bootstrap_values[
                metric
            ].append(
                metrics_b[
                    metric
                ]
            )


        if (
            (b + 1) % 250
            == 0
        ):

            print(
                f"Bootstrap "
                f"{b + 1} / "
                f"{N_BOOTSTRAPS}"
            )


else:

    n = len(y)


    for b in range(
        N_BOOTSTRAPS
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        y_b = y[idx]
        score_b = scores[idx]


        if len(
            np.unique(
                y_b
            )
        ) < 2:

            continue


        metrics_b = calculate_metrics(
            y_b,
            score_b,
            FROZEN_THRESHOLD,
        )


        for metric in metric_names:

            bootstrap_values[
                metric
            ].append(
                metrics_b[
                    metric
                ]
            )


        if (
            (b + 1) % 250
            == 0
        ):

            print(
                f"Bootstrap "
                f"{b + 1} / "
                f"{N_BOOTSTRAPS}"
            )


ci_rows = []


for metric in metric_names:

    point_estimate = (
        final_metrics[
            metric
        ]
    )

    low, high = percentile_ci(
        bootstrap_values[
            metric
        ]
    )


    ci_rows.append({

        "metric":
            metric,

        "point_estimate":
            point_estimate,

        "ci_95_lower":
            low,

        "ci_95_upper":
            high,

        "bootstrap_samples":
            len(
                bootstrap_values[
                    metric
                ]
            ),

        "bootstrap_type":
            bootstrap_type,
    })


ci_df = pd.DataFrame(
    ci_rows
)


print(
    "\nBOOTSTRAP 95% CONFIDENCE INTERVALS"
)

print(
    ci_df.to_string(
        index=False
    )
)


ci_file = (
    OUTPUT_DIR
    / "bootstrap_95ci_final_test.csv"
)


ci_df.to_csv(
    ci_file,
    index=False,
)


# ============================================================
# 7. ORIGINAL CLASS SUBGROUP ROBUSTNESS
# ============================================================

section(
    "7. ORIGINAL-CLASS SUBGROUP ROBUSTNESS"
)


if "original_class" not in test_df.columns:

    raise KeyError(
        "original_class column "
        "is missing from test.csv."
    )


pred = (
    scores
    >= FROZEN_THRESHOLD
).astype(int)


subgroup_rows = []


original_classes = sorted(
    test_df[
        "original_class"
    ]
    .dropna()
    .unique()
)


for cls in original_classes:

    mask = (
        test_df[
            "original_class"
        ]
        == cls
    ).to_numpy()


    y_cls = y[
        mask
    ]

    pred_cls = pred[
        mask
    ]

    score_cls = scores[
        mask
    ]


    n = int(
        mask.sum()
    )


    correct = int(
        (
            y_cls
            == pred_cls
        ).sum()
    )


    errors = int(
        (
            y_cls
            != pred_cls
        ).sum()
    )


    row = {

        "original_class":
            cls,

        "n":
            n,

        "correct":
            correct,

        "errors":
            errors,

        "accuracy":
            correct / n,

        "mean_score":
            np.mean(
                score_cls
            ),

        "median_score":
            np.median(
                score_cls
            ),

        "score_q25":
            np.quantile(
                score_cls,
                0.25,
            ),

        "score_q75":
            np.quantile(
                score_cls,
                0.75,
            ),
    }


    # Demented class
    if np.all(
        y_cls == 1
    ):

        tp = int(
            (
                pred_cls == 1
            ).sum()
        )

        fn = int(
            (
                pred_cls == 0
            ).sum()
        )


        row["TP"] = tp
        row["FN"] = fn


        row["sensitivity"] = (
            tp / (tp + fn)
            if (tp + fn)
            else np.nan
        )


        row[
            "false_negative_rate"
        ] = (
            fn / (tp + fn)
            if (tp + fn)
            else np.nan
        )


    # Normal class
    elif np.all(
        y_cls == 0
    ):

        tn = int(
            (
                pred_cls == 0
            ).sum()
        )

        fp = int(
            (
                pred_cls == 1
            ).sum()
        )


        row["TN"] = tn
        row["FP"] = fp


        row["specificity"] = (
            tn / (tn + fp)
            if (tn + fp)
            else np.nan
        )


        row[
            "false_positive_rate"
        ] = (
            fp / (tn + fp)
            if (tn + fp)
            else np.nan
        )


    subgroup_rows.append(
        row
    )


subgroup_df = pd.DataFrame(
    subgroup_rows
)


print(
    subgroup_df.to_string(
        index=False
    )
)


subgroup_file = (
    OUTPUT_DIR
    / "original_class_subgroup_robustness.csv"
)


subgroup_df.to_csv(
    subgroup_file,
    index=False,
)


# ============================================================
# 8. THRESHOLD ROBUSTNESS
# ============================================================

section(
    "8. THRESHOLD ROBUSTNESS — DESCRIPTIVE ONLY"
)


print(
    "IMPORTANT:"
)

print(
    "The frozen threshold will NOT be changed."
)

print(
    "This analysis is descriptive only."
)


threshold_offsets = np.array([
    -0.20,
    -0.10,
    -0.05,
    -0.025,
     0.00,
     0.025,
     0.05,
     0.10,
     0.20,
])


thresholds = (
    FROZEN_THRESHOLD
    + threshold_offsets
)


threshold_rows = []


for threshold in thresholds:

    metrics_t = calculate_metrics(
        y,
        scores,
        float(
            threshold
        ),
    )


    threshold_rows.append({

        "threshold":
            threshold,

        "offset_from_frozen":
            threshold
            - FROZEN_THRESHOLD,

        "is_frozen_threshold":
            bool(
                np.isclose(
                    threshold,
                    FROZEN_THRESHOLD,
                )
            ),

        "accuracy":
            metrics_t[
                "accuracy"
            ],

        "balanced_accuracy":
            metrics_t[
                "balanced_accuracy"
            ],

        "sensitivity":
            metrics_t[
                "sensitivity"
            ],

        "specificity":
            metrics_t[
                "specificity"
            ],

        "precision":
            metrics_t[
                "precision"
            ],

        "f1":
            metrics_t[
                "f1"
            ],

        "TN":
            metrics_t[
                "TN"
            ],

        "FP":
            metrics_t[
                "FP"
            ],

        "FN":
            metrics_t[
                "FN"
            ],

        "TP":
            metrics_t[
                "TP"
            ],
    })


threshold_df = pd.DataFrame(
    threshold_rows
)


print(
    threshold_df.to_string(
        index=False
    )
)


threshold_file = (
    OUTPUT_DIR
    / "threshold_robustness_descriptive.csv"
)


threshold_df.to_csv(
    threshold_file,
    index=False,
)


# ============================================================
# 9. DECISION SCORE DISTRIBUTIONS
# ============================================================

section(
    "9. DECISION-SCORE DISTRIBUTION BY ORIGINAL CLASS"
)


score_df = test_df[
    [
        "original_class"
    ]
].copy()


if "filename" in test_df.columns:

    score_df[
        "filename"
    ] = test_df[
        "filename"
    ]


score_df[
    "true_binary"
] = y


score_df[
    "decision_score"
] = scores


score_df[
    "prediction"
] = pred


score_df[
    "correct"
] = (
    pred == y
)


score_summary = (
    score_df
    .groupby(
        "original_class"
    )[
        "decision_score"
    ]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max",
    ])
)


print(
    score_summary
)


score_summary_file = (
    OUTPUT_DIR
    / "decision_score_summary_by_original_class.csv"
)


score_summary.to_csv(
    score_summary_file
)


score_rows_file = (
    OUTPUT_DIR
    / "final_test_scores.csv"
)


score_df.to_csv(
    score_rows_file,
    index=False,
)


# ============================================================
# 10. FINAL ROBUSTNESS SUMMARY
# ============================================================

section(
    "10. FINAL ROBUSTNESS SUMMARY"
)


summary_lines = []


summary_lines.append(
    "FINAL ROBUSTNESS CHECKS"
)


summary_lines.append(
    "=" * 60
)


summary_lines.append(
    f"Test images: "
    f"{len(test_df)}"
)


summary_lines.append(
    f"Frozen threshold: "
    f"{FROZEN_THRESHOLD}"
)


summary_lines.append(
    f"Bootstrap type: "
    f"{bootstrap_type}"
)


summary_lines.append(
    f"Bootstrap repetitions: "
    f"{N_BOOTSTRAPS}"
)


summary_lines.append("")


summary_lines.append(
    "FINAL TEST CONFUSION MATRIX"
)


summary_lines.append(
    f"TN = "
    f"{final_metrics['TN']}"
)


summary_lines.append(
    f"FP = "
    f"{final_metrics['FP']}"
)


summary_lines.append(
    f"FN = "
    f"{final_metrics['FN']}"
)


summary_lines.append(
    f"TP = "
    f"{final_metrics['TP']}"
)


summary_lines.append("")


summary_lines.append(
    "FINAL TEST METRICS"
)


for metric in metric_names:

    summary_lines.append(
        f"{metric}: "
        f"{final_metrics[metric]:.6f}"
    )


summary_lines.append("")


summary_lines.append(
    "95% CONFIDENCE INTERVALS"
)


for _, row in ci_df.iterrows():

    summary_lines.append(

        f"{row['metric']}: "

        f"{row['point_estimate']:.6f} "

        f"[{row['ci_95_lower']:.6f}, "

        f"{row['ci_95_upper']:.6f}]"
    )


summary_lines.append("")


summary_lines.append(
    "MODEL / FEATURE SET / THRESHOLD:"
)


summary_lines.append(
    "FROZEN — NO CHANGES PERMITTED"
)


summary_file = (
    OUTPUT_DIR
    / "final_robustness_summary.txt"
)


summary_file.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8",
)


print(
    "\n".join(
        summary_lines
    )
)


# ============================================================
# 11. SAVE CONFIGURATION SNAPSHOT
# ============================================================

section(
    "11. ROBUSTNESS CONFIGURATION SNAPSHOT"
)


robustness_config = {

    "test_file":
        str(TEST_FILE),

    "model_file":
        str(MODEL_FILE),

    "feature_file":
        str(FEATURE_FILE),

    "config_file":
        str(CONFIG_FILE),

    "test_rows":
        len(test_df),

    "n_features":
        len(features),

    "frozen_threshold":
        FROZEN_THRESHOLD,

    "bootstrap_type":
        bootstrap_type,

    "bootstrap_repetitions":
        N_BOOTSTRAPS,

    "random_state":
        RANDOM_STATE,

    "expected_confusion_matrix": {

        "TN":
            EXPECTED_TN,

        "FP":
            EXPECTED_FP,

        "FN":
            EXPECTED_FN,

        "TP":
            EXPECTED_TP,
    },

    "actual_confusion_matrix": {

        "TN":
            final_metrics[
                "TN"
            ],

        "FP":
            final_metrics[
                "FP"
            ],

        "FN":
            final_metrics[
                "FN"
            ],

        "TP":
            final_metrics[
                "TP"
            ],
    },

    "final_test_reproduction":
        "PASS",
}


robustness_config_file = (
    OUTPUT_DIR
    / "robustness_configuration.json"
)


with open(
    robustness_config_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        robustness_config,
        f,
        indent=4,
    )


print(
    robustness_config_file
)


# ============================================================
# 12. FILES SAVED
# ============================================================

section(
    "12. FILES SAVED"
)


saved_files = [

    ci_file,

    subgroup_file,

    threshold_file,

    score_summary_file,

    score_rows_file,

    summary_file,

    robustness_config_file,
]


for path in saved_files:

    print(path)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "FINAL ROBUSTNESS CHECKS COMPLETE"
)

print(
    "=" * 72
)


print(
    "\nIMPORTANT:"
)

print(
    "The frozen model was NOT modified."
)

print(
    "The frozen feature set was NOT modified."
)

print(
    "The frozen threshold was NOT modified."
)

print(
    "The final test set was used only for final "
    "evaluation and robustness analysis."
)