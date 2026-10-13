from pathlib import Path
import json
import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\Users\97338\Desktop\RAW DATA")

TEST_FILE = ROOT / "results" / "GROUP_AWARE_SPLIT" / "test.csv"

FINAL_MODEL_DIR = ROOT / "results" / "FINAL_MODEL"

MODEL_FILE = FINAL_MODEL_DIR / "final_model.joblib"
FEATURE_FILE = FINAL_MODEL_DIR / "final_features.txt"
CONFIG_FILE = FINAL_MODEL_DIR / "final_model_config.json"
FREEZE_MANIFEST_FILE = FINAL_MODEL_DIR / "freeze_manifest.json"

OUTPUT_DIR = ROOT / "results" / "STRUCTURED_ERROR_ANALYSIS"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Frozen threshold from the completed threshold-optimization /
# model-freeze stage.
FALLBACK_THRESHOLD = 0.108724729218


# These MUST match the already completed one-time final test.
EXPECTED_CONFUSION = {
    "TN": 1206,
    "FP": 293,
    "FN": 549,
    "TP": 2432,
}

EXPECTED_TEST_ROWS = 4480
EXPECTED_FEATURES = 30


# Full frozen feature set
EXPECTED_FEATURE_NAMES = [
    "brain_area",
    "brain_ratio",
    "brain_perimeter",
    "compactness",
    "bbox_aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "median_intensity",
    "entropy",
    "p10",
    "p25",
    "p75",
    "p90",
    "edge_density",
    "central_mean",
    "central_std",
    "central_median",
    "central_entropy",
    "central_dark_ratio",
    "central_bright_ratio",
    "left_mean",
    "right_mean",
    "left_dark_ratio",
    "right_dark_ratio",
    "intensity_asymmetry",
    "dark_ratio_asymmetry",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def separator(title=None):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def sha256_file(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            sha.update(block)

    return sha.hexdigest()


def find_numeric_threshold(obj):
    """
    Recursively search a JSON-like object for a numeric threshold.
    """
    if isinstance(obj, dict):

        preferred_keys = [
            "final_threshold",
            "frozen_threshold",
            "decision_threshold",
            "threshold",
        ]

        for key in preferred_keys:
            if key in obj:
                value = obj[key]

                if isinstance(value, (int, float)):
                    return float(value)

        for value in obj.values():
            found = find_numeric_threshold(value)

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:
            found = find_numeric_threshold(value)

            if found is not None:
                return found

    return None


def load_frozen_threshold():
    threshold = None

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)

            threshold = find_numeric_threshold(config)

        except Exception as exc:
            print(
                "Warning: could not extract threshold from "
                f"final_model_config.json: {exc}"
            )

    if threshold is None:
        threshold = FALLBACK_THRESHOLD

    return float(threshold)


def load_feature_names(test_df):
    """
    Prefer final_features.txt, but verify against the actual test dataframe.
    """

    feature_names = []

    if FEATURE_FILE.exists():

        lines = FEATURE_FILE.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            cleaned = (
                line.strip()
                .strip("'")
                .strip('"')
                .strip(",")
            )

            if cleaned in test_df.columns:
                feature_names.append(cleaned)

    # Remove accidental duplicates while preserving order
    feature_names = list(dict.fromkeys(feature_names))

    # Safe fallback to the frozen known 30 features
    if len(feature_names) != EXPECTED_FEATURES:

        fallback = [
            f for f in EXPECTED_FEATURE_NAMES
            if f in test_df.columns
        ]

        if len(fallback) == EXPECTED_FEATURES:
            feature_names = fallback

    return feature_names


def determine_target(test_df):

    if "target_name" in test_df.columns:

        mapping = {
            "Normal": 0,
            "Early_Demented": 1,
        }

        unknown = set(
            test_df["target_name"].dropna().unique()
        ) - set(mapping.keys())

        if unknown:
            raise ValueError(
                f"Unknown target_name values: {unknown}"
            )

        return test_df["target_name"].map(mapping).astype(int)

    candidate_columns = [
        "target",
        "binary_target",
        "label",
        "y",
    ]

    for column in candidate_columns:

        if column in test_df.columns:

            values = sorted(
                pd.Series(test_df[column])
                .dropna()
                .unique()
                .tolist()
            )

            if set(values).issubset({0, 1}):
                return test_df[column].astype(int)

    raise ValueError(
        "Could not determine binary target column."
    )


def assign_outcome(y_true, y_pred):

    conditions = [
        (y_true == 0) & (y_pred == 0),
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 0),
        (y_true == 1) & (y_pred == 1),
    ]

    labels = [
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    return np.select(
        conditions,
        labels,
        default="UNKNOWN",
    )


def standardized_mean_difference(group_a, group_b):
    """
    Cohen-style standardized mean difference.

    Positive:
        group_a > group_b

    Negative:
        group_a < group_b

    Used descriptively only.
    """

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    pooled_variance = (
        ((len(a) - 1) * var_a)
        +
        ((len(b) - 1) * var_b)
    ) / (
        len(a) + len(b) - 2
    )

    if pooled_variance <= 0:
        return 0.0

    pooled_std = np.sqrt(pooled_variance)

    return (
        np.mean(a) - np.mean(b)
    ) / pooled_std


def build_feature_difference_table(
    dataframe,
    feature_names,
    group_a_name,
    group_b_name,
):

    a = dataframe[
        dataframe["outcome"] == group_a_name
    ]

    b = dataframe[
        dataframe["outcome"] == group_b_name
    ]

    rows = []

    for feature in feature_names:

        a_values = a[feature].astype(float)
        b_values = b[feature].astype(float)

        smd = standardized_mean_difference(
            a_values,
            b_values,
        )

        rows.append({
            "feature": feature,

            f"{group_a_name}_mean":
                float(a_values.mean()),

            f"{group_b_name}_mean":
                float(b_values.mean()),

            "mean_difference":
                float(
                    a_values.mean()
                    -
                    b_values.mean()
                ),

            f"{group_a_name}_median":
                float(a_values.median()),

            f"{group_b_name}_median":
                float(b_values.median()),

            "median_difference":
                float(
                    a_values.median()
                    -
                    b_values.median()
                ),

            "standardized_mean_difference":
                float(smd),

            "absolute_standardized_difference":
                float(abs(smd)),
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "absolute_standardized_difference",
        ascending=False,
    ).reset_index(drop=True)

    return result


def save_feature_difference_plot(
    table,
    title,
    output_file,
    top_n=15,
):

    plot_df = (
        table
        .head(top_n)
        .sort_values(
            "absolute_standardized_difference",
            ascending=True,
        )
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_df["feature"],
        plot_df["standardized_mean_difference"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Standardized mean difference"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# START
# ============================================================

separator("STRUCTURED ERROR ANALYSIS")

print("IMPORTANT:")
print("This script performs POST-HOC analysis only.")
print("It does NOT retrain the model.")
print("It does NOT change the threshold.")
print("It does NOT select new features.")
print("It does NOT modify the final model.")


# ============================================================
# 1. REQUIRED FILE CHECK
# ============================================================

separator("1. REQUIRED FILE CHECK")

required_files = [
    TEST_FILE,
    MODEL_FILE,
]

for file_path in required_files:

    print(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

print("\nRequired files: PASS")


# ============================================================
# 2. LOAD TEST SET
# ============================================================

separator("2. FINAL TEST SET")

test = pd.read_csv(TEST_FILE)

print("Rows:")
print(len(test))

if len(test) != EXPECTED_TEST_ROWS:
    raise RuntimeError(
        "Unexpected test-set size.\n"
        f"Expected: {EXPECTED_TEST_ROWS}\n"
        f"Found: {len(test)}"
    )

print("\nTest-set size integrity: PASS")


# ============================================================
# 3. FEATURE LIST
# ============================================================

separator("3. FROZEN FEATURE SET")

feature_names = load_feature_names(test)

print("Number of features:")
print(len(feature_names))

for i, feature in enumerate(
    feature_names,
    start=1,
):
    print(
        f"{i:02d}. {feature}"
    )

if len(feature_names) != EXPECTED_FEATURES:

    raise RuntimeError(
        "\nFrozen feature count mismatch.\n"
        f"Expected: {EXPECTED_FEATURES}\n"
        f"Found: {len(feature_names)}"
    )

missing = [
    f for f in feature_names
    if f not in test.columns
]

if missing:

    raise RuntimeError(
        "Missing test features:\n"
        + "\n".join(missing)
    )

print("\nFeature integrity: PASS")


# ============================================================
# 4. NUMERICAL QC
# ============================================================

separator("4. TEST NUMERICAL QC")

X_df = test[feature_names].copy()

X = X_df.to_numpy(dtype=float)

nan_count = int(
    np.isnan(X).sum()
)

inf_count = int(
    np.isinf(X).sum()
)

print("Shape:")
print(X.shape)

print("\nNaN values:")
print(nan_count)

print("\nInfinite values:")
print(inf_count)

if nan_count != 0 or inf_count != 0:
    raise RuntimeError(
        "Numerical QC failed."
    )

print("\nNumerical QC: PASS")


# ============================================================
# 5. LOAD FROZEN MODEL + THRESHOLD
# ============================================================

separator("5. FROZEN MODEL")

model = joblib.load(MODEL_FILE)

threshold = load_frozen_threshold()

print("Model loaded:")
print(MODEL_FILE)

print("\nModel SHA256:")
print(
    sha256_file(MODEL_FILE)
)

print("\nFrozen threshold:")
print(
    f"{threshold:.12f}"
)

if not hasattr(model, "decision_function"):
    raise RuntimeError(
        "Frozen model does not expose decision_function()."
    )


# ============================================================
# 6. RECONSTRUCT FINAL TEST
# ============================================================

separator("6. FINAL TEST RECONSTRUCTION")

y_true = determine_target(test).to_numpy()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    scores = np.asarray(
        model.decision_function(X)
    ).reshape(-1)

y_pred = (
    scores >= threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()

actual_confusion = {
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
}

print("Confusion Matrix:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print("\nExpected:")
for key, value in EXPECTED_CONFUSION.items():
    print(
        f"{key} = {value}"
    )

if actual_confusion != EXPECTED_CONFUSION:

    raise RuntimeError(
        "\nFINAL TEST RECONSTRUCTION FAILED.\n"
        "Do not continue error analysis because the "
        "predictions do not match the frozen final test."
    )

print(
    "\nFINAL TEST RECONSTRUCTION: PASS"
)


# ============================================================
# 7. BUILD MASTER PREDICTION TABLE
# ============================================================

separator("7. BUILDING ERROR-ANALYSIS TABLE")

analysis = test.copy()

analysis["true_binary"] = y_true
analysis["predicted_binary"] = y_pred

analysis["true_target"] = np.where(
    y_true == 1,
    "Early_Demented",
    "Normal",
)

analysis["predicted_target"] = np.where(
    y_pred == 1,
    "Early_Demented",
    "Normal",
)

analysis["decision_score"] = scores

analysis["threshold"] = threshold

analysis["signed_margin"] = (
    analysis["decision_score"]
    -
    threshold
)

analysis["absolute_margin"] = np.abs(
    analysis["signed_margin"]
)

analysis["outcome"] = assign_outcome(
    y_true,
    y_pred,
)

analysis["is_correct"] = (
    y_true == y_pred
)

analysis["is_error"] = (
    y_true != y_pred
)

MASTER_FILE = (
    OUTPUT_DIR
    /
    "all_test_predictions_with_error_analysis.csv"
)

analysis.to_csv(
    MASTER_FILE,
    index=False,
)

print("Master table saved:")
print(MASTER_FILE)


# ============================================================
# 8. OVERALL ERROR SUMMARY
# ============================================================

separator("8. OVERALL ERROR SUMMARY")

outcome_counts = (
    analysis["outcome"]
    .value_counts()
    .reindex(
        ["TP", "TN", "FP", "FN"],
        fill_value=0,
    )
)

print(outcome_counts)

total_errors = int(
    analysis["is_error"].sum()
)

error_percentage = (
    total_errors
    /
    len(analysis)
    *
    100
)

print("\nTotal test errors:")
print(total_errors)

print("\nTotal error percentage:")
print(
    f"{error_percentage:.3f}%"
)

errors = analysis[
    analysis["is_error"]
].copy()

ERROR_FILE = (
    OUTPUT_DIR
    /
    "all_error_cases.csv"
)

errors.to_csv(
    ERROR_FILE,
    index=False,
)

print("\nError table saved:")
print(ERROR_FILE)


# ============================================================
# 9. ERRORS BY ORIGINAL CLASS
# ============================================================

separator("9. ERRORS BY ORIGINAL CLASS")

if "original_class" not in analysis.columns:

    raise RuntimeError(
        "original_class column is required "
        "for structured error analysis."
    )

class_summary = (
    analysis
    .groupby("original_class")
    .agg(
        total_images=("outcome", "size"),
        error_count=("is_error", "sum"),
        mean_score=("decision_score", "mean"),
        median_score=("decision_score", "median"),
        mean_abs_margin=("absolute_margin", "mean"),
        median_abs_margin=("absolute_margin", "median"),
    )
    .reset_index()
)

class_summary["error_rate_percent"] = (
    class_summary["error_count"]
    /
    class_summary["total_images"]
    *
    100
)

fp_counts = (
    analysis[
        analysis["outcome"] == "FP"
    ]
    ["original_class"]
    .value_counts()
)

fn_counts = (
    analysis[
        analysis["outcome"] == "FN"
    ]
    ["original_class"]
    .value_counts()
)

class_summary["FP"] = (
    class_summary["original_class"]
    .map(fp_counts)
    .fillna(0)
    .astype(int)
)

class_summary["FN"] = (
    class_summary["original_class"]
    .map(fn_counts)
    .fillna(0)
    .astype(int)
)

class_summary = class_summary.sort_values(
    "error_rate_percent",
    ascending=False,
).reset_index(drop=True)

print(
    class_summary.to_string(
        index=False
    )
)

CLASS_SUMMARY_FILE = (
    OUTPUT_DIR
    /
    "error_rates_by_original_class.csv"
)

class_summary.to_csv(
    CLASS_SUMMARY_FILE,
    index=False,
)


# ============================================================
# 10. SCORE / MARGIN SUMMARY BY OUTCOME
# ============================================================

separator("10. DECISION SCORE SUMMARY BY OUTCOME")

margin_summary = (
    analysis
    .groupby("outcome")
    .agg(
        count=("decision_score", "size"),

        score_min=("decision_score", "min"),
        score_mean=("decision_score", "mean"),
        score_median=("decision_score", "median"),
        score_max=("decision_score", "max"),

        abs_margin_min=("absolute_margin", "min"),
        abs_margin_mean=("absolute_margin", "mean"),
        abs_margin_median=("absolute_margin", "median"),
        abs_margin_max=("absolute_margin", "max"),
    )
    .reindex(
        ["TP", "TN", "FP", "FN"]
    )
)

print(
    margin_summary.to_string()
)

MARGIN_SUMMARY_FILE = (
    OUTPUT_DIR
    /
    "decision_score_margin_summary.csv"
)

margin_summary.to_csv(
    MARGIN_SUMMARY_FILE
)


# ============================================================
# 11. HOW MANY ERRORS ARE CLOSE TO THRESHOLD?
# ============================================================

separator("11. ERROR DISTANCE FROM THRESHOLD")

# These are descriptive score-distance windows only.
# They are NOT new decision thresholds.
distance_windows = [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
]

distance_rows = []

for window in distance_windows:

    subset = errors[
        errors["absolute_margin"] <= window
    ]

    fp_n = int(
        (
            subset["outcome"] == "FP"
        ).sum()
    )

    fn_n = int(
        (
            subset["outcome"] == "FN"
        ).sum()
    )

    distance_rows.append({
        "absolute_margin_leq": window,
        "all_errors": len(subset),
        "percentage_of_all_errors":
            len(subset)
            /
            len(errors)
            *
            100,
        "FP": fp_n,
        "FN": fn_n,
    })


distance_table = pd.DataFrame(
    distance_rows
)

print(
    distance_table.to_string(
        index=False
    )
)

DISTANCE_FILE = (
    OUTPUT_DIR
    /
    "error_distance_from_threshold.csv"
)

distance_table.to_csv(
    DISTANCE_FILE,
    index=False,
)


# ============================================================
# 12. CLOSEST ERRORS TO THRESHOLD
# ============================================================

separator("12. CLOSEST ERRORS TO THRESHOLD")

display_columns = [
    col
    for col in [
        "filename",
        "original_class",
        "true_target",
        "predicted_target",
        "outcome",
        "decision_score",
        "signed_margin",
        "absolute_margin",
    ]
    if col in errors.columns
]

closest_errors = (
    errors
    .sort_values(
        "absolute_margin",
        ascending=True,
    )
    .head(20)
)

print(
    closest_errors[
        display_columns
    ].to_string(
        index=False
    )
)

CLOSEST_FILE = (
    OUTPUT_DIR
    /
    "20_closest_errors_to_threshold.csv"
)

closest_errors.to_csv(
    CLOSEST_FILE,
    index=False,
)


# ============================================================
# 13. MOST CONFIDENT ERRORS
# ============================================================

separator("13. MOST CONFIDENT ERRORS")

most_confident_errors = (
    errors
    .sort_values(
        "absolute_margin",
        ascending=False,
    )
    .head(20)
)

print(
    most_confident_errors[
        display_columns
    ].to_string(
        index=False
    )
)

CONFIDENT_ERROR_FILE = (
    OUTPUT_DIR
    /
    "20_most_confident_errors.csv"
)

most_confident_errors.to_csv(
    CONFIDENT_ERROR_FILE,
    index=False,
)


# ============================================================
# 14. FALSE NEGATIVES BY DEMENTIA STAGE
# ============================================================

separator("14. FALSE NEGATIVES BY ORIGINAL DEMENTIA STAGE")

fn_cases = analysis[
    analysis["outcome"] == "FN"
].copy()

fn_by_stage = (
    fn_cases[
        "original_class"
    ]
    .value_counts()
)

print(fn_by_stage)

positive_class_totals = (
    analysis[
        analysis["true_binary"] == 1
    ]
    ["original_class"]
    .value_counts()
)

fn_stage_rows = []

for stage, total in positive_class_totals.items():

    fn_stage_count = int(
        fn_by_stage.get(
            stage,
            0,
        )
    )

    fn_stage_rows.append({
        "original_class": stage,
        "total_test_images": int(total),
        "false_negatives": fn_stage_count,
        "false_negative_rate_percent":
            fn_stage_count
            /
            int(total)
            *
            100,
    })

fn_stage_table = pd.DataFrame(
    fn_stage_rows
).sort_values(
    "false_negative_rate_percent",
    ascending=False,
)

print(
    "\nFalse-negative rates:"
)

print(
    fn_stage_table.to_string(
        index=False
    )
)

FN_STAGE_FILE = (
    OUTPUT_DIR
    /
    "false_negative_rate_by_dementia_stage.csv"
)

fn_stage_table.to_csv(
    FN_STAGE_FILE,
    index=False,
)


# ============================================================
# 15. FP vs TN FEATURE DIFFERENCES
# ============================================================

separator("15. FP vs TN FEATURE DIFFERENCES")

fp_vs_tn = build_feature_difference_table(
    dataframe=analysis,
    feature_names=feature_names,
    group_a_name="FP",
    group_b_name="TN",
)

print(
    fp_vs_tn[
        [
            "feature",
            "FP_mean",
            "TN_mean",
            "standardized_mean_difference",
            "absolute_standardized_difference",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)

FP_TN_FILE = (
    OUTPUT_DIR
    /
    "feature_differences_FP_vs_TN.csv"
)

fp_vs_tn.to_csv(
    FP_TN_FILE,
    index=False,
)


# ============================================================
# 16. FN vs TP FEATURE DIFFERENCES
# ============================================================

separator("16. FN vs TP FEATURE DIFFERENCES")

fn_vs_tp = build_feature_difference_table(
    dataframe=analysis,
    feature_names=feature_names,
    group_a_name="FN",
    group_b_name="TP",
)

print(
    fn_vs_tp[
        [
            "feature",
            "FN_mean",
            "TP_mean",
            "standardized_mean_difference",
            "absolute_standardized_difference",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)

FN_TP_FILE = (
    OUTPUT_DIR
    /
    "feature_differences_FN_vs_TP.csv"
)

fn_vs_tp.to_csv(
    FN_TP_FILE,
    index=False,
)


# ============================================================
# 17. PLOT: DECISION SCORES BY OUTCOME
# ============================================================

separator("17. CREATING PLOTS")

outcome_order = [
    "TN",
    "FP",
    "FN",
    "TP",
]

score_data = [
    analysis.loc[
        analysis["outcome"] == outcome,
        "decision_score",
    ].to_numpy()
    for outcome in outcome_order
]

plt.figure(
    figsize=(9, 6)
)

plt.boxplot(
    score_data,
    tick_labels=outcome_order,
    showfliers=False,
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=1.5,
)

plt.xlabel(
    "Final-test outcome"
)

plt.ylabel(
    "Frozen SVM decision score"
)

plt.title(
    "Decision Score Distribution by Final-Test Outcome"
)

plt.tight_layout()

SCORE_PLOT = (
    PLOTS_DIR
    /
    "decision_scores_by_outcome.png"
)

plt.savefig(
    SCORE_PLOT,
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 18. PLOT: ABSOLUTE MARGIN BY OUTCOME
# ============================================================

margin_data = [
    analysis.loc[
        analysis["outcome"] == outcome,
        "absolute_margin",
    ].to_numpy()
    for outcome in outcome_order
]

plt.figure(
    figsize=(9, 6)
)

plt.boxplot(
    margin_data,
    tick_labels=outcome_order,
    showfliers=False,
)

plt.xlabel(
    "Final-test outcome"
)

plt.ylabel(
    "Absolute distance from frozen threshold"
)

plt.title(
    "Distance from Decision Threshold by Outcome"
)

plt.tight_layout()

MARGIN_PLOT = (
    PLOTS_DIR
    /
    "absolute_margin_by_outcome.png"
)

plt.savefig(
    MARGIN_PLOT,
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 19. PLOT: ERROR RATE BY ORIGINAL CLASS
# ============================================================

plot_class_summary = class_summary.sort_values(
    "error_rate_percent",
    ascending=True,
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    plot_class_summary["original_class"],
    plot_class_summary["error_rate_percent"],
)

plt.xlabel(
    "Error rate (%)"
)

plt.ylabel(
    "Original class"
)

plt.title(
    "Final-Test Error Rate by Original Class"
)

plt.tight_layout()

CLASS_ERROR_PLOT = (
    PLOTS_DIR
    /
    "error_rate_by_original_class.png"
)

plt.savefig(
    CLASS_ERROR_PLOT,
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 20. PLOTS: FEATURE SHIFTS
# ============================================================

save_feature_difference_plot(
    table=fp_vs_tn,
    title=(
        "Feature Shifts: False Positives vs True Negatives"
    ),
    output_file=(
        PLOTS_DIR
        /
        "feature_shifts_FP_vs_TN.png"
    ),
)

save_feature_difference_plot(
    table=fn_vs_tp,
    title=(
        "Feature Shifts: False Negatives vs True Positives"
    ),
    output_file=(
        PLOTS_DIR
        /
        "feature_shifts_FN_vs_TP.png"
    ),
)


# ============================================================
# 21. SAVE SUMMARY JSON
# ============================================================

summary = {

    "test_rows": int(
        len(analysis)
    ),

    "threshold": float(
        threshold
    ),

    "confusion_matrix": actual_confusion,

    "total_errors": total_errors,

    "error_percentage": float(
        error_percentage
    ),

    "number_of_features": int(
        len(feature_names)
    ),

    "model_sha256": sha256_file(
        MODEL_FILE
    ),

    "final_test_reconstruction_passed":
        True,

    "important_note": (
        "Post-hoc final-test error analysis only. "
        "No model, threshold, feature, preprocessing, "
        "or hyperparameter modification is permitted "
        "based on these results."
    ),
}

SUMMARY_JSON_FILE = (
    OUTPUT_DIR
    /
    "structured_error_analysis_summary.json"
)

with open(
    SUMMARY_JSON_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

separator("STRUCTURED ERROR ANALYSIS COMPLETE")

print("Frozen final-test confusion matrix:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print("\nTotal errors:")
print(total_errors)

print("\nError rate:")
print(
    f"{error_percentage:.3f}%"
)

print("\nOutputs saved inside:")
print(OUTPUT_DIR)

print("\nPlots saved inside:")
print(PLOTS_DIR)

print(
    "\nIMPORTANT:"
)

print(
    "These test-set findings are for scientific "
    "interpretation only."
)

print(
    "Do NOT retrain, retune, change features, "
    "or move the frozen threshold after this analysis."
)

print(
    "\nNEXT STEP AFTER REVIEW:"
)

print(
    "FINAL ROBUSTNESS CHECKS"
)