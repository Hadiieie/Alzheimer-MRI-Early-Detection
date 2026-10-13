# ============================================================
# GLOBAL MODEL EXPLAINABILITY
# Final Frozen RBF-SVM
#
# Purpose:
#   - Explain which features are globally important
#   - Use VALIDATION set only for global explainability
#   - DO NOT retrain or modify final model
#   - DO NOT modify frozen threshold
#
# Outputs:
#   results/EXPLAINABILITY/
#       global_permutation_importance.csv
#       global_importance_roc_auc.png
#       global_importance_balanced_accuracy.png
#       explainability_summary.txt
# ============================================================

from pathlib import Path
import json
import re
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\97338\Desktop\RAW DATA")

FINAL_MODEL_DIR = ROOT / "results" / "FINAL_MODEL"
SPLIT_DIR = ROOT / "results" / "GROUP_AWARE_SPLIT"
OUTPUT_DIR = ROOT / "results" / "EXPLAINABILITY"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = FINAL_MODEL_DIR / "final_model.joblib"
FEATURE_FILE = FINAL_MODEL_DIR / "final_features.txt"
CONFIG_FILE = FINAL_MODEL_DIR / "final_model_config.json"

VALIDATION_FILE = SPLIT_DIR / "validation.csv"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Number of times each feature is shuffled.
# 5 is a good scientific starting point.
N_REPEATS = 5

# None = use all validation images.
# If execution is extremely slow, temporarily change to 2000.
MAX_ANALYSIS_ROWS = None

# Number of features to show in plots
TOP_N = 20


# ============================================================
# HELPER: LOAD FEATURE NAMES
# ============================================================

def clean_feature_line(line):
    """
    Cleans possible formats like:
        01. brain_area
        KEEP: brain_area
        brain_area
    """

    line = line.strip()

    if not line:
        return None

    line = re.sub(r"^\d+\s*[\.\)]\s*", "", line)
    line = re.sub(r"^(KEEP|FEATURE)\s*:\s*", "", line, flags=re.I)

    return line.strip()


def load_feature_names():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    features = []

    for line in lines:

        cleaned = clean_feature_line(line)

        if cleaned:
            features.append(cleaned)

    # Remove duplicates while preserving order
    features = list(dict.fromkeys(features))

    return features


# ============================================================
# LOAD FROZEN THRESHOLD
# ============================================================

def load_threshold():

    # Exact frozen threshold from previous validated step
    fallback_threshold = 0.108724729218

    if not CONFIG_FILE.exists():
        print(
            "WARNING: final_model_config.json not found.\n"
            f"Using frozen fallback threshold: {fallback_threshold}"
        )
        return fallback_threshold

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
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

            print(
                f"Frozen threshold loaded from configuration: "
                f"{threshold}"
            )

            return threshold

    print(
        "WARNING: threshold key not found in config.\n"
        f"Using frozen fallback threshold: {fallback_threshold}"
    )

    return fallback_threshold


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("GLOBAL MODEL EXPLAINABILITY")
print("=" * 70)

print()
print("IMPORTANT:")
print("The final model will NOT be retrained.")
print("The final threshold will NOT be changed.")
print("TEST set will NOT be used in this step.")

print()
print("=" * 70)
print("LOADING FROZEN ARTIFACTS")
print("=" * 70)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Final model not found:\n{MODEL_FILE}"
    )

if not VALIDATION_FILE.exists():
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

model = joblib.load(MODEL_FILE)

features = load_feature_names()
threshold = load_threshold()

print()
print("Model:")
print(MODEL_FILE)

print()
print("Number of frozen features:")
print(len(features))

print()
print("Frozen threshold:")
print(threshold)

if len(features) != 30:

    print()
    print("WARNING:")
    print(
        f"Expected 30 frozen features but feature file contains "
        f"{len(features)}."
    )


# ============================================================
# LOAD VALIDATION
# ============================================================

df = pd.read_csv(VALIDATION_FILE)

print()
print("=" * 70)
print("VALIDATION DATASET")
print("=" * 70)

print()
print("Rows:")
print(len(df))

print()
print("Columns:")
print(len(df.columns))


# ============================================================
# CHECK TARGET
# ============================================================

if "target_name" not in df.columns:
    raise ValueError(
        "target_name column not found in validation.csv"
    )

print()
print("Target distribution:")
print(df["target_name"].value_counts())

expected_targets = {"Normal", "Early_Demented"}

actual_targets = set(df["target_name"].dropna().unique())

if not actual_targets.issubset(expected_targets):

    raise ValueError(
        "Unexpected target labels found:\n"
        f"{actual_targets}"
    )

y = (
    df["target_name"]
    .map({
        "Normal": 0,
        "Early_Demented": 1
    })
    .to_numpy(dtype=int)
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print()
    print("Missing features:")

    for f in missing_features:
        print(f" - {f}")

    raise ValueError(
        "Some frozen model features are missing."
    )


# Use exact frozen feature order
X_df = df[features].copy()

# Convert to numpy intentionally:
# avoids sklearn feature-name warning while preserving order
X = X_df.to_numpy(dtype=float)


# ============================================================
# NUMERICAL QC
# ============================================================

print()
print("=" * 70)
print("NUMERICAL QC")
print("=" * 70)

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()

print()
print("Shape:")
print(X.shape)

print()
print("NaN values:")
print(nan_count)

print()
print("Infinite values:")
print(inf_count)

if nan_count != 0:
    raise ValueError("NaN values detected.")

if inf_count != 0:
    raise ValueError("Infinite values detected.")


# ============================================================
# OPTIONAL STRATIFIED SUBSAMPLE
# ============================================================

if MAX_ANALYSIS_ROWS is not None and len(X) > MAX_ANALYSIS_ROWS:

    print()
    print("=" * 70)
    print("CREATING STRATIFIED EXPLAINABILITY SAMPLE")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_STATE)

    selected_indices = []

    for label in np.unique(y):

        label_indices = np.where(y == label)[0]

        n_label = round(
            MAX_ANALYSIS_ROWS *
            len(label_indices) /
            len(y)
        )

        selected = rng.choice(
            label_indices,
            size=n_label,
            replace=False
        )

        selected_indices.extend(selected.tolist())

    selected_indices = np.array(selected_indices)

    rng.shuffle(selected_indices)

    X = X[selected_indices]
    y = y[selected_indices]

    print()
    print("Images used:")
    print(len(X))

    print()
    print("Class distribution:")
    print(pd.Series(y).value_counts().sort_index())


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("BASELINE PERFORMANCE")
print("=" * 70)

start = time.time()

baseline_scores = model.decision_function(X)

baseline_predictions = (
    baseline_scores >= threshold
).astype(int)

baseline_auc = roc_auc_score(
    y,
    baseline_scores
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y,
    baseline_predictions
)

print()
print("Baseline ROC-AUC:")
print(f"{baseline_auc:.6f}")

print()
print("Baseline balanced accuracy:")
print(f"{baseline_balanced_accuracy:.6f}")

print()
print("Frozen threshold:")
print(f"{threshold:.12f}")

print()
print(
    f"Baseline calculation time: "
    f"{time.time() - start:.2f} seconds"
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("PERMUTATION IMPORTANCE")
print("=" * 70)

print()
print(f"Features: {len(features)}")
print(f"Repeats per feature: {N_REPEATS}")

print()
print(
    "Interpretation:\n"
    "A larger performance drop after shuffling a feature means\n"
    "the frozen model depends more strongly on that feature."
)

rng = np.random.default_rng(RANDOM_STATE)

results = []

overall_start = time.time()

for feature_index, feature_name in enumerate(features):

    roc_drops = []
    bal_acc_drops = []

    feature_start = time.time()

    for repeat in range(N_REPEATS):

        X_perm = X.copy()

        shuffled_column = X_perm[:, feature_index].copy()

        rng.shuffle(shuffled_column)

        X_perm[:, feature_index] = shuffled_column

        perm_scores = model.decision_function(X_perm)

        perm_predictions = (
            perm_scores >= threshold
        ).astype(int)

        perm_auc = roc_auc_score(
            y,
            perm_scores
        )

        perm_balanced_accuracy = balanced_accuracy_score(
            y,
            perm_predictions
        )

        roc_drop = baseline_auc - perm_auc

        bal_acc_drop = (
            baseline_balanced_accuracy
            - perm_balanced_accuracy
        )

        roc_drops.append(roc_drop)
        bal_acc_drops.append(bal_acc_drop)

    row = {
        "feature": feature_name,

        "roc_auc_drop_mean": np.mean(roc_drops),
        "roc_auc_drop_std": np.std(roc_drops),

        "balanced_accuracy_drop_mean":
            np.mean(bal_acc_drops),

        "balanced_accuracy_drop_std":
            np.std(bal_acc_drops)
    }

    results.append(row)

    print(
        f"[{feature_index + 1:02d}/{len(features):02d}] "
        f"{feature_name:<25} | "
        f"ROC-AUC drop = "
        f"{row['roc_auc_drop_mean']:.6f} | "
        f"Balanced Acc drop = "
        f"{row['balanced_accuracy_drop_mean']:.6f} | "
        f"{time.time() - feature_start:.1f}s"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

importance_df = pd.DataFrame(results)

importance_df = importance_df.sort_values(
    by="roc_auc_drop_mean",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0,
    "rank_roc_auc",
    np.arange(1, len(importance_df) + 1)
)

# Separate balanced-accuracy ranking
bal_rank = (
    importance_df["balanced_accuracy_drop_mean"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

importance_df.insert(
    1,
    "rank_balanced_accuracy",
    bal_rank
)

csv_file = (
    OUTPUT_DIR /
    "global_permutation_importance.csv"
)

importance_df.to_csv(
    csv_file,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print()
print("=" * 70)
print("TOP FEATURES BY ROC-AUC IMPORTANCE")
print("=" * 70)

print(
    importance_df[
        [
            "rank_roc_auc",
            "feature",
            "roc_auc_drop_mean",
            "roc_auc_drop_std"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print()
print("=" * 70)
print("TOP FEATURES BY BALANCED ACCURACY IMPORTANCE")
print("=" * 70)

bal_sorted = importance_df.sort_values(
    by="balanced_accuracy_drop_mean",
    ascending=False
)

print(
    bal_sorted[
        [
            "rank_balanced_accuracy",
            "feature",
            "balanced_accuracy_drop_mean",
            "balanced_accuracy_drop_std"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# PLOT 1: ROC-AUC IMPORTANCE
# ============================================================

plot_df = (
    importance_df
    .head(TOP_N)
    .sort_values(
        "roc_auc_drop_mean",
        ascending=True
    )
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df["feature"],
    plot_df["roc_auc_drop_mean"],
    xerr=plot_df["roc_auc_drop_std"]
)

plt.xlabel("Mean decrease in ROC-AUC after permutation")

plt.ylabel("Feature")

plt.title(
    "Global Feature Importance\n"
    "Frozen RBF-SVM - Permutation Importance"
)

plt.tight_layout()

roc_plot_file = (
    OUTPUT_DIR /
    "global_importance_roc_auc.png"
)

plt.savefig(
    roc_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT 2: BALANCED ACCURACY IMPORTANCE
# ============================================================

plot_df_bal = (
    importance_df
    .sort_values(
        "balanced_accuracy_drop_mean",
        ascending=False
    )
    .head(TOP_N)
    .sort_values(
        "balanced_accuracy_drop_mean",
        ascending=True
    )
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df_bal["feature"],
    plot_df_bal["balanced_accuracy_drop_mean"],
    xerr=plot_df_bal["balanced_accuracy_drop_std"]
)

plt.xlabel(
    "Mean decrease in Balanced Accuracy after permutation"
)

plt.ylabel("Feature")

plt.title(
    "Global Feature Importance at Frozen Threshold\n"
    "Frozen RBF-SVM"
)

plt.tight_layout()

bal_plot_file = (
    OUTPUT_DIR /
    "global_importance_balanced_accuracy.png"
)

plt.savefig(
    bal_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR /
    "explainability_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GLOBAL MODEL EXPLAINABILITY\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Model:\n"
        f"{MODEL_FILE}\n\n"
    )

    f.write(
        "Dataset used:\n"
        "VALIDATION ONLY\n\n"
    )

    f.write(
        f"Images analyzed: {len(X)}\n"
    )

    f.write(
        f"Features analyzed: {len(features)}\n"
    )

    f.write(
        f"Permutation repeats: {N_REPEATS}\n\n"
    )

    f.write(
        f"Frozen threshold: "
        f"{threshold:.12f}\n\n"
    )

    f.write(
        f"Baseline ROC-AUC: "
        f"{baseline_auc:.6f}\n"
    )

    f.write(
        f"Baseline balanced accuracy: "
        f"{baseline_balanced_accuracy:.6f}\n\n"
    )

    f.write(
        "TOP FEATURES BY ROC-AUC DROP\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        importance_df[
            [
                "rank_roc_auc",
                "feature",
                "roc_auc_drop_mean",
                "roc_auc_drop_std"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "TOP FEATURES BY BALANCED ACCURACY DROP\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        bal_sorted[
            [
                "rank_balanced_accuracy",
                "feature",
                "balanced_accuracy_drop_mean",
                "balanced_accuracy_drop_std"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    f.write("\n")


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 70)
print("GLOBAL EXPLAINABILITY COMPLETE")
print("=" * 70)

print()
print("Total execution time:")
print(
    f"{time.time() - overall_start:.2f} seconds"
)

print()
print("Files saved:")

print(csv_file)
print(roc_plot_file)
print(bal_plot_file)
print(summary_file)

print()
print("IMPORTANT:")
print(
    "No model parameters were changed."
)

print(
    "No feature was removed."
)

print(
    "The frozen threshold was not changed."
)

print(
    "The TEST set was not used."
)

print()
print("NEXT STEP:")
print(
    "Local explainability for individual TP / TN / FP / FN cases."
)