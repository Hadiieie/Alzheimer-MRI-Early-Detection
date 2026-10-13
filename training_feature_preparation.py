from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "GROUP_AWARE_SPLIT"
)

TRAIN_FILE = SPLIT_DIR / "train.csv"
VAL_FILE   = SPLIT_DIR / "validation.csv"
TEST_FILE  = SPLIT_DIR / "test.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "TRAINING_PREPARATION"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD SPLITS
# =========================================================

train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)
test_df  = pd.read_csv(TEST_FILE)


print("\n==========================================")
print("TRAINING-ONLY FEATURE PREPARATION")
print("==========================================")


print("\nTrain images:")
print(len(train_df))

print("\nValidation images:")
print(len(val_df))

print("\nTest images:")
print(len(test_df))


# =========================================================
# DEFINE NON-FEATURE COLUMNS
# =========================================================

NON_FEATURE_COLUMNS = [
    "filename",
    "original_class",
    "path",
    "processed_path",
    "otsu_threshold",
    "leakage_group",
    "study_group",
    "target",
    "target_name",
    "split"
]


# =========================================================
# IDENTIFY NUMERICAL FEATURES
# FROM TRAIN ONLY
# =========================================================

feature_columns = []


for col in train_df.columns:

    if col in NON_FEATURE_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(
        train_df[col]
    ):
        feature_columns.append(col)


print("\n==========================================")
print("FULL FEATURE SET")
print("==========================================")


print("\nNumber of features:")
print(len(feature_columns))


for i, feature in enumerate(
    feature_columns,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )


if len(feature_columns) != 30:

    print(
        "\nWARNING:"
        f" Expected 30 features,"
        f" found {len(feature_columns)}."
    )


# =========================================================
# QC
# =========================================================

for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    missing_features = [
        f
        for f in feature_columns
        if f not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name}: Missing features:"
            f" {missing_features}"
        )


    X = data[feature_columns]


    nan_count = int(
        X.isna().sum().sum()
    )

    inf_count = int(
        np.isinf(
            X.to_numpy(
                dtype=float
            )
        ).sum()
    )


    print(
        f"\n{name}:"
        f" NaN={nan_count},"
        f" Inf={inf_count}"
    )


    if (
        nan_count > 0
        or
        inf_count > 0
    ):
        raise ValueError(
            f"{name} contains invalid values."
        )


# =========================================================
# TRAIN-ONLY CORRELATION
# =========================================================

train_corr = (
    train_df[
        feature_columns
    ]
    .corr()
)


train_corr.to_csv(
    OUTPUT_DIR /
    "train_feature_correlation_matrix.csv"
)


CORRELATION_THRESHOLD = 0.95


# =========================================================
# GREEDY CORRELATION REDUCTION
# TRAIN ONLY
# =========================================================

abs_corr = train_corr.abs()


features_to_keep = []
features_to_drop = []
drop_reason = []


for feature in feature_columns:

    should_drop = False


    for kept_feature in features_to_keep:

        correlation = abs_corr.loc[
            feature,
            kept_feature
        ]


        if correlation >= CORRELATION_THRESHOLD:

            should_drop = True

            features_to_drop.append(
                feature
            )

            drop_reason.append({
                "dropped_feature":
                    feature,

                "correlated_with":
                    kept_feature,

                "correlation":
                    train_corr.loc[
                        feature,
                        kept_feature
                    ]
            })

            break


    if not should_drop:
        features_to_keep.append(
            feature
        )


correlation_drop_df = pd.DataFrame(
    drop_reason
)


correlation_drop_df.to_csv(
    OUTPUT_DIR /
    "correlation_reduction_train_only.csv",
    index=False
)


print("\n==========================================")
print("CORRELATION REDUCTION — TRAIN ONLY")
print("==========================================")


print("\nThreshold:")
print(CORRELATION_THRESHOLD)


print("\nFeatures before:")
print(len(feature_columns))


print("\nFeatures removed:")
print(len(features_to_drop))


for feature in features_to_drop:
    print(
        f"DROP: {feature}"
    )


print("\nFeatures remaining:")
print(len(features_to_keep))


for feature in features_to_keep:
    print(
        f"KEEP: {feature}"
    )


# =========================================================
# SAVE FEATURE LISTS
# =========================================================

with open(
    OUTPUT_DIR /
    "full_feature_set.txt",
    "w",
    encoding="utf-8"
) as f:

    for feature in feature_columns:
        f.write(
            feature + "\n"
        )


with open(
    OUTPUT_DIR /
    "reduced_feature_set.txt",
    "w",
    encoding="utf-8"
) as f:

    for feature in features_to_keep:
        f.write(
            feature + "\n"
        )


# =========================================================
# STANDARD SCALER
# FULL FEATURE SET
# =========================================================

X_train_full = train_df[
    feature_columns
].copy()

X_val_full = val_df[
    feature_columns
].copy()

X_test_full = test_df[
    feature_columns
].copy()


full_scaler = StandardScaler()


X_train_full_scaled = (
    full_scaler.fit_transform(
        X_train_full
    )
)

X_val_full_scaled = (
    full_scaler.transform(
        X_val_full
    )
)

X_test_full_scaled = (
    full_scaler.transform(
        X_test_full
    )
)


joblib.dump(
    full_scaler,
    OUTPUT_DIR /
    "standard_scaler_full.joblib"
)


# =========================================================
# STANDARD SCALER
# REDUCED FEATURE SET
# =========================================================

X_train_reduced = train_df[
    features_to_keep
].copy()

X_val_reduced = val_df[
    features_to_keep
].copy()

X_test_reduced = test_df[
    features_to_keep
].copy()


reduced_scaler = StandardScaler()


X_train_reduced_scaled = (
    reduced_scaler.fit_transform(
        X_train_reduced
    )
)

X_val_reduced_scaled = (
    reduced_scaler.transform(
        X_val_reduced
    )
)

X_test_reduced_scaled = (
    reduced_scaler.transform(
        X_test_reduced
    )
)


joblib.dump(
    reduced_scaler,
    OUTPUT_DIR /
    "standard_scaler_reduced.joblib"
)


# =========================================================
# SAVE SCALED DATA
# =========================================================

def save_scaled_dataset(
    array,
    feature_names,
    original_df,
    filename
):

    scaled_df = pd.DataFrame(
        array,
        columns=feature_names
    )


    scaled_df.insert(
        0,
        "target",
        original_df[
            "target"
        ].to_numpy()
    )


    scaled_df.insert(
        0,
        "target_name",
        original_df[
            "target_name"
        ].to_numpy()
    )


    scaled_df.insert(
        0,
        "leakage_group",
        original_df[
            "leakage_group"
        ].to_numpy()
    )


    scaled_df.insert(
        0,
        "filename",
        original_df[
            "filename"
        ].to_numpy()
    )


    scaled_df.to_csv(
        OUTPUT_DIR / filename,
        index=False
    )


save_scaled_dataset(
    X_train_full_scaled,
    feature_columns,
    train_df,
    "train_full_scaled.csv"
)

save_scaled_dataset(
    X_val_full_scaled,
    feature_columns,
    val_df,
    "validation_full_scaled.csv"
)

save_scaled_dataset(
    X_test_full_scaled,
    feature_columns,
    test_df,
    "test_full_scaled.csv"
)


save_scaled_dataset(
    X_train_reduced_scaled,
    features_to_keep,
    train_df,
    "train_reduced_scaled.csv"
)

save_scaled_dataset(
    X_val_reduced_scaled,
    features_to_keep,
    val_df,
    "validation_reduced_scaled.csv"
)

save_scaled_dataset(
    X_test_reduced_scaled,
    features_to_keep,
    test_df,
    "test_reduced_scaled.csv"
)


# =========================================================
# SCALER QC
# =========================================================

train_scaled_mean = (
    np.abs(
        X_train_full_scaled.mean(
            axis=0
        )
    ).max()
)

train_scaled_std_error = (
    np.abs(
        X_train_full_scaled.std(
            axis=0
        )
        -
        1
    ).max()
)


print("\n==========================================")
print("SCALING QC")
print("==========================================")


print(
    "\nMaximum absolute TRAIN scaled mean:"
)

print(
    train_scaled_mean
)


print(
    "\nMaximum TRAIN std deviation from 1:"
)

print(
    train_scaled_std_error
)


# =========================================================
# TARGET COUNTS
# =========================================================

print("\n==========================================")
print("TARGET DISTRIBUTION")
print("==========================================")


print("\nTRAIN:")
print(
    train_df[
        "target_name"
    ].value_counts()
)


print("\nVALIDATION:")
print(
    val_df[
        "target_name"
    ].value_counts()
)


print("\nTEST:")
print(
    test_df[
        "target_name"
    ].value_counts()
)


# =========================================================
# FINAL SUMMARY
# =========================================================

summary = {

    "train_images":
        len(train_df),

    "validation_images":
        len(val_df),

    "test_images":
        len(test_df),

    "full_feature_count":
        len(feature_columns),

    "reduced_feature_count":
        len(features_to_keep),

    "correlation_threshold":
        CORRELATION_THRESHOLD,

    "removed_features":
        features_to_drop

}


with open(
    OUTPUT_DIR /
    "training_preparation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print("\n==========================================")
print("TRAINING PREPARATION COMPLETE")
print("==========================================")


print(
    "\nFull feature set:",
    len(feature_columns)
)


print(
    "Reduced feature set:",
    len(features_to_keep)
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)