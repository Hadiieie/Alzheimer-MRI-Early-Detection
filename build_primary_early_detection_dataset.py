from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "FULL_FEATURE_EXTRACTION"
    / "full_features_39538.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "PRIMARY_EARLY_DETECTION"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "primary_early_detection_dataset.csv"
)

EXCLUDED_FILE = (
    OUTPUT_DIR
    / "excluded_moderate_dataset.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    FEATURE_FILE
)


print("\n==========================================")
print("BUILD PRIMARY EARLY-DETECTION DATASET")
print("==========================================")


print("\nOriginal rows:")
print(len(df))


print("\nOriginal class distribution:")
print(
    df["original_class"].value_counts()
)


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "filename",
    "original_class"
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )


# =========================================================
# IMPORTANT: LEAKAGE GROUP CHECK
# =========================================================

if "leakage_group" not in df.columns:

    raise ValueError(
        "\nLEAKAGE GROUP COLUMN IS MISSING.\n"
        "Do NOT continue to ML splitting yet.\n"
        "The leakage_group information must be restored "
        "before Train/Validation/Test splitting."
    )


if df["leakage_group"].isna().any():

    missing_groups = int(
        df["leakage_group"].isna().sum()
    )

    raise ValueError(
        f"{missing_groups} rows have missing leakage_group."
    )


print("\nLeakage-group column:")
print("FOUND ✅")


print("\nUnique leakage groups in full dataset:")
print(
    df["leakage_group"].nunique()
)


# =========================================================
# SEPARATE MODERATE CLASS
# =========================================================

moderate_df = df[
    df["original_class"]
    == "ModerateDemented"
].copy()


moderate_df.to_csv(
    EXCLUDED_FILE,
    index=False
)


# =========================================================
# CREATE PRIMARY DATASET
# =========================================================

primary_classes = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented"
]


primary_df = df[
    df["original_class"]
    .isin(primary_classes)
].copy()


# =========================================================
# CREATE BINARY TARGET
# =========================================================

primary_df["target_name"] = np.where(
    primary_df["original_class"]
    == "NonDemented",
    "Normal",
    "Early_Demented"
)


primary_df["target"] = np.where(
    primary_df["original_class"]
    == "NonDemented",
    0,
    1
)


# =========================================================
# QC
# =========================================================

print("\n==========================================")
print("PRIMARY DATASET SUMMARY")
print("==========================================")


print("\nTotal primary images:")
print(
    len(primary_df)
)


print("\nOriginal classes:")
print(
    primary_df[
        "original_class"
    ].value_counts()
)


print("\nBinary classes:")
print(
    primary_df[
        "target_name"
    ].value_counts()
)


print("\nBinary percentages:")
print(
    (
        primary_df[
            "target_name"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(3)
)


print("\nUnique leakage groups:")
print(
    primary_df[
        "leakage_group"
    ].nunique()
)


# =========================================================
# CHECK GROUP PURITY
# =========================================================

group_class_counts = (
    primary_df
    .groupby(
        "leakage_group"
    )[
        "target"
    ]
    .nunique()
)


mixed_target_groups = (
    group_class_counts[
        group_class_counts > 1
    ]
)


print("\nGroups containing BOTH binary targets:")
print(
    len(
        mixed_target_groups
    )
)


# =========================================================
# CHECK DUPLICATES
# =========================================================

print("\nDuplicate filenames:")
print(
    int(
        primary_df[
            "filename"
        ].duplicated().sum()
    )
)


# =========================================================
# SAVE
# =========================================================

primary_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("PRIMARY DATASET CREATED")
print("==========================================")


print("\nSaved to:")
print(
    OUTPUT_FILE
)


print("\nModerate class preserved separately:")
print(
    EXCLUDED_FILE
)