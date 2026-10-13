from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "FEATURE_EXTRACTION_PILOT"
    / "feature_extraction_pilot.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "FEATURE_EXTRACTION_PILOT"
    / "suspicious_feature_rows.csv"
)


# =========================================================
# LOAD FEATURES
# =========================================================

df = pd.read_csv(FEATURE_FILE)


print("\n==========================================")
print("FEATURE PILOT QC")
print("==========================================")


print("\nImages:")
print(len(df))


# =========================================================
# SORT ALL IMAGES BY BRAIN RATIO
# =========================================================

sorted_df = df[
    [
        "filename",
        "original_class",
        "brain_area",
        "brain_ratio",
        "brain_perimeter",
        "compactness",
        "bbox_aspect_ratio"
    ]
].sort_values(
    "brain_ratio",
    ascending=False
)


print("\n==========================================")
print("ALL IMAGES SORTED BY BRAIN RATIO")
print("==========================================\n")

print(
    sorted_df.to_string(
        index=False
    )
)


# =========================================================
# FLAG OBVIOUSLY SUSPICIOUS MASKS
# =========================================================

# A brain mask occupying >80% of the full 220x220 image
# is suspicious for this dataset because visible black
# background should remain around the brain.

high_ratio = df[
    df["brain_ratio"] > 0.80
].copy()


# Very small masks are also worth reviewing.
low_ratio = df[
    df["brain_ratio"] < 0.30
].copy()


suspicious = pd.concat(
    [
        high_ratio,
        low_ratio
    ],
    ignore_index=True
).drop_duplicates(
    subset=["filename"]
)


print("\n==========================================")
print("SUSPICIOUS MASKS")
print("==========================================")

print("\nBrain ratio > 0.80:")
print(len(high_ratio))


if len(high_ratio) > 0:

    print(
        high_ratio[
            [
                "filename",
                "original_class",
                "brain_area",
                "brain_ratio"
            ]
        ].sort_values(
            "brain_ratio",
            ascending=False
        ).to_string(
            index=False
        )
    )


print("\nBrain ratio < 0.30:")
print(len(low_ratio))


if len(low_ratio) > 0:

    print(
        low_ratio[
            [
                "filename",
                "original_class",
                "brain_area",
                "brain_ratio"
            ]
        ].sort_values(
            "brain_ratio"
        ).to_string(
            index=False
        )
    )


# =========================================================
# SAVE
# =========================================================

suspicious.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("QC COMPLETE")
print("==========================================")

print("\nSuspicious images:")
print(len(suspicious))

print("\nSaved to:")
print(OUTPUT_FILE)