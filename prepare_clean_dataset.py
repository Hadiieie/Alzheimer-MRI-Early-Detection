from pathlib import Path
import pandas as pd


# ==========================================
# Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

METADATA_FILE = RESULTS_DIR / "metadata.csv"


# ==========================================
# Read metadata
# ==========================================

df = pd.read_csv(METADATA_FILE)


print("\n======================================")
print("CLEAN DATASET PREPARATION")
print("======================================\n")


print("Original number of images:")
print(len(df))


# ==========================================
# Find duplicate groups
# ==========================================

duplicate_rows = df[
    df.duplicated(
        subset=["hash"],
        keep=False
    )
]

duplicate_groups = duplicate_rows["hash"].nunique()


print("\nDuplicate rows involved:")
print(len(duplicate_rows))


print("\nUnique duplicate groups:")
print(duplicate_groups)


# ==========================================
# Remove duplicate COPIES
# Keep first copy only
# ==========================================

clean_df = df.drop_duplicates(
    subset=["hash"],
    keep="first"
).copy()


removed_count = len(df) - len(clean_df)


print("\nRedundant duplicate copies removed:")
print(removed_count)


print("\nClean dataset size:")
print(len(clean_df))


# ==========================================
# Verify duplicates are gone
# ==========================================

remaining_duplicates = clean_df[
    clean_df.duplicated(
        subset=["hash"],
        keep=False
    )
]


print("\nRemaining exact duplicate rows:")
print(len(remaining_duplicates))


# ==========================================
# Show clean class distribution
# ==========================================

print("\nCLEAN CLASS DISTRIBUTION")
print("------------------------")

print(
    clean_df["class"].value_counts()
)


print("\nCLEAN CLASS PERCENTAGES")
print("-----------------------")

print(
    clean_df["class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ==========================================
# Save clean metadata
# ==========================================

output_file = RESULTS_DIR / "clean_metadata.csv"

clean_df.to_csv(
    output_file,
    index=False
)


print("\n======================================")
print("CLEAN DATASET MANIFEST CREATED")
print("======================================")

print("\nSaved to:")

print(output_file)