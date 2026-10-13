from pathlib import Path
import pandas as pd


# ==========================================
# 1. Define project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

METADATA_FILE = RESULTS_DIR / "metadata.csv"


# ==========================================
# 2. Read metadata
# ==========================================

df = pd.read_csv(METADATA_FILE)

print("\n==========================================")
print("FINAL DATASET AUDIT")
print("==========================================\n")


# ==========================================
# 3. Total number of images
# ==========================================

print("TOTAL NUMBER OF IMAGES")
print("----------------------")

print(len(df))


# ==========================================
# 4. Class distribution
# ==========================================

print("\nCLASS DISTRIBUTION")
print("------------------")

class_counts = df["class"].value_counts()

print(class_counts)


print("\nCLASS PERCENTAGES")
print("-----------------")

class_percentages = (
    df["class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(class_percentages)


# ==========================================
# 5. Image status
# ==========================================

print("\nIMAGE STATUS")
print("------------")

print(df["status"].value_counts())


# ==========================================
# 6. Image modes
# ==========================================

print("\nIMAGE MODES")
print("-----------")

print(df["mode"].value_counts())


# ==========================================
# 7. Image dimensions
# ==========================================

print("\nIMAGE DIMENSIONS")
print("----------------")

dimensions = (
    df.groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
)

print(dimensions)


# ==========================================
# 8. Exact duplicate detection
# ==========================================

duplicate_mask = df.duplicated(
    subset=["hash"],
    keep=False
)

duplicates = df[duplicate_mask].copy()

print("\nEXACT DUPLICATE ROWS")
print("--------------------")

print(len(duplicates))


unique_duplicate_hashes = duplicates["hash"].nunique()

print("\nUNIQUE DUPLICATE GROUPS")
print("-----------------------")

print(unique_duplicate_hashes)


# ==========================================
# 9. Cross-class duplicates
# ==========================================

cross_class_hashes = (
    df.groupby("hash")["class"]
    .nunique()
)

cross_class_hashes = cross_class_hashes[
    cross_class_hashes > 1
].index


cross_class_duplicates = df[
    df["hash"].isin(cross_class_hashes)
].copy()


print("\nCROSS-CLASS DUPLICATE IMAGES")
print("----------------------------")

print(len(cross_class_duplicates))


# ==========================================
# 10. Save reports
# ==========================================

duplicates.to_csv(
    RESULTS_DIR / "all_duplicates.csv",
    index=False
)


cross_class_duplicates.to_csv(
    RESULTS_DIR / "cross_class_duplicates.csv",
    index=False
)


class_counts.to_csv(
    RESULTS_DIR / "class_distribution.csv"
)


dimensions.to_csv(
    RESULTS_DIR / "image_dimensions.csv"
)


# ==========================================
# 11. Final message
# ==========================================

print("\n==========================================")
print("AUDIT COMPLETE")
print("==========================================")

print("\nReports saved inside the results folder.")