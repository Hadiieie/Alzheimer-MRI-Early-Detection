from pathlib import Path
from PIL import Image
import pandas as pd
import imagehash


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "clean_4class_metadata.csv"


# =========================================================
# LOAD CLEAN MASTER DATASET
# =========================================================

df = pd.read_csv(INPUT_FILE)


print("\n========================================")
print("PERCEPTUAL HASH / NEAR-DUPLICATE AUDIT")
print("========================================\n")


print("Images to analyze:")
print(len(df))


# =========================================================
# CALCULATE PERCEPTUAL HASH
# =========================================================

phashes = []

failed = 0


for index, row in df.iterrows():

    image_path = Path(row["path"])

    try:

        with Image.open(image_path) as img:

            img = img.convert("L")

            phash = imagehash.phash(img)

            phashes.append(str(phash))

    except Exception:

        phashes.append(None)

        failed += 1


    if (index + 1) % 2000 == 0:

        print(
            f"Processed {index + 1} / {len(df)} images"
        )


df["phash"] = phashes


print("\nFailed perceptual hashes:")
print(failed)


# =========================================================
# EXACT PERCEPTUAL-HASH MATCHES
# =========================================================

phash_duplicates = df[
    df.duplicated(
        subset=["phash"],
        keep=False
    )
].copy()


print("\nPERCEPTUAL-HASH DUPLICATE ROWS")
print("------------------------------")

print(len(phash_duplicates))


print("\nPERCEPTUAL-HASH DUPLICATE GROUPS")
print("--------------------------------")

print(
    phash_duplicates["phash"]
    .nunique()
)


# =========================================================
# CROSS-CLASS PHASH MATCHES
# =========================================================

phash_class_counts = (
    df.groupby("phash")["original_class"]
    .nunique()
)


cross_class_phashes = phash_class_counts[
    phash_class_counts > 1
].index


cross_class_near_duplicates = df[
    df["phash"].isin(cross_class_phashes)
].copy()


print("\nCROSS-CLASS PHASH ROWS")
print("----------------------")

print(len(cross_class_near_duplicates))


print("\nCROSS-CLASS PHASH GROUPS")
print("------------------------")

print(len(cross_class_phashes))


# =========================================================
# SAVE RESULTS
# =========================================================

df.to_csv(
    RESULTS_DIR / "clean_4class_with_phash.csv",
    index=False
)


phash_duplicates.to_csv(
    RESULTS_DIR / "phash_duplicate_groups.csv",
    index=False
)


cross_class_near_duplicates.to_csv(
    RESULTS_DIR / "cross_class_phash_duplicates.csv",
    index=False
)


print("\n========================================")
print("PERCEPTUAL HASH AUDIT COMPLETE")
print("========================================")

print("\nFiles saved inside:")
print(RESULTS_DIR)