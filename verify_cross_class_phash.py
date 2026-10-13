from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np

from PIL import Image
from skimage.metrics import structural_similarity as ssim


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "cross_class_phash_duplicates.csv"

OUTPUT_FILE = RESULTS_DIR / "cross_class_ssim_report.csv"


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("\n==========================================")
print("CROSS-CLASS PHASH VERIFICATION")
print("==========================================")

print("\nImages involved:")
print(len(df))

print("\nPHASH groups involved:")
print(df["phash"].nunique())


# =========================================================
# IMAGE LOADING FUNCTION
# =========================================================

def load_for_comparison(path):

    with Image.open(path) as img:

        # Convert everything to grayscale
        img = img.convert("L")

        # Standardize only for similarity comparison
        img = img.resize(
            (220, 220),
            Image.Resampling.BILINEAR
        )

        array = np.array(img)

    return array


# =========================================================
# COMPARE CROSS-CLASS PAIRS
# =========================================================

records = []

group_number = 0


for phash_value, group in df.groupby("phash"):

    group_number += 1

    rows = group.to_dict("records")


    for row_a, row_b in combinations(rows, 2):

        # We only care about different original classes
        if row_a["original_class"] == row_b["original_class"]:
            continue


        try:

            img_a = load_for_comparison(row_a["path"])

            img_b = load_for_comparison(row_b["path"])


            score = ssim(
                img_a,
                img_b,
                data_range=255
            )


            records.append({

                "phash": phash_value,

                "class_a": row_a["original_class"],
                "filename_a": row_a["filename"],
                "path_a": row_a["path"],

                "class_b": row_b["original_class"],
                "filename_b": row_b["filename"],
                "path_b": row_b["path"],

                "ssim": score

            })


        except Exception as e:

            print(
                "Comparison failed:",
                row_a["filename"],
                row_b["filename"]
            )


    if group_number % 20 == 0:

        print(
            f"Processed {group_number} / "
            f"{df['phash'].nunique()} groups"
        )


# =========================================================
# CREATE REPORT
# =========================================================

report = pd.DataFrame(records)


if len(report) == 0:

    print("\nNo cross-class pairs found.")

else:

    report = report.sort_values(
        by="ssim",
        ascending=False
    )


    # -----------------------------------------------------
    # Similarity categories
    # -----------------------------------------------------

    def classify_similarity(score):

        if score >= 0.98:

            return "EXTREMELY_SIMILAR"

        elif score >= 0.95:

            return "VERY_SIMILAR"

        elif score >= 0.90:

            return "POSSIBLY_SIMILAR"

        else:

            return "LOWER_SIMILARITY"


    report["similarity_level"] = (
        report["ssim"]
        .apply(classify_similarity)
    )


    report.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\n==========================================")
    print("RESULTS")
    print("==========================================")


    print("\nTotal cross-class image pairs:")
    print(len(report))


    print("\nSimilarity categories:")
    print(
        report["similarity_level"]
        .value_counts()
    )


    print("\nHighest SSIM:")
    print(report["ssim"].max())


    print("\nLowest SSIM:")
    print(report["ssim"].min())


    print("\nAverage SSIM:")
    print(report["ssim"].mean())


    print("\nTop 20 most similar cross-class pairs:\n")

    print(
        report[
            [
                "class_a",
                "filename_a",
                "class_b",
                "filename_b",
                "ssim"
            ]
        ].head(20).to_string(index=False)
    )


    print("\nReport saved to:")

    print(OUTPUT_FILE)