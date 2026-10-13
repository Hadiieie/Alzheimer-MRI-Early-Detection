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

INPUT_FILE = RESULTS_DIR / "phash_duplicate_groups.csv"

OUTPUT_FILE = RESULTS_DIR / "same_class_ssim_report.csv"


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)


print("\n==========================================")
print("SAME-CLASS NEAR-DUPLICATE VERIFICATION")
print("==========================================\n")


print("Images involved:")
print(len(df))


print("\nPHASH groups:")
print(df["phash"].nunique())


# =========================================================
# LOAD IMAGE FOR SSIM
# =========================================================

def load_image(path):

    with Image.open(path) as img:

        img = img.convert("L")

        img = img.resize(
            (220, 220),
            Image.Resampling.BILINEAR
        )

        return np.array(img)


# =========================================================
# PAIRWISE COMPARISON
# =========================================================

records = []

group_counter = 0

total_groups = df["phash"].nunique()


for phash_value, group in df.groupby("phash"):

    group_counter += 1

    rows = group.to_dict("records")


    for row_a, row_b in combinations(rows, 2):

        # Only compare images from SAME class
        if row_a["original_class"] != row_b["original_class"]:
            continue


        try:

            img_a = load_image(row_a["path"])
            img_b = load_image(row_b["path"])


            score = ssim(
                img_a,
                img_b,
                data_range=255
            )


            records.append({

                "phash": phash_value,

                "original_class":
                    row_a["original_class"],

                "filename_a":
                    row_a["filename"],

                "path_a":
                    row_a["path"],

                "filename_b":
                    row_b["filename"],

                "path_b":
                    row_b["path"],

                "ssim":
                    score
            })


        except Exception as e:

            print(
                "Failed:",
                row_a["filename"],
                row_b["filename"]
            )


    if group_counter % 200 == 0:

        print(
            f"Processed "
            f"{group_counter} / {total_groups} groups"
        )


# =========================================================
# CREATE REPORT
# =========================================================

report = pd.DataFrame(records)


if len(report) == 0:

    print("\nNo same-class pairs found.")

else:

    report = report.sort_values(
        by="ssim",
        ascending=False
    )


    def classify(score):

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
        .apply(classify)
    )


    report.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\n==========================================")
    print("RESULTS")
    print("==========================================\n")


    print("Total same-class pairs:")
    print(len(report))


    print("\nSimilarity categories:")
    print(
        report[
            "similarity_level"
        ].value_counts()
    )


    print("\nHighest SSIM:")
    print(
        report["ssim"].max()
    )


    print("\nLowest SSIM:")
    print(
        report["ssim"].min()
    )


    print("\nAverage SSIM:")
    print(
        report["ssim"].mean()
    )


    print("\nPairs with SSIM >= 0.95:")
    print(
        len(
            report[
                report["ssim"] >= 0.95
            ]
        )
    )


    print("\nPairs with SSIM >= 0.98:")
    print(
        len(
            report[
                report["ssim"] >= 0.98
            ]
        )
    )


    print(
        "\nTop 20 most similar "
        "same-class pairs:\n"
    )


    print(
        report[
            [
                "original_class",
                "filename_a",
                "filename_b",
                "ssim"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


    print("\nReport saved to:")

    print(OUTPUT_FILE)