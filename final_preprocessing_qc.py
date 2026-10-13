from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

PILOT_FILE = (
    RESULTS_DIR
    / "PREPROCESSING_PILOT"
    / "pilot_images.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "final_preprocessing_qc.csv"
)


# =========================================================
# LOAD PILOT IMAGE LIST
# =========================================================

pilot_df = pd.read_csv(PILOT_FILE)


# =========================================================
# ASPECT-RATIO PRESERVING RESIZE + BLACK PADDING
# =========================================================

def resize_with_padding(image, target_size=220):

    h, w = image.shape

    scale = min(
        target_size / w,
        target_size / h
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (target_size, target_size),
        dtype=np.uint8
    )

    x_offset = (
        target_size - new_w
    ) // 2

    y_offset = (
        target_size - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# =========================================================
# FINAL PREPROCESSING PIPELINE
# =========================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


def final_preprocess(image_path):

    # 1. Load as grayscale
    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # 2. Aspect-preserving resize + padding
    standardized = resize_with_padding(
        gray,
        target_size=220
    )

    # 3. CLAHE
    enhanced = clahe.apply(
        standardized
    )

    # 4. Median 3x3
    filtered = cv2.medianBlur(
        enhanced,
        3
    )

    # 5. Normalize to [0,1]
    normalized = (
        filtered.astype(np.float32)
        / 255.0
    )

    return normalized


# =========================================================
# QC
# =========================================================

records = []


for _, row in pilot_df.iterrows():

    image_path = Path(
        row["path"]
    )

    try:

        processed = final_preprocess(
            image_path
        )

        shape_ok = (
            processed.shape
            == (220, 220)
        )

        dtype_ok = (
            processed.dtype
            == np.float32
        )

        min_value = float(
            processed.min()
        )

        max_value = float(
            processed.max()
        )

        range_ok = (
            min_value >= 0.0
            and
            max_value <= 1.0
        )

        nan_count = int(
            np.isnan(processed).sum()
        )

        nan_ok = (
            nan_count == 0
        )

        passed = (
            shape_ok
            and dtype_ok
            and range_ok
            and nan_ok
        )

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

    except Exception as e:

        shape_ok = False
        dtype_ok = False
        range_ok = False
        nan_ok = False

        min_value = None
        max_value = None
        nan_count = None

        status = "ERROR"


    records.append({

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "shape_ok":
            shape_ok,

        "dtype_ok":
            dtype_ok,

        "min_value":
            min_value,

        "max_value":
            max_value,

        "range_ok":
            range_ok,

        "nan_count":
            nan_count,

        "nan_ok":
            nan_ok,

        "status":
            status
    })


# =========================================================
# SAVE QC REPORT
# =========================================================

qc_df = pd.DataFrame(
    records
)

qc_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n==========================================")
print("FINAL PREPROCESSING QC")
print("==========================================")

print("\nImages checked:")
print(len(qc_df))

print("\nStatus:")
print(
    qc_df["status"]
    .value_counts()
)

print("\nShape check:")
print(
    qc_df["shape_ok"]
    .value_counts()
)

print("\nData type check:")
print(
    qc_df["dtype_ok"]
    .value_counts()
)

print("\nRange check:")
print(
    qc_df["range_ok"]
    .value_counts()
)

print("\nNaN check:")
print(
    qc_df["nan_ok"]
    .value_counts()
)

print("\nOverall minimum:")
print(
    qc_df["min_value"]
    .min()
)

print("\nOverall maximum:")
print(
    qc_df["max_value"]
    .max()
)

print("\nQC report saved to:")
print(
    OUTPUT_FILE
)