from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

METADATA_FILE = (
    RESULTS_DIR /
    "processed_metadata.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR /
    "brain_mask_v4_full_audit.csv"
)

FAILURE_FILE = (
    RESULTS_DIR /
    "brain_mask_v4_failures.csv"
)

REVIEW_FILE = (
    RESULTS_DIR /
    "brain_mask_v4_review_outliers.csv"
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    METADATA_FILE
)


print("\n==========================================")
print("FULL BRAIN MASK V4 AUDIT")
print("==========================================")

print("\nTotal images:")
print(len(df))


# =========================================================
# RESIZE RAW IMAGE WITH ASPECT-RATIO PRESERVATION
# =========================================================

def resize_with_padding(
    image,
    target_size=220
):

    h, w = image.shape

    scale = min(
        target_size / w,
        target_size / h
    )

    new_w = int(
        round(w * scale)
    )

    new_h = int(
        round(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (
            target_size,
            target_size
        ),
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
# BORDER UTILITY
# =========================================================

def zero_border(
    image,
    border=5
):

    output = image.copy()

    output[:border, :] = 0
    output[-border:, :] = 0
    output[:, :border] = 0
    output[:, -border:] = 0

    return output


# =========================================================
# APPROVED BRAIN MASK V4
# =========================================================

def create_brain_mask_v4(
    standardized_raw
):

    h, w = standardized_raw.shape

    BORDER = 5


    # -----------------------------------------------------
    # 1. Light smoothing
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        standardized_raw,
        (5, 5),
        0
    )


    # -----------------------------------------------------
    # 2. Adaptive Otsu threshold
    # -----------------------------------------------------

    otsu_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )


    # -----------------------------------------------------
    # 3. Border suppression
    # -----------------------------------------------------

    binary = zero_border(
        binary,
        BORDER
    )


    # -----------------------------------------------------
    # 4. Morphological opening
    # -----------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )


    # -----------------------------------------------------
    # 5. Morphological closing
    # -----------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )


    cleaned = zero_border(
        cleaned,
        BORDER
    )


    # -----------------------------------------------------
    # 6. Connected components
    # -----------------------------------------------------

    (
        num_labels,
        labels,
        stats,
        centroids

    ) = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8
    )


    if num_labels <= 1:

        return (
            None,
            otsu_value,
            "NO_COMPONENT"
        )


    image_center = np.array(
        [
            w / 2.0,
            h / 2.0
        ]
    )


    candidates = []


    # -----------------------------------------------------
    # 7. Candidate selection
    # -----------------------------------------------------

    for label in range(
        1,
        num_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        cw = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        ch = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )


        area_ratio = (
            area /
            float(h * w)
        )


        # Ignore tiny regions
        if area < 1000:
            continue


        # Hard sanity limits
        if (
            area_ratio < 0.15
            or
            area_ratio > 0.85
        ):
            continue


        # Reject components touching image borders
        touches_edge = (
            x <= BORDER
            or
            y <= BORDER
            or
            x + cw >= w - BORDER
            or
            y + ch >= h - BORDER
        )


        if touches_edge:
            continue


        centroid = np.array(
            centroids[label]
        )


        distance = np.linalg.norm(
            centroid -
            image_center
        )


        normalized_distance = (
            distance
            /
            (
                np.linalg.norm(
                    image_center
                )
                +
                1e-8
            )
        )


        score = (
            area_ratio
            -
            0.20
            *
            normalized_distance
        )


        candidates.append({

            "label":
                label,

            "area":
                area,

            "area_ratio":
                area_ratio,

            "score":
                score

        })


    # -----------------------------------------------------
    # No unsafe fallback
    # -----------------------------------------------------

    if len(candidates) == 0:

        return (
            None,
            otsu_value,
            "NO_VALID_COMPONENT"
        )


    best = max(
        candidates,
        key=lambda item:
            item["score"]
    )


    best_label = (
        best["label"]
    )


    # -----------------------------------------------------
    # 8. Selected component
    # -----------------------------------------------------

    component_mask = np.zeros_like(
        standardized_raw,
        dtype=np.uint8
    )


    component_mask[
        labels == best_label
    ] = 255


    # -----------------------------------------------------
    # 9. External contour
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) == 0:

        return (
            None,
            otsu_value,
            "NO_CONTOUR"
        )


    contour = max(
        contours,
        key=cv2.contourArea
    )


    # -----------------------------------------------------
    # 10. Filled whole-brain silhouette
    # -----------------------------------------------------

    final_mask = np.zeros_like(
        standardized_raw,
        dtype=np.uint8
    )


    cv2.drawContours(
        final_mask,
        [contour],
        -1,
        255,
        thickness=cv2.FILLED
    )


    # -----------------------------------------------------
    # 11. Final smoothing
    # -----------------------------------------------------

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        ),
        iterations=1
    )


    final_mask = zero_border(
        final_mask,
        BORDER
    )


    # -----------------------------------------------------
    # 12. Final mask ratio
    # -----------------------------------------------------

    brain_ratio = float(
        np.mean(
            final_mask > 0
        )
    )


    if (
        brain_ratio < 0.20
        or
        brain_ratio > 0.85
    ):

        return (
            None,
            otsu_value,
            "FAILED_RATIO_QC"
        )


    return (
        final_mask,
        otsu_value,
        "PASS"
    )


# =========================================================
# FULL AUDIT
# =========================================================

records = []

total = len(df)


for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    raw_path = Path(
        row["path"]
    )


    raw = cv2.imread(
        str(raw_path),
        cv2.IMREAD_GRAYSCALE
    )


    # -----------------------------------------------------
    # Read failure
    # -----------------------------------------------------

    if raw is None:

        records.append({

            "filename":
                row["filename"],

            "original_class":
                row["original_class"],

            "raw_path":
                str(raw_path),

            "otsu_threshold":
                np.nan,

            "mask_status":
                "READ_FAILURE",

            "brain_pixels":
                np.nan,

            "brain_ratio":
                np.nan,

            "bbox_width":
                np.nan,

            "bbox_height":
                np.nan,

            "bbox_aspect_ratio":
                np.nan,

            "review_flag":
                True

        })

        continue


    standardized_raw = resize_with_padding(
        raw,
        target_size=220
    )


    (
        mask,
        otsu_value,
        status

    ) = create_brain_mask_v4(
        standardized_raw
    )


    # -----------------------------------------------------
    # Mask failure
    # -----------------------------------------------------

    if mask is None:

        records.append({

            "filename":
                row["filename"],

            "original_class":
                row["original_class"],

            "raw_path":
                str(raw_path),

            "otsu_threshold":
                otsu_value,

            "mask_status":
                status,

            "brain_pixels":
                np.nan,

            "brain_ratio":
                np.nan,

            "bbox_width":
                np.nan,

            "bbox_height":
                np.nan,

            "bbox_aspect_ratio":
                np.nan,

            "review_flag":
                True

        })


    # -----------------------------------------------------
    # Successful mask
    # -----------------------------------------------------

    else:

        brain_pixels = int(
            np.sum(
                mask > 0
            )
        )


        brain_ratio = (
            brain_pixels
            /
            float(
                mask.size
            )
        )


        points = cv2.findNonZero(
            mask
        )


        if points is not None:

            x, y, w, h = cv2.boundingRect(
                points
            )

            bbox_width = int(w)
            bbox_height = int(h)

            bbox_aspect_ratio = (
                float(w)
                /
                float(h)
            )

        else:

            bbox_width = np.nan
            bbox_height = np.nan
            bbox_aspect_ratio = np.nan


        # -------------------------------------------------
        # REVIEW flag only.
        #
        # This does NOT mean automatic rejection.
        # It identifies unusually small/large masks
        # for later inspection.
        # -------------------------------------------------

        review_flag = bool(
            brain_ratio < 0.30
            or
            brain_ratio > 0.70
        )


        records.append({

            "filename":
                row["filename"],

            "original_class":
                row["original_class"],

            "raw_path":
                str(raw_path),

            "otsu_threshold":
                float(
                    otsu_value
                ),

            "mask_status":
                "PASS",

            "brain_pixels":
                brain_pixels,

            "brain_ratio":
                brain_ratio,

            "bbox_width":
                bbox_width,

            "bbox_height":
                bbox_height,

            "bbox_aspect_ratio":
                bbox_aspect_ratio,

            "review_flag":
                review_flag

        })


    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    if position % 1000 == 0:

        print(
            f"Audited "
            f"{position} / {total}"
        )


# =========================================================
# CREATE DATAFRAME
# =========================================================

audit_df = pd.DataFrame(
    records
)


# =========================================================
# SAVE COMPLETE AUDIT
# =========================================================

audit_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# FAILURES
# =========================================================

failure_df = audit_df[
    audit_df["mask_status"]
    != "PASS"
].copy()


failure_df.to_csv(
    FAILURE_FILE,
    index=False
)


# =========================================================
# REVIEW OUTLIERS
# =========================================================

review_df = audit_df[
    audit_df["review_flag"]
    == True
].copy()


review_df.to_csv(
    REVIEW_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n==========================================")
print("FULL BRAIN MASK V4 AUDIT COMPLETE")
print("==========================================")


print("\nTotal images:")
print(
    len(audit_df)
)


print("\nMask status:")

print(
    audit_df[
        "mask_status"
    ].value_counts(
        dropna=False
    )
)


print("\nStatus by class:")

print(
    pd.crosstab(
        audit_df[
            "original_class"
        ],
        audit_df[
            "mask_status"
        ]
    )
)


# =========================================================
# PASS-ONLY STATISTICS
# =========================================================

pass_df = audit_df[
    audit_df["mask_status"]
    == "PASS"
].copy()


print("\n==========================================")
print("PASS MASK STATISTICS")
print("==========================================")


if len(pass_df) > 0:

    print("\nBrain ratio summary:")

    print(
        pass_df[
            "brain_ratio"
        ].describe()
    )


    print("\nBrain ratio by class:")

    print(
        pass_df.groupby(
            "original_class"
        )[
            "brain_ratio"
        ].agg(
            [
                "count",
                "min",
                "mean",
                "median",
                "max"
            ]
        )
    )


    print("\nOtsu threshold summary:")

    print(
        pass_df[
            "otsu_threshold"
        ].describe()
    )


# =========================================================
# REVIEW COUNTS
# =========================================================

print("\n==========================================")
print("REVIEW OUTLIERS")
print("==========================================")


print("\nReview-flagged images:")
print(
    len(review_df)
)


if len(review_df) > 0:

    print("\nReview flags by class:")

    print(
        review_df[
            "original_class"
        ].value_counts()
    )


    print("\nLowest 10 brain ratios:")

    print(
        pass_df[
            [
                "filename",
                "original_class",
                "brain_ratio"
            ]
        ]
        .sort_values(
            "brain_ratio",
            ascending=True
        )
        .head(10)
        .to_string(
            index=False
        )
    )


    print("\nHighest 10 brain ratios:")

    print(
        pass_df[
            [
                "filename",
                "original_class",
                "brain_ratio"
            ]
        ]
        .sort_values(
            "brain_ratio",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )


# =========================================================
# FILE LOCATIONS
# =========================================================

print("\n==========================================")
print("FILES SAVED")
print("==========================================")


print("\nFull audit:")
print(
    OUTPUT_FILE
)


print("\nFailures:")
print(
    FAILURE_FILE
)


print("\nReview outliers:")
print(
    REVIEW_FILE
)