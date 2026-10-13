from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

METADATA_FILE = (
    RESULTS_DIR /
    "processed_metadata.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR /
    "BRAIN_MASK_PILOT_V4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD METADATA
# =========================================================

df = pd.read_csv(
    METADATA_FILE
)


# =========================================================
# EXACT TEST IMAGES
# =========================================================

TEST_FILENAMES = [

    # Suspicious / failed Moderate cases
    "MRI-MoD-09300.jpg",
    "MRI-MoD-06490.jpg",
    "MRI-MoD-07742.jpg",

    # Previously validated references
    "MRI-ND-01781.jpg",
    "MRI-VMD-01255.jpg",
    "MRI-MD-00736.jpg",
    "MRI-MoD-03525.jpg"
]


test_df = df[
    df["filename"].isin(
        TEST_FILENAMES
    )
].copy()


print("\n==========================================")
print("BRAIN MASK PILOT V4")
print("==========================================")

print("\nRequested images:")
print(len(TEST_FILENAMES))

print("\nFound images:")
print(len(test_df))


# =========================================================
# RESIZE ORIGINAL IMAGE WITH ASPECT RATIO
# SAME METHOD USED DURING PREPROCESSING
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
        y_offset:
        y_offset + new_h,

        x_offset:
        x_offset + new_w
    ] = resized


    return canvas


# =========================================================
# ZERO BORDER UTILITY
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
# BRAIN MASK V4
# =========================================================

def create_brain_mask_v4(
    standardized_raw
):

    h, w = standardized_raw.shape

    BORDER = 5


    # =====================================================
    # 1. LIGHT BLUR FOR STABLE THRESHOLDING
    # =====================================================

    blurred = cv2.GaussianBlur(
        standardized_raw,
        (5, 5),
        0
    )


    # =====================================================
    # 2. OTSU THRESHOLD
    #
    # No fixed value such as >8.
    # Threshold adapts to each image.
    # =====================================================

    otsu_value, binary = cv2.threshold(

        blurred,

        0,

        255,

        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU

    )


    # =====================================================
    # 3. FORCE BORDER TO BACKGROUND
    # =====================================================

    binary = zero_border(
        binary,
        BORDER
    )


    # =====================================================
    # 4. REMOVE SMALL NOISE
    # =====================================================

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


    # =====================================================
    # 5. CONNECT SMALL BREAKS
    # =====================================================

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


    # =====================================================
    # 6. CONNECTED COMPONENT ANALYSIS
    # =====================================================

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
            binary,
            cleaned,
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


    # =====================================================
    # 7. FIND PLAUSIBLE BRAIN COMPONENT
    # =====================================================

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


        # ---------------------------------------------
        # Size sanity
        # ---------------------------------------------

        if area < 1000:
            continue


        if (
            area_ratio < 0.15
            or
            area_ratio > 0.85
        ):
            continue


        # ---------------------------------------------
        # Must not touch image border
        # ---------------------------------------------

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


        # ---------------------------------------------
        # Position sanity
        # ---------------------------------------------

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


        # Large + central is preferred
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

            "distance":
                distance,

            "score":
                score

        })


    # =====================================================
    # NO UNSAFE FALLBACK
    # =====================================================

    if len(candidates) == 0:

        return (
            None,
            binary,
            cleaned,
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


    # =====================================================
    # 8. COMPONENT MASK
    # =====================================================

    component_mask = np.zeros_like(
        standardized_raw,
        dtype=np.uint8
    )


    component_mask[
        labels == best_label
    ] = 255


    # =====================================================
    # 9. EXTERNAL CONTOUR
    # =====================================================

    contours, _ = cv2.findContours(

        component_mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    if len(contours) == 0:

        return (
            None,
            binary,
            cleaned,
            otsu_value,
            "NO_CONTOUR"
        )


    contour = max(
        contours,
        key=cv2.contourArea
    )


    # =====================================================
    # 10. FILL WHOLE BRAIN SILHOUETTE
    # =====================================================

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


    # =====================================================
    # 11. SMALL FINAL SMOOTH
    # =====================================================

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


    # =====================================================
    # 12. FINAL QC
    # =====================================================

    mask_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    brain_ratio = (
        mask_pixels
        /
        float(
            final_mask.size
        )
    )


    if (
        brain_ratio < 0.20
        or
        brain_ratio > 0.85
    ):

        return (
            None,
            binary,
            cleaned,
            otsu_value,
            "FAILED_RATIO_QC"
        )


    return (
        final_mask,
        binary,
        cleaned,
        otsu_value,
        "PASS"
    )


# =========================================================
# PROCESS TEST IMAGES
# =========================================================

records = []


for _, row in test_df.iterrows():

    # =====================================================
    # ORIGINAL RAW IMAGE
    # =====================================================

    raw_path = Path(
        row["path"]
    )


    raw = cv2.imread(
        str(raw_path),
        cv2.IMREAD_GRAYSCALE
    )


    if raw is None:

        print(
            "\nFAILED TO READ RAW:",
            row["filename"]
        )

        continue


    standardized_raw = resize_with_padding(
        raw,
        220
    )


    # =====================================================
    # PROCESSED IMAGE
    # Used only for final visualization/features
    # =====================================================

    processed_path = Path(
        row["processed_path"]
    )


    processed = cv2.imread(
        str(processed_path),
        cv2.IMREAD_GRAYSCALE
    )


    if processed is None:

        print(
            "\nFAILED TO READ PROCESSED:",
            row["filename"]
        )

        continue


    # =====================================================
    # CREATE MASK FROM RAW/STANDARDIZED IMAGE
    # =====================================================

    (
        final_mask,
        binary,
        cleaned,
        otsu_value,
        status

    ) = create_brain_mask_v4(
        standardized_raw
    )


    # =====================================================
    # FAILED MASK
    # =====================================================

    if final_mask is None:

        records.append({

            "filename":
                row["filename"],

            "original_class":
                row["original_class"],

            "otsu_threshold":
                otsu_value,

            "mask_status":
                status,

            "brain_ratio":
                np.nan

        })


        fig, axes = plt.subplots(
            1,
            4,
            figsize=(15, 4)
        )


        axes[0].imshow(
            standardized_raw,
            cmap="gray"
        )

        axes[0].set_title(
            "Raw Standardized MRI"
        )

        axes[0].axis("off")


        axes[1].imshow(
            processed,
            cmap="gray"
        )

        axes[1].set_title(
            "Processed MRI"
        )

        axes[1].axis("off")


        axes[2].imshow(
            binary,
            cmap="gray"
        )

        axes[2].set_title(
            f"Otsu Binary\nT={otsu_value:.1f}"
        )

        axes[2].axis("off")


        axes[3].imshow(
            cleaned,
            cmap="gray"
        )

        axes[3].set_title(
            status
        )

        axes[3].axis("off")


        fig.suptitle(
            f"{row['original_class']} | "
            f"{row['filename']}"
        )


        plt.tight_layout()


        plt.savefig(

            OUTPUT_DIR /
            (
                f"{row['filename']}_"
                f"FAILED_V4.png"
            ),

            dpi=200

        )


        plt.close()


        print(

            "\nFAILED:",

            row["filename"],

            "|",

            status,

            "| Otsu:",

            round(
                otsu_value,
                2
            )

        )


        continue


    # =====================================================
    # APPLY RAW-DERIVED MASK TO PROCESSED IMAGE
    # =====================================================

    masked_processed = cv2.bitwise_and(

        processed,

        processed,

        mask=final_mask

    )


    mask_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    brain_ratio = (
        mask_pixels
        /
        float(
            final_mask.size
        )
    )


    # =====================================================
    # SAVE REPORT ROW
    # =====================================================

    records.append({

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "otsu_threshold":
            otsu_value,

        "mask_status":
            "PASS",

        "brain_ratio":
            brain_ratio

    })


    # =====================================================
    # VISUALIZATION
    # =====================================================

    fig, axes = plt.subplots(

        1,

        6,

        figsize=(22, 4)

    )


    axes[0].imshow(
        standardized_raw,
        cmap="gray"
    )

    axes[0].set_title(
        "Raw Standardized"
    )

    axes[0].axis("off")


    axes[1].imshow(
        processed,
        cmap="gray"
    )

    axes[1].set_title(
        "Processed MRI"
    )

    axes[1].axis("off")


    axes[2].imshow(
        binary,
        cmap="gray"
    )

    axes[2].set_title(
        f"Otsu Binary\nT={otsu_value:.1f}"
    )

    axes[2].axis("off")


    axes[3].imshow(
        cleaned,
        cmap="gray"
    )

    axes[3].set_title(
        "Cleaned Foreground"
    )

    axes[3].axis("off")


    axes[4].imshow(
        final_mask,
        cmap="gray"
    )

    axes[4].set_title(
        "Final Brain Mask V4"
    )

    axes[4].axis("off")


    axes[5].imshow(
        masked_processed,
        cmap="gray"
    )

    axes[5].set_title(
        f"Masked Processed MRI\n"
        f"Ratio={brain_ratio:.4f}"
    )

    axes[5].axis("off")


    fig.suptitle(

        f"{row['original_class']} | "
        f"{row['filename']}",

        fontsize=13

    )


    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR /
        (
            f"{row['filename']}_"
            f"MASK_V4.png"
        ),

        dpi=200

    )


    plt.close()


    print(

        "\nPASS:",

        row["filename"],

        "| class:",

        row["original_class"],

        "| Otsu:",

        round(
            otsu_value,
            2
        ),

        "| ratio:",

        round(
            brain_ratio,
            4
        )

    )


# =========================================================
# REPORT
# =========================================================

report_df = pd.DataFrame(
    records
)


REPORT_FILE = (
    OUTPUT_DIR /
    "brain_mask_v4_report.csv"
)


report_df.to_csv(
    REPORT_FILE,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n==========================================")
print("BRAIN MASK V4 TEST COMPLETE")
print("==========================================")


print("\nStatus:")

print(
    report_df[
        "mask_status"
    ].value_counts(
        dropna=False
    )
)


print("\nResults:\n")


print(
    report_df[
        [
            "filename",
            "original_class",
            "otsu_threshold",
            "mask_status",
            "brain_ratio"
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved inside:")
print(
    OUTPUT_DIR
)