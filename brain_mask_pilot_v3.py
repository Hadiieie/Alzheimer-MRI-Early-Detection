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

PROCESSED_METADATA = (
    RESULTS_DIR /
    "processed_metadata.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR /
    "BRAIN_MASK_PILOT_V3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD METADATA
# =========================================================

df = pd.read_csv(
    PROCESSED_METADATA
)


# =========================================================
# TEST THESE EXACT SUSPICIOUS IMAGES
# PLUS OUR 4 REFERENCE IMAGES
# =========================================================

TEST_FILENAMES = [

    # suspicious Moderate images
    "MRI-MoD-09300.jpg",
    "MRI-MoD-06490.jpg",
    "MRI-MoD-07742.jpg",

    # reference images we already inspected
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
print("BRAIN MASK PILOT V3")
print("==========================================")

print("\nRequested images:")
print(len(TEST_FILENAMES))

print("\nFound images:")
print(len(test_df))

print("\nImages:")
print(
    test_df[
        [
            "filename",
            "original_class"
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# UTILITY: ZERO IMAGE BORDER
# =========================================================

def zero_border(binary, border=5):

    output = binary.copy()

    output[:border, :] = 0
    output[-border:, :] = 0
    output[:, :border] = 0
    output[:, -border:] = 0

    return output


# =========================================================
# ROBUST BRAIN MASK V3
# =========================================================

def create_brain_mask_v3(image):

    h, w = image.shape

    BORDER = 5


    # =====================================================
    # 1. INITIAL THRESHOLD
    # =====================================================

    binary = np.zeros_like(
        image,
        dtype=np.uint8
    )

    binary[
        image > 8
    ] = 255


    # Initial border suppression
    binary = zero_border(
        binary,
        BORDER
    )


    # =====================================================
    # 2. REMOVE SMALL NOISE
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
    # 3. CONNECT SMALL GAPS
    # Smaller than before to avoid over-expansion
    # =====================================================

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )


    # =====================================================
    # CRITICAL V3 FIX:
    # ZERO BORDER AGAIN AFTER MORPHOLOGY
    # =====================================================

    cleaned = zero_border(
        cleaned,
        BORDER
    )


    # =====================================================
    # 4. CONNECTED COMPONENTS
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
    # 5. EVALUATE COMPONENTS
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


        # ---------------------------------------------
        # Reject tiny components
        # ---------------------------------------------

        if area < 1000:
            continue


        # ---------------------------------------------
        # Reject components too close to edge
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
        # Component size sanity check
        # ---------------------------------------------

        area_ratio = (
            area /
            float(h * w)
        )


        # For this dataset a component occupying almost
        # the whole image is suspicious.
        if area_ratio > 0.85:
            continue


        # ---------------------------------------------
        # Prefer large components near center
        # ---------------------------------------------

        centroid = np.array(
            centroids[label]
        )


        distance = np.linalg.norm(
            centroid -
            image_center
        )


        normalized_distance = (
            distance /
            np.linalg.norm(
                image_center
            )
        )


        score = (

            area_ratio
            -
            0.25
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
    # 6. CREATE COMPONENT MASK
    # =====================================================

    component_mask = np.zeros_like(
        image,
        dtype=np.uint8
    )


    component_mask[
        labels == best_label
    ] = 255


    # =====================================================
    # 7. EXTERNAL CONTOUR
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
            "NO_CONTOUR"
        )


    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    # =====================================================
    # 8. FILL BRAIN SILHOUETTE
    # =====================================================

    final_mask = np.zeros_like(
        image,
        dtype=np.uint8
    )


    cv2.drawContours(

        final_mask,

        [largest_contour],

        -1,

        255,

        thickness=cv2.FILLED

    )


    # =====================================================
    # 9. SMALL FINAL SMOOTHING
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


    # CRITICAL:
    # enforce border one final time
    final_mask = zero_border(
        final_mask,
        BORDER
    )


    # =====================================================
    # 10. FINAL QC
    # =====================================================

    mask_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    pixel_ratio = (
        mask_pixels /
        float(
            final_mask.size
        )
    )


    # Reject clearly implausible result
    if (
        pixel_ratio < 0.20
        or
        pixel_ratio > 0.85
    ):

        return (
            None,
            binary,
            cleaned,
            "FAILED_RATIO_QC"
        )


    return (
        final_mask,
        binary,
        cleaned,
        "PASS"
    )


# =========================================================
# PROCESS TEST IMAGES
# =========================================================

records = []


for _, row in test_df.iterrows():

    path = Path(
        row["processed_path"]
    )


    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )


    if image is None:

        print(
            "\nFAILED TO READ:",
            row["filename"]
        )

        continue


    (
        final_mask,
        initial_binary,
        cleaned_binary,
        status

    ) = create_brain_mask_v3(
        image
    )


    # =====================================================
    # HANDLE FAILED MASK
    # =====================================================

    if final_mask is None:

        print(
            "\nMASK FAILED QC:",
            row["filename"],
            "| status:",
            status
        )


        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )


        axes[0].imshow(
            image,
            cmap="gray"
        )

        axes[0].set_title(
            "Processed MRI"
        )

        axes[0].axis(
            "off"
        )


        axes[1].imshow(
            initial_binary,
            cmap="gray"
        )

        axes[1].set_title(
            "Initial Binary"
        )

        axes[1].axis(
            "off"
        )


        axes[2].imshow(
            cleaned_binary,
            cmap="gray"
        )

        axes[2].set_title(
            f"Cleaned\n{status}"
        )

        axes[2].axis(
            "off"
        )


        fig.suptitle(
            f"{row['original_class']} | "
            f"{row['filename']}"
        )


        plt.tight_layout()


        plt.savefig(

            OUTPUT_DIR /
            (
                f"{row['filename']}_"
                f"FAILED_V3.png"
            ),

            dpi=200

        )


        plt.close()


        records.append({

            "filename":
                row["filename"],

            "original_class":
                row["original_class"],

            "mask_status":
                status,

            "mask_pixels":
                np.nan,

            "brain_ratio":
                np.nan

        })


        continue


    # =====================================================
    # MASKED BRAIN
    # =====================================================

    masked = cv2.bitwise_and(
        image,
        image,
        mask=final_mask
    )


    mask_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    brain_ratio = (
        mask_pixels /
        float(
            final_mask.size
        )
    )


    # =====================================================
    # VISUALIZATION
    # =====================================================

    fig, axes = plt.subplots(

        1,

        5,

        figsize=(18, 4)

    )


    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        "Processed MRI"
    )

    axes[0].axis(
        "off"
    )


    axes[1].imshow(
        initial_binary,
        cmap="gray"
    )

    axes[1].set_title(
        "Initial Foreground"
    )

    axes[1].axis(
        "off"
    )


    axes[2].imshow(
        cleaned_binary,
        cmap="gray"
    )

    axes[2].set_title(
        "Cleaned Foreground"
    )

    axes[2].axis(
        "off"
    )


    axes[3].imshow(
        final_mask,
        cmap="gray"
    )

    axes[3].set_title(
        "Final Brain Mask V3"
    )

    axes[3].axis(
        "off"
    )


    axes[4].imshow(
        masked,
        cmap="gray"
    )

    axes[4].set_title(
        f"Masked Brain\nRatio={brain_ratio:.4f}"
    )

    axes[4].axis(
        "off"
    )


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
            f"MASK_V3.png"
        ),

        dpi=200

    )


    plt.close()


    records.append({

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "mask_status":
            status,

        "mask_pixels":
            mask_pixels,

        "brain_ratio":
            brain_ratio

    })


    print(

        "\nPASS:",

        row["filename"],

        "| class:",

        row["original_class"],

        "| brain ratio:",

        round(
            brain_ratio,
            4
        )

    )


# =========================================================
# SAVE REPORT
# =========================================================

report_df = pd.DataFrame(
    records
)


REPORT_FILE = (
    OUTPUT_DIR /
    "brain_mask_v3_report.csv"
)


report_df.to_csv(
    REPORT_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n==========================================")
print("BRAIN MASK V3 TEST COMPLETE")
print("==========================================")


print("\nStatus:")

print(
    report_df[
        "mask_status"
    ].value_counts(
        dropna=False
    )
)


print("\nResults:")

print(
    report_df[
        [
            "filename",
            "original_class",
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