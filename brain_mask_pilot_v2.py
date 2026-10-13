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
    "BRAIN_MASK_PILOT_V2"
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


CLASSES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

SAMPLES_PER_CLASS = 5


# =========================================================
# ROBUST BRAIN MASK V2
# =========================================================

def create_brain_mask_v2(image):

    h, w = image.shape


    # -----------------------------------------------------
    # 1. Initial foreground
    # -----------------------------------------------------

    binary = np.zeros_like(
        image,
        dtype=np.uint8
    )

    binary[
        image > 8
    ] = 255


    # -----------------------------------------------------
    # 2. Force image borders to black
    #    This removes edge-connected artifacts
    # -----------------------------------------------------

    BORDER = 5

    binary[
        :BORDER,
        :
    ] = 0

    binary[
        -BORDER:,
        :
    ] = 0

    binary[
        :,
        :BORDER
    ] = 0

    binary[
        :,
        -BORDER:
    ] = 0


    # -----------------------------------------------------
    # 3. Morphological opening
    #    Remove small foreground noise
    # -----------------------------------------------------

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )


    # -----------------------------------------------------
    # 4. Closing
    #    Connect brain boundary
    # -----------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )


    # -----------------------------------------------------
    # 5. Connected component analysis
    # -----------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )


    if num_labels <= 1:
        return None, binary, cleaned


    image_center = np.array(
        [
            w / 2.0,
            h / 2.0
        ]
    )


    best_label = None
    best_score = -1


    # -----------------------------------------------------
    # Ignore label 0 = background
    # -----------------------------------------------------

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        x = stats[
            label,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            label,
            cv2.CC_STAT_TOP
        ]

        component_w = stats[
            label,
            cv2.CC_STAT_WIDTH
        ]

        component_h = stats[
            label,
            cv2.CC_STAT_HEIGHT
        ]


        # ---------------------------------------------
        # Reject tiny components
        # ---------------------------------------------

        if area < 500:
            continue


        # ---------------------------------------------
        # Reject components touching image edges
        # ---------------------------------------------

        touches_border = (
            x <= BORDER
            or
            y <= BORDER
            or
            x + component_w >= w - BORDER
            or
            y + component_h >= h - BORDER
        )

        if touches_border:
            continue


        # ---------------------------------------------
        # Prefer large components near image center
        # ---------------------------------------------

        centroid = np.array(
            centroids[label]
        )

        distance = np.linalg.norm(
            centroid - image_center
        )


        score = (
            float(area)
            /
            (1.0 + distance)
        )


        if score > best_score:

            best_score = score
            best_label = label


    # -----------------------------------------------------
    # Fallback:
    # if no valid centered component was found
    # choose largest component
    # -----------------------------------------------------

    if best_label is None:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        best_label = (
            np.argmax(areas)
            + 1
        )


    # -----------------------------------------------------
    # 6. Component mask
    # -----------------------------------------------------

    component_mask = np.zeros_like(
        image,
        dtype=np.uint8
    )

    component_mask[
        labels == best_label
    ] = 255


    # -----------------------------------------------------
    # 7. Find external contour
    # -----------------------------------------------------

    contours, _ = cv2.findContours(

        component_mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    if len(contours) == 0:

        return None, binary, cleaned


    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    # -----------------------------------------------------
    # 8. Fill contour completely
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 9. Smooth final mask slightly
    # -----------------------------------------------------

    final_mask = cv2.morphologyEx(

        final_mask,

        cv2.MORPH_CLOSE,

        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        ),

        iterations=1

    )


    return (
        final_mask,
        binary,
        cleaned
    )


# =========================================================
# CENTRAL ANALYSIS REGION
# =========================================================

def get_central_region(mask):

    points = cv2.findNonZero(
        mask
    )

    if points is None:
        return None


    x, y, w, h = cv2.boundingRect(
        points
    )


    x1 = int(
        x + 0.25 * w
    )

    x2 = int(
        x + 0.75 * w
    )

    y1 = int(
        y + 0.25 * h
    )

    y2 = int(
        y + 0.75 * h
    )


    return (
        x1,
        y1,
        x2,
        y2
    )


# =========================================================
# PILOT SAMPLE
# =========================================================

sample_frames = []


for class_name in CLASSES:

    class_df = df[
        df["original_class"]
        == class_name
    ]


    sample = class_df.sample(

        n=min(
            SAMPLES_PER_CLASS,
            len(class_df)
        ),

        random_state=42

    )


    sample_frames.append(
        sample
    )


pilot_df = pd.concat(
    sample_frames,
    ignore_index=True
)


# =========================================================
# PROCESS
# =========================================================

records = []


for _, row in pilot_df.iterrows():

    image_path = Path(
        row["processed_path"]
    )


    image = cv2.imread(

        str(image_path),

        cv2.IMREAD_GRAYSCALE

    )


    if image is None:

        print(
            "FAILED:",
            image_path
        )

        continue


    (
        final_mask,
        initial_binary,
        cleaned_binary

    ) = create_brain_mask_v2(
        image
    )


    if final_mask is None:

        print(
            "MASK FAILED:",
            image_path
        )

        continue


    # =====================================================
    # MASKED BRAIN
    # =====================================================

    masked_brain = cv2.bitwise_and(

        image,

        image,

        mask=final_mask

    )


    # =====================================================
    # CENTRAL REGION
    # =====================================================

    central_region = get_central_region(
        final_mask
    )


    overlay = cv2.cvtColor(

        masked_brain,

        cv2.COLOR_GRAY2RGB

    )


    if central_region is not None:

        x1, y1, x2, y2 = central_region


        cv2.rectangle(

            overlay,

            (x1, y1),

            (x2, y2),

            (255, 255, 255),

            2

        )


    # =====================================================
    # MASK STATISTICS
    # =====================================================

    brain_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    total_pixels = int(
        final_mask.size
    )


    brain_ratio = (
        brain_pixels
        /
        total_pixels
    )


    # =====================================================
    # VISUALIZATION
    # =====================================================

    fig, axes = plt.subplots(

        1,

        6,

        figsize=(22, 5)

    )


    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        "Processed MRI"
    )

    axes[0].axis("off")


    axes[1].imshow(
        initial_binary,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Initial Foreground"
    )

    axes[1].axis("off")


    axes[2].imshow(
        cleaned_binary,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[2].set_title(
        "Cleaned Foreground"
    )

    axes[2].axis("off")


    axes[3].imshow(
        final_mask,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[3].set_title(
        "Final Brain Mask V2"
    )

    axes[3].axis("off")


    axes[4].imshow(
        masked_brain,
        cmap="gray"
    )

    axes[4].set_title(
        "Masked Brain"
    )

    axes[4].axis("off")


    axes[5].imshow(
        overlay
    )

    axes[5].set_title(
        "Central Analysis Region"
    )

    axes[5].axis("off")


    fig.suptitle(

        f"{row['original_class']} | "
        f"{row['filename']}",

        fontsize=14

    )


    plt.tight_layout()


    output_name = (

        f"{row['original_class']}_"
        f"{row['filename']}_MASK_V2.png"

    )


    plt.savefig(

        OUTPUT_DIR /
        output_name,

        dpi=200

    )


    plt.close()


    # =====================================================
    # RECORD
    # =====================================================

    records.append({

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "processed_path":
            row["processed_path"],

        "brain_pixels":
            brain_pixels,

        "brain_ratio":
            brain_ratio,

        "mask_status":
            "PASS"

    })


    print(

        "MASK V2:",

        row["original_class"],

        row["filename"],

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
    "brain_mask_pilot_v2_report.csv"

)


report_df.to_csv(

    REPORT_FILE,

    index=False

)


# =========================================================
# SUMMARY
# =========================================================

print("\n==========================================")
print("BRAIN MASK PILOT V2 COMPLETE")
print("==========================================")

print("\nImages analyzed:")
print(
    len(report_df)
)


if len(report_df) > 0:

    print("\nBrain ratio summary:")

    print(

        report_df[
            "brain_ratio"
        ].describe()

    )


print("\nSaved inside:")
print(
    OUTPUT_DIR
)