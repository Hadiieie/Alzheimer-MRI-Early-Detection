from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

PROCESSED_METADATA = (
    RESULTS_DIR /
    "processed_metadata.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR /
    "BRAIN_MASK_PILOT"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD PROCESSED METADATA
# =========================================================

df = pd.read_csv(
    PROCESSED_METADATA
)


print("\n==========================================")
print("BRAIN MASK PILOT")
print("==========================================")

print("\nProcessed images available:")
print(len(df))


# =========================================================
# CLASSES
# =========================================================

CLASSES = [

    "NonDemented",

    "VeryMildDemented",

    "MildDemented",

    "ModerateDemented"

]


SAMPLES_PER_CLASS = 5


# =========================================================
# BRAIN MASK FUNCTION
# =========================================================

def create_brain_mask(image):

    """
    Creates a filled whole-brain silhouette mask.

    This is NOT hippocampus segmentation.
    It isolates the brain from the black background.
    """

    # -----------------------------------------------------
    # 1. Initial foreground threshold
    # -----------------------------------------------------

    initial_mask = np.zeros_like(
        image,
        dtype=np.uint8
    )

    initial_mask[
        image > 5
    ] = 255


    # -----------------------------------------------------
    # 2. Small morphological closing
    #    Connect small gaps in brain boundary
    # -----------------------------------------------------

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5, 5)

    )


    closed_mask = cv2.morphologyEx(

        initial_mask,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=2

    )


    # -----------------------------------------------------
    # 3. Find external contours
    # -----------------------------------------------------

    contours, _ = cv2.findContours(

        closed_mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )


    if len(contours) == 0:

        return None, initial_mask


    # -----------------------------------------------------
    # 4. Keep largest component only
    # -----------------------------------------------------

    largest_contour = max(

        contours,

        key=cv2.contourArea

    )


    # -----------------------------------------------------
    # 5. Create filled brain silhouette
    # -----------------------------------------------------

    final_mask = np.zeros_like(
        image,
        dtype=np.uint8
    )


    cv2.drawContours(

        final_mask,

        [largest_contour],

        contourIdx=-1,

        color=255,

        thickness=cv2.FILLED

    )


    # -----------------------------------------------------
    # 6. Final smoothing of mask boundary
    # -----------------------------------------------------

    final_mask = cv2.morphologyEx(

        final_mask,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=1

    )


    return final_mask, initial_mask


# =========================================================
# CENTRAL ANALYSIS REGION
# =========================================================

def get_central_region(mask):

    """
    Creates a central region relative to the detected
    brain bounding box.

    This is NOT called a hippocampal ROI.
    It is only a central-structure analysis region.
    """

    points = cv2.findNonZero(mask)

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
# SELECT PILOT IMAGES
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


print("\nPilot images selected:")
print(len(pilot_df))


print("\nPilot class distribution:")

print(

    pilot_df[
        "original_class"
    ].value_counts()

)


# =========================================================
# PROCESS PILOT
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
            "FAILED TO READ:",
            image_path
        )

        continue


    # =====================================================
    # BRAIN MASK
    # =====================================================

    brain_mask, initial_mask = (
        create_brain_mask(
            image
        )
    )


    if brain_mask is None:

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

        mask=brain_mask

    )


    # =====================================================
    # CENTRAL REGION
    # =====================================================

    central_region = get_central_region(
        brain_mask
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
            brain_mask > 0
        )
    )


    total_pixels = (
        brain_mask.shape[0]
        *
        brain_mask.shape[1]
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

        5,

        figsize=(20, 5)

    )


    # Original processed image
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


    # Initial threshold
    axes[1].imshow(

        initial_mask,

        cmap="gray",

        vmin=0,

        vmax=255

    )

    axes[1].set_title(
        "Initial Foreground"
    )

    axes[1].axis(
        "off"
    )


    # Final brain mask
    axes[2].imshow(

        brain_mask,

        cmap="gray",

        vmin=0,

        vmax=255

    )

    axes[2].set_title(
        "Final Brain Mask"
    )

    axes[2].axis(
        "off"
    )


    # Masked MRI
    axes[3].imshow(

        masked_brain,

        cmap="gray"

    )

    axes[3].set_title(
        "Masked Brain"
    )

    axes[3].axis(
        "off"
    )


    # Central analysis region
    axes[4].imshow(
        overlay
    )

    axes[4].set_title(
        "Central Analysis Region"
    )

    axes[4].axis(
        "off"
    )


    fig.suptitle(

        f"{row['original_class']} | "
        f"{row['filename']}",

        fontsize=14

    )


    plt.tight_layout()


    # =====================================================
    # SAVE PILOT IMAGE
    # =====================================================

    output_name = (

        f"{row['original_class']}_"
        f"{row['filename']}_BRAIN_MASK.png"

    )


    output_path = (
        OUTPUT_DIR /
        output_name
    )


    plt.savefig(

        output_path,

        dpi=200

    )


    plt.close()


    # =====================================================
    # RECORD RESULTS
    # =====================================================

    record = {

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "processed_path":
            row["processed_path"],

        "brain_pixels":
            brain_pixels,

        "total_pixels":
            total_pixels,

        "brain_ratio":
            brain_ratio,

        "mask_status":
            "PASS"

    }


    if central_region is not None:

        x1, y1, x2, y2 = (
            central_region
        )


        record.update({

            "central_x1": x1,

            "central_y1": y1,

            "central_x2": x2,

            "central_y2": y2

        })


    records.append(
        record
    )


    print(

        "Brain mask created:",

        row["original_class"],

        row["filename"],

        "| brain ratio =",

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
    "brain_mask_pilot_report.csv"

)


report_df.to_csv(

    REPORT_FILE,

    index=False

)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n==========================================")
print("BRAIN MASK PILOT COMPLETE")
print("==========================================")

print("\nImages successfully analyzed:")
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


print("\nPilot images saved inside:")

print(
    OUTPUT_DIR
)


print("\nReport saved to:")

print(
    REPORT_FILE
)