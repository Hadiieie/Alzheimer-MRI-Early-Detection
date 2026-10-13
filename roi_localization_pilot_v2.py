from pathlib import Path

import cv2
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

ROI_DIR = (
    RESULTS_DIR /
    "ROI_PILOT_V2"
)

ROI_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD PROCESSED DATASET METADATA
# =========================================================

df = pd.read_csv(
    PROCESSED_METADATA
)


print("\n==========================================")
print("ROI LOCALIZATION PILOT V2")
print("==========================================")

print("\nTotal processed images available:")
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
# FIND BRAIN BOUNDING BOX
# =========================================================

def get_brain_bbox(image):

    """
    Detect the approximate brain area by separating
    non-black pixels from the background.
    """

    # Anything > 5 is treated as part of the brain/image
    mask = image > 5

    mask_uint8 = (
        mask.astype("uint8")
        * 255
    )


    points = cv2.findNonZero(
        mask_uint8
    )


    if points is None:

        return None


    x, y, w, h = cv2.boundingRect(
        points
    )


    return x, y, w, h


# =========================================================
# CANDIDATE MEDIAL-TEMPORAL ROI V2
# =========================================================

def get_temporal_rois(
    x,
    y,
    w,
    h
):

    """
    Create two smaller and more medial candidate ROIs.

    IMPORTANT:
    These are engineering candidate regions only.
    They are NOT anatomically validated hippocampus masks.
    """


    # -----------------------------------------------------
    # Vertical ROI range
    # -----------------------------------------------------

    roi_y1 = int(
        y + 0.40 * h
    )

    roi_y2 = int(
        y + 0.67 * h
    )


    # -----------------------------------------------------
    # LEFT candidate ROI
    # -----------------------------------------------------

    left_x1 = int(
        x + 0.27 * w
    )

    left_x2 = int(
        x + 0.48 * w
    )


    # -----------------------------------------------------
    # RIGHT candidate ROI
    # -----------------------------------------------------

    right_x1 = int(
        x + 0.52 * w
    )

    right_x2 = int(
        x + 0.73 * w
    )


    left_roi = (

        left_x1,

        roi_y1,

        left_x2,

        roi_y2

    )


    right_roi = (

        right_x1,

        roi_y1,

        right_x2,

        roi_y2

    )


    return (
        left_roi,
        right_roi
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


    sample_size = min(
        SAMPLES_PER_CLASS,
        len(class_df)
    )


    sample = class_df.sample(

        n=sample_size,

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


print("\nPilot distribution:")

print(
    pilot_df[
        "original_class"
    ].value_counts()
)


# =========================================================
# STORAGE FOR ROI COORDINATES
# =========================================================

roi_records = []


# =========================================================
# PROCESS PILOT IMAGES
# =========================================================

for _, row in pilot_df.iterrows():

    image_path = Path(
        row["processed_path"]
    )


    # -----------------------------------------------------
    # Load final preprocessed MRI
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Detect brain bounding box
    # -----------------------------------------------------

    bbox = get_brain_bbox(
        image
    )


    if bbox is None:

        print(
            "NO BRAIN REGION FOUND:",
            image_path
        )

        continue


    x, y, w, h = bbox


    # -----------------------------------------------------
    # Generate candidate bilateral ROIs
    # -----------------------------------------------------

    (
        left_roi,
        right_roi

    ) = get_temporal_rois(

        x,
        y,
        w,
        h

    )


    lx1, ly1, lx2, ly2 = left_roi

    rx1, ry1, rx2, ry2 = right_roi


    # -----------------------------------------------------
    # Make RGB copy for visualization
    # -----------------------------------------------------

    display = cv2.cvtColor(

        image,

        cv2.COLOR_GRAY2RGB

    )


    # -----------------------------------------------------
    # Draw brain bounding box
    # Thin line
    # -----------------------------------------------------

    cv2.rectangle(

        display,

        (x, y),

        (x + w, y + h),

        (255, 255, 255),

        1

    )


    # -----------------------------------------------------
    # Draw LEFT candidate ROI
    # -----------------------------------------------------

    cv2.rectangle(

        display,

        (lx1, ly1),

        (lx2, ly2),

        (255, 255, 255),

        2

    )


    # -----------------------------------------------------
    # Draw RIGHT candidate ROI
    # -----------------------------------------------------

    cv2.rectangle(

        display,

        (rx1, ry1),

        (rx2, ry2),

        (255, 255, 255),

        2

    )


    # =====================================================
    # CROP ROIs FOR VISUAL INSPECTION
    # =====================================================

    left_crop = image[
        ly1:ly2,
        lx1:lx2
    ]


    right_crop = image[
        ry1:ry2,
        rx1:rx2
    ]


    # =====================================================
    # CREATE VISUALIZATION
    # =====================================================

    fig, axes = plt.subplots(

        1,

        4,

        figsize=(16, 5)

    )


    # -----------------------------------------------------
    # Original processed MRI
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # MRI with ROI rectangles
    # -----------------------------------------------------

    axes[1].imshow(
        display
    )

    axes[1].set_title(
        "Candidate Bilateral ROI V2"
    )

    axes[1].axis(
        "off"
    )


    # -----------------------------------------------------
    # Left ROI crop
    # -----------------------------------------------------

    axes[2].imshow(

        left_crop,

        cmap="gray"

    )

    axes[2].set_title(
        "Left Candidate ROI"
    )

    axes[2].axis(
        "off"
    )


    # -----------------------------------------------------
    # Right ROI crop
    # -----------------------------------------------------

    axes[3].imshow(

        right_crop,

        cmap="gray"

    )

    axes[3].set_title(
        "Right Candidate ROI"
    )

    axes[3].axis(
        "off"
    )


    fig.suptitle(

        f"{row['original_class']} | "
        f"{row['filename']}",

        fontsize=14

    )


    plt.tight_layout()


    # =====================================================
    # SAVE FIGURE
    # =====================================================

    output_name = (

        f"{row['original_class']}_"
        f"{row['filename']}_ROI_V2.png"

    )


    output_path = (
        ROI_DIR /
        output_name
    )


    plt.savefig(

        output_path,

        dpi=200

    )


    plt.close()


    # =====================================================
    # SAVE ROI INFORMATION
    # =====================================================

    roi_records.append({

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "processed_path":
            row["processed_path"],


        # Brain bounding box
        "brain_x":
            x,

        "brain_y":
            y,

        "brain_width":
            w,

        "brain_height":
            h,


        # Left ROI
        "left_x1":
            lx1,

        "left_y1":
            ly1,

        "left_x2":
            lx2,

        "left_y2":
            ly2,


        # Right ROI
        "right_x1":
            rx1,

        "right_y1":
            ry1,

        "right_x2":
            rx2,

        "right_y2":
            ry2,


        # ROI dimensions
        "left_roi_width":
            lx2 - lx1,

        "left_roi_height":
            ly2 - ly1,

        "right_roi_width":
            rx2 - rx1,

        "right_roi_height":
            ry2 - ry1

    })


    print(

        "ROI V2 created:",

        row["original_class"],

        row["filename"]

    )


# =========================================================
# SAVE ROI METADATA
# =========================================================

roi_df = pd.DataFrame(
    roi_records
)


ROI_COORDINATES_FILE = (

    ROI_DIR /
    "roi_pilot_v2_coordinates.csv"

)


roi_df.to_csv(

    ROI_COORDINATES_FILE,

    index=False

)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n==========================================")
print("ROI PILOT V2 COMPLETE")
print("==========================================")


print("\nImages successfully analyzed:")

print(
    len(roi_df)
)


print("\nClass distribution:")

print(

    roi_df[
        "original_class"
    ].value_counts()

)


if len(roi_df) > 0:

    print("\nLeft ROI size range:")

    print(
        roi_df[
            [
                "left_roi_width",
                "left_roi_height"
            ]
        ].describe()
    )


    print("\nRight ROI size range:")

    print(
        roi_df[
            [
                "right_roi_width",
                "right_roi_height"
            ]
        ].describe()
    )


print("\nROI images saved inside:")

print(
    ROI_DIR
)


print("\nROI coordinate report saved to:")

print(
    ROI_COORDINATES_FILE
)