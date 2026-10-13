from pathlib import Path

import cv2
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

ROI_DIR = (
    RESULTS_DIR /
    "ROI_PILOT"
)

ROI_DIR.mkdir(
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
# FIND BRAIN BOUNDING BOX
# =========================================================

def get_brain_bbox(image):

    # Brain pixels are above black background
    mask = image > 5

    points = cv2.findNonZero(
        mask.astype("uint8")
    )

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(
        points
    )

    return x, y, w, h


# =========================================================
# CREATE INITIAL BILATERAL ROI
# =========================================================

def get_temporal_rois(
    x,
    y,
    w,
    h
):

    # Vertical range:
    # approximate medial-temporal level
    roi_y1 = int(
        y + 0.45 * h
    )

    roi_y2 = int(
        y + 0.78 * h
    )


    # Left medial-temporal ROI
    left_x1 = int(
        x + 0.18 * w
    )

    left_x2 = int(
        x + 0.48 * w
    )


    # Right medial-temporal ROI
    right_x1 = int(
        x + 0.52 * w
    )

    right_x2 = int(
        x + 0.82 * w
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


    return left_roi, right_roi


# =========================================================
# SAMPLE 5 IMAGES PER CLASS
# =========================================================

sample_frames = []


for class_name in CLASSES:

    class_df = df[
        df["original_class"]
        == class_name
    ]

    sample = class_df.sample(
        n=SAMPLES_PER_CLASS,
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
# PROCESS PILOT IMAGES
# =========================================================

roi_records = []


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


    bbox = get_brain_bbox(
        image
    )


    if bbox is None:

        print(
            "NO BRAIN MASK:",
            image_path
        )

        continue


    x, y, w, h = bbox


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


    # =====================================================
    # CREATE DISPLAY COPY
    # =====================================================

    display = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )


    # Brain bounding box
    cv2.rectangle(
        display,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        1
    )


    # Left ROI
    cv2.rectangle(
        display,
        (lx1, ly1),
        (lx2, ly2),
        (255, 255, 255),
        2
    )


    # Right ROI
    cv2.rectangle(
        display,
        (rx1, ry1),
        (rx2, ry2),
        (255, 255, 255),
        2
    )


    # =====================================================
    # SHOW ORIGINAL + ROI
    # =====================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
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
        display
    )

    axes[1].set_title(
        "Initial Bilateral ROI"
    )

    axes[1].axis(
        "off"
    )


    fig.suptitle(
        f"{row['original_class']} | "
        f"{row['filename']}",
        fontsize=13
    )


    plt.tight_layout()


    output_name = (
        f"{row['original_class']}_"
        f"{row['filename']}_ROI.png"
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
    # SAVE ROI COORDINATES
    # =====================================================

    roi_records.append({

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "processed_path":
            row["processed_path"],

        "brain_x":
            x,

        "brain_y":
            y,

        "brain_width":
            w,

        "brain_height":
            h,

        "left_x1":
            lx1,

        "left_y1":
            ly1,

        "left_x2":
            lx2,

        "left_y2":
            ly2,

        "right_x1":
            rx1,

        "right_y1":
            ry1,

        "right_x2":
            rx2,

        "right_y2":
            ry2
    })


    print(
        "ROI created:",
        row["original_class"],
        row["filename"]
    )


# =========================================================
# SAVE ROI REPORT
# =========================================================

roi_df = pd.DataFrame(
    roi_records
)


roi_df.to_csv(
    ROI_DIR /
    "roi_pilot_coordinates.csv",
    index=False
)


print("\n==========================================")
print("ROI PILOT COMPLETE")
print("==========================================")

print("\nImages analyzed:")
print(
    len(roi_df)
)

print("\nResults saved inside:")
print(
    ROI_DIR
)