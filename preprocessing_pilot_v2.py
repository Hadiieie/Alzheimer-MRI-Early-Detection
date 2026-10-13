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

OLD_PILOT_DIR = RESULTS_DIR / "PREPROCESSING_PILOT"

NEW_PILOT_DIR = RESULTS_DIR / "PREPROCESSING_PILOT_V2"

NEW_PILOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_FILE = (
    RESULTS_DIR /
    "clean_4class_grouped_metadata.csv"
)


PILOT_LIST = (
    OLD_PILOT_DIR /
    "pilot_images.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(MASTER_FILE)

pilot_df = pd.read_csv(PILOT_LIST)


# =========================================================
# RESIZE WITH ASPECT RATIO + PADDING
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
# CLAHE
# =========================================================

clahe = cv2.createCLAHE(

    clipLimit=2.0,

    tileGridSize=(8, 8)

)


# =========================================================
# PROCESS EACH PILOT IMAGE
# =========================================================

for _, row in pilot_df.iterrows():

    path = Path(
        row["path"]
    )


    gray = cv2.imread(

        str(path),

        cv2.IMREAD_GRAYSCALE

    )


    if gray is None:

        print(
            "Failed:",
            path
        )

        continue


    # -------------------------------------
    # Aspect-ratio preserving resize
    # -------------------------------------

    standardized = resize_with_padding(
        gray,
        220
    )


    # -------------------------------------
    # Global histogram equalization
    # -------------------------------------

    global_eq = cv2.equalizeHist(
        standardized
    )


    # -------------------------------------
    # CLAHE
    # -------------------------------------

    clahe_img = clahe.apply(
        standardized
    )


    # -------------------------------------
    # Different median kernels
    # -------------------------------------

    median3 = cv2.medianBlur(
        clahe_img,
        3
    )


    median5 = cv2.medianBlur(
        clahe_img,
        5
    )


    median7 = cv2.medianBlur(
        clahe_img,
        7
    )


    median11 = cv2.medianBlur(
        clahe_img,
        11
    )


    # -------------------------------------
    # Plot comparison
    # -------------------------------------

    images = [

        gray,

        standardized,

        global_eq,

        clahe_img,

        median3,

        median5,

        median7,

        median11

    ]


    titles = [

        "Original",

        "220x220\nAspect Preserved",

        "Global Hist Eq",

        "CLAHE",

        "Median 3x3",

        "Median 5x5",

        "Median 7x7",

        "Median 11x11"

    ]


    fig, axes = plt.subplots(

        2,

        4,

        figsize=(18, 9)

    )


    axes = axes.flatten()


    for ax, img, title in zip(

        axes,

        images,

        titles

    ):

        ax.imshow(

            img,

            cmap="gray",

            vmin=0,

            vmax=255

        )

        ax.set_title(
            title
        )

        ax.axis(
            "off"
        )


    fig.suptitle(

        f"{row['original_class']} | "
        f"{row['filename']}",

        fontsize=15

    )


    plt.tight_layout()


    output_name = (

        f"{row['original_class']}_"
        f"{row['filename']}_V2.png"

    )


    plt.savefig(

        NEW_PILOT_DIR /
        output_name,

        dpi=200

    )


    plt.close()


    print(

        "Processed:",

        row["original_class"],

        row["filename"]

    )


print("\n==========================================")
print("PREPROCESSING PILOT V2 COMPLETE")
print("==========================================")

print("\nSaved to:")

print(
    NEW_PILOT_DIR
)