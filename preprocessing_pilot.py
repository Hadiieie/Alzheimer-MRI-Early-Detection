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

PILOT_DIR = RESULTS_DIR / "PREPROCESSING_PILOT"

PILOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_FILE = (
    RESULTS_DIR /
    "clean_4class_grouped_metadata.csv"
)


# =========================================================
# LOAD CLEAN METADATA
# =========================================================

df = pd.read_csv(MASTER_FILE)


CLASSES = [

    "NonDemented",

    "VeryMildDemented",

    "MildDemented",

    "ModerateDemented"

]


SAMPLES_PER_CLASS = 5


# =========================================================
# PREPROCESSING FUNCTION
# =========================================================

def preprocess_image(image_path):

    # -------------------------------------
    # 1. Load as grayscale
    # -------------------------------------

    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )


    if gray is None:

        raise ValueError(
            f"Could not read image: {image_path}"
        )


    # -------------------------------------
    # 2. Resize to 220 x 220
    # -------------------------------------

    resized = cv2.resize(

        gray,

        (220, 220),

        interpolation=cv2.INTER_AREA

    )


    # -------------------------------------
    # 3. Histogram Equalization
    # -------------------------------------

    equalized = cv2.equalizeHist(
        resized
    )


    # -------------------------------------
    # 4. Median Filter 11x11
    # -------------------------------------

    filtered = cv2.medianBlur(

        equalized,

        11

    )


    # -------------------------------------
    # 5. Normalize to [0,1]
    # -------------------------------------

    normalized = (

        filtered.astype(np.float32)
        / 255.0

    )


    return (

        gray,

        resized,

        equalized,

        filtered,

        normalized

    )


# =========================================================
# SAMPLE IMAGES
# =========================================================

sample_rows = []


for class_name in CLASSES:

    class_df = df[
        df["original_class"]
        == class_name
    ]


    sampled = class_df.sample(

        n=SAMPLES_PER_CLASS,

        random_state=42

    )


    sample_rows.append(
        sampled
    )


pilot_df = pd.concat(
    sample_rows,
    ignore_index=True
)


# =========================================================
# PROCESS + CREATE FIGURES
# =========================================================

for index, row in pilot_df.iterrows():

    image_path = Path(
        row["path"]
    )


    (
        gray,
        resized,
        equalized,
        filtered,
        normalized

    ) = preprocess_image(
        image_path
    )


    fig, axes = plt.subplots(

        1,

        5,

        figsize=(18, 4)

    )


    images = [

        gray,

        resized,

        equalized,

        filtered,

        normalized

    ]


    titles = [

        "Original Grayscale",

        "Resized 220x220",

        "Histogram Equalized",

        "Median 11x11",

        "Normalized [0,1]"

    ]


    for ax, image, title in zip(

        axes,

        images,

        titles

    ):

        ax.imshow(

            image,

            cmap="gray"

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

        fontsize=14

    )


    plt.tight_layout()


    output_name = (

        f"{row['original_class']}_"
        f"{row['filename']}.png"

    )


    plt.savefig(

        PILOT_DIR /
        output_name,

        dpi=200

    )


    plt.close()


    print(

        f"Processed: "
        f"{row['original_class']} | "
        f"{row['filename']}"

    )


# =========================================================
# SAVE PILOT LIST
# =========================================================

pilot_df.to_csv(

    PILOT_DIR /
    "pilot_images.csv",

    index=False

)


print("\n==========================================")
print("PREPROCESSING PILOT COMPLETE")
print("==========================================")

print("\nOutput folder:")

print(
    PILOT_DIR
)