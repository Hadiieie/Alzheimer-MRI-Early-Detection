from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR /
    "clean_4class_grouped_metadata.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT /
    "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_METADATA = (
    RESULTS_DIR /
    "processed_metadata.csv"
)

FAILURE_REPORT = (
    RESULTS_DIR /
    "preprocessing_failures.csv"
)


# =========================================================
# LOAD CLEAN MASTER DATASET
# =========================================================

df = pd.read_csv(INPUT_FILE)

print("\n==========================================")
print("FULL DATASET PREPROCESSING")
print("==========================================")

print("\nImages to process:")
print(len(df))


# =========================================================
# CREATE CLASS FOLDERS
# =========================================================

classes = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]


for class_name in classes:

    (
        PROCESSED_DIR /
        class_name
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# RESIZE + PADDING
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
# CLAHE
# =========================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# =========================================================
# FINAL PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    # 1. Grayscale
    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise ValueError(
            "Could not read image"
        )

    # 2. Aspect-preserved resize + padding
    standardized = resize_with_padding(
        gray,
        target_size=220
    )

    # Save original zero-background mask
    background_mask = (
        standardized == 0
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

    # Restore true black background/padding
    filtered[
        background_mask
    ] = 0

    return filtered


# =========================================================
# PROCESS DATASET
# =========================================================

processed_records = []

failure_records = []


total = len(df)


for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    source_path = Path(
        row["path"]
    )

    class_name = (
        row["original_class"]
    )

    output_name = (
        source_path.stem +
        ".png"
    )

    output_path = (
        PROCESSED_DIR /
        class_name /
        output_name
    )

    try:

        # -------------------------------------
        # Resume support:
        # don't redo an existing valid image
        # -------------------------------------

        if output_path.exists():

            check_image = cv2.imread(
                str(output_path),
                cv2.IMREAD_GRAYSCALE
            )

            if (
                check_image is not None
                and
                check_image.shape
                == (220, 220)
            ):

                processed = check_image

            else:

                processed = preprocess_image(
                    source_path
                )

                cv2.imwrite(
                    str(output_path),
                    processed
                )

        else:

            processed = preprocess_image(
                source_path
            )

            success = cv2.imwrite(
                str(output_path),
                processed
            )

            if not success:

                raise ValueError(
                    "cv2.imwrite failed"
                )


        # -------------------------------------
        # QC
        # -------------------------------------

        shape_ok = (
            processed.shape
            == (220, 220)
        )

        min_value = int(
            processed.min()
        )

        max_value = int(
            processed.max()
        )

        if not shape_ok:

            raise ValueError(
                "Incorrect output shape"
            )


        # -------------------------------------
        # Metadata
        # -------------------------------------

        record = row.to_dict()

        record[
            "processed_path"
        ] = str(
            output_path
        )

        record[
            "processed_width"
        ] = 220

        record[
            "processed_height"
        ] = 220

        record[
            "processed_min"
        ] = min_value

        record[
            "processed_max"
        ] = max_value

        record[
            "preprocessing_status"
        ] = "PASS"

        processed_records.append(
            record
        )


    except Exception as e:

        failure_records.append({

            "filename":
                row["filename"],

            "original_class":
                class_name,

            "source_path":
                str(source_path),

            "error":
                str(e)

        })


    if position % 1000 == 0:

        print(
            f"Processed "
            f"{position} / {total}"
        )


# =========================================================
# SAVE METADATA
# =========================================================

processed_df = pd.DataFrame(
    processed_records
)

processed_df.to_csv(
    OUTPUT_METADATA,
    index=False
)


failures_df = pd.DataFrame(
    failure_records
)

failures_df.to_csv(
    FAILURE_REPORT,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n==========================================")
print("FULL PREPROCESSING COMPLETE")
print("==========================================")

print("\nTotal input images:")
print(total)

print("\nSuccessfully processed:")
print(
    len(processed_df)
)

print("\nFailures:")
print(
    len(failures_df)
)


if len(processed_df) > 0:

    print("\nProcessed class distribution:")

    print(
        processed_df[
            "original_class"
        ].value_counts()
    )


    print("\nAll output shapes:")

    print(
        processed_df[
            [
                "processed_width",
                "processed_height"
            ]
        ]
        .value_counts()
    )


    print("\nOverall output minimum:")

    print(
        processed_df[
            "processed_min"
        ].min()
    )


    print("\nOverall output maximum:")

    print(
        processed_df[
            "processed_max"
        ].max()
    )


print("\nProcessed images saved inside:")

print(
    PROCESSED_DIR
)


print("\nProcessed metadata saved to:")

print(
    OUTPUT_METADATA
)


print("\nFailure report saved to:")

print(
    FAILURE_REPORT
)