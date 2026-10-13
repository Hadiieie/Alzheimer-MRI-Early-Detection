from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy


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
    "FEATURE_EXTRACTION_PILOT"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "feature_extraction_pilot.csv"
)


# =========================================================
# LOAD DATA
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
# BRAIN MASK V2
# SAME APPROVED METHOD
# =========================================================

def create_brain_mask_v2(image):

    h, w = image.shape

    binary = np.zeros_like(
        image,
        dtype=np.uint8
    )

    binary[image > 8] = 255


    BORDER = 5

    binary[:BORDER, :] = 0
    binary[-BORDER:, :] = 0
    binary[:, :BORDER] = 0
    binary[:, -BORDER:] = 0


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


    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )


    if num_labels <= 1:
        return None


    image_center = np.array(
        [w / 2.0, h / 2.0]
    )


    best_label = None
    best_score = -1


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


        if area < 500:
            continue


        touches_border = (
            x <= BORDER
            or y <= BORDER
            or x + component_w >= w - BORDER
            or y + component_h >= h - BORDER
        )


        if touches_border:
            continue


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


    if best_label is None:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        best_label = (
            np.argmax(areas)
            + 1
        )


    component_mask = np.zeros_like(
        image,
        dtype=np.uint8
    )

    component_mask[
        labels == best_label
    ] = 255


    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) == 0:
        return None


    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


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


    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        ),
        iterations=1
    )


    return final_mask


# =========================================================
# CENTRAL REGION
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


    x1 = int(x + 0.25 * w)
    x2 = int(x + 0.75 * w)

    y1 = int(y + 0.25 * h)
    y2 = int(y + 0.75 * h)


    return (
        x1,
        y1,
        x2,
        y2
    )


# =========================================================
# SAFE ENTROPY
# =========================================================

def entropy_from_pixels(pixels):

    if len(pixels) == 0:
        return np.nan

    hist, _ = np.histogram(
        pixels,
        bins=256,
        range=(0, 256)
    )

    hist = hist.astype(
        np.float64
    )

    hist = hist[
        hist > 0
    ]

    probabilities = (
        hist /
        hist.sum()
    )

    return float(
        -np.sum(
            probabilities
            *
            np.log2(probabilities)
        )
    )


# =========================================================
# GLCM FEATURES
# =========================================================

def extract_glcm_features(region):

    # Quantize to 16 gray levels
    quantized = (
        region.astype(np.float32)
        / 256.0
        * 16.0
    ).astype(np.uint8)

    quantized = np.clip(
        quantized,
        0,
        15
    )


    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=16,
        symmetric=True,
        normed=True
    )


    return {

        "glcm_contrast":
            float(
                graycoprops(
                    glcm,
                    "contrast"
                ).mean()
            ),

        "glcm_homogeneity":
            float(
                graycoprops(
                    glcm,
                    "homogeneity"
                ).mean()
            ),

        "glcm_energy":
            float(
                graycoprops(
                    glcm,
                    "energy"
                ).mean()
            ),

        "glcm_correlation":
            float(
                graycoprops(
                    glcm,
                    "correlation"
                ).mean()
            )
    }


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    mask = create_brain_mask_v2(
        image
    )


    if mask is None:
        return None


    brain_pixels = image[
        mask > 0
    ]


    if len(brain_pixels) == 0:
        return None


    # =====================================================
    # MORPHOLOGY
    # =====================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    brain_area = float(
        cv2.contourArea(
            largest_contour
        )
    )


    brain_perimeter = float(
        cv2.arcLength(
            largest_contour,
            True
        )
    )


    total_image_area = float(
        image.shape[0]
        *
        image.shape[1]
    )


    brain_ratio = (
        brain_area
        /
        total_image_area
    )


    if brain_perimeter > 0:

        compactness = (
            4.0
            *
            np.pi
            *
            brain_area
            /
            (
                brain_perimeter
                ** 2
            )
        )

    else:

        compactness = np.nan


    x, y, w, h = cv2.boundingRect(
        largest_contour
    )


    bbox_aspect_ratio = (
        float(w)
        /
        float(h)
        if h > 0
        else np.nan
    )


    # =====================================================
    # WHOLE-BRAIN INTENSITY
    # =====================================================

    mean_intensity = float(
        np.mean(brain_pixels)
    )

    std_intensity = float(
        np.std(brain_pixels)
    )

    median_intensity = float(
        np.median(brain_pixels)
    )


    p10 = float(
        np.percentile(
            brain_pixels,
            10
        )
    )

    p25 = float(
        np.percentile(
            brain_pixels,
            25
        )
    )

    p75 = float(
        np.percentile(
            brain_pixels,
            75
        )
    )

    p90 = float(
        np.percentile(
            brain_pixels,
            90
        )
    )


    entropy = entropy_from_pixels(
        brain_pixels
    )


    # =====================================================
    # EDGE DENSITY
    # =====================================================

    edges = cv2.Canny(
        image,
        50,
        150
    )


    edge_pixels_inside_brain = np.sum(
        (
            edges > 0
        )
        &
        (
            mask > 0
        )
    )


    mask_pixels = np.sum(
        mask > 0
    )


    edge_density = (
        float(
            edge_pixels_inside_brain
        )
        /
        float(
            mask_pixels
        )
    )


    # =====================================================
    # CENTRAL REGION
    # =====================================================

    central = get_central_region(
        mask
    )


    if central is None:
        return None


    x1, y1, x2, y2 = central


    central_image = image[
        y1:y2,
        x1:x2
    ]


    central_mask = mask[
        y1:y2,
        x1:x2
    ]


    central_pixels = central_image[
        central_mask > 0
    ]


    if len(central_pixels) == 0:
        return None


    central_mean = float(
        np.mean(
            central_pixels
        )
    )


    central_std = float(
        np.std(
            central_pixels
        )
    )


    central_median = float(
        np.median(
            central_pixels
        )
    )


    central_entropy = (
        entropy_from_pixels(
            central_pixels
        )
    )


    # 0.30 × 255 ≈ 77
    central_dark_ratio = float(
        np.mean(
            central_pixels
            <= 77
        )
    )


    # 0.70 × 255 ≈ 179
    central_bright_ratio = float(
        np.mean(
            central_pixels
            >= 179
        )
    )


    # =====================================================
    # LEFT / RIGHT CENTRAL HALVES
    # =====================================================

    mid_x = (
        central_image.shape[1]
        //
        2
    )


    left_image = central_image[
        :,
        :mid_x
    ]


    right_image = central_image[
        :,
        mid_x:
    ]


    left_mask = central_mask[
        :,
        :mid_x
    ]


    right_mask = central_mask[
        :,
        mid_x:
    ]


    left_pixels = left_image[
        left_mask > 0
    ]


    right_pixels = right_image[
        right_mask > 0
    ]


    left_mean = float(
        np.mean(
            left_pixels
        )
    )


    right_mean = float(
        np.mean(
            right_pixels
        )
    )


    left_dark_ratio = float(
        np.mean(
            left_pixels
            <= 77
        )
    )


    right_dark_ratio = float(
        np.mean(
            right_pixels
            <= 77
        )
    )


    intensity_asymmetry = float(
        abs(
            left_mean
            -
            right_mean
        )
        /
        (
            (
                left_mean
                +
                right_mean
            )
            / 2.0
            +
            1e-8
        )
    )


    dark_ratio_asymmetry = float(
        abs(
            left_dark_ratio
            -
            right_dark_ratio
        )
    )


    # =====================================================
    # CENTRAL TEXTURE / GLCM
    # =====================================================

    glcm_features = (
        extract_glcm_features(
            central_image
        )
    )


    # =====================================================
    # FINAL FEATURE DICTIONARY
    # =====================================================

    features = {

        "brain_area":
            brain_area,

        "brain_ratio":
            brain_ratio,

        "brain_perimeter":
            brain_perimeter,

        "compactness":
            compactness,

        "bbox_aspect_ratio":
            bbox_aspect_ratio,

        "mean_intensity":
            mean_intensity,

        "std_intensity":
            std_intensity,

        "median_intensity":
            median_intensity,

        "entropy":
            entropy,

        "p10":
            p10,

        "p25":
            p25,

        "p75":
            p75,

        "p90":
            p90,

        "edge_density":
            edge_density,

        "central_mean":
            central_mean,

        "central_std":
            central_std,

        "central_median":
            central_median,

        "central_entropy":
            central_entropy,

        "central_dark_ratio":
            central_dark_ratio,

        "central_bright_ratio":
            central_bright_ratio,

        "left_mean":
            left_mean,

        "right_mean":
            right_mean,

        "left_dark_ratio":
            left_dark_ratio,

        "right_dark_ratio":
            right_dark_ratio,

        "intensity_asymmetry":
            intensity_asymmetry,

        "dark_ratio_asymmetry":
            dark_ratio_asymmetry
    }


    features.update(
        glcm_features
    )


    return features


# =========================================================
# SELECT SAME 20-PILOT STRUCTURE
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
# EXTRACT
# =========================================================

records = []

failures = []


for _, row in pilot_df.iterrows():

    path = Path(
        row["processed_path"]
    )


    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )


    if image is None:

        failures.append(
            row["filename"]
        )

        continue


    features = extract_features(
        image
    )


    if features is None:

        failures.append(
            row["filename"]
        )

        continue


    record = {

        "filename":
            row["filename"],

        "original_class":
            row["original_class"],

        "leakage_group":
            row["leakage_group"],

        "processed_path":
            row["processed_path"]

    }


    record.update(
        features
    )


    records.append(
        record
    )


    print(
        "Features extracted:",
        row["original_class"],
        row["filename"]
    )


# =========================================================
# DATAFRAME
# =========================================================

feature_df = pd.DataFrame(
    records
)


feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# QC
# =========================================================

feature_columns = [
    col
    for col in feature_df.columns
    if col not in [
        "filename",
        "original_class",
        "leakage_group",
        "processed_path"
    ]
]


nan_count = int(
    feature_df[
        feature_columns
    ]
    .isna()
    .sum()
    .sum()
)


inf_count = int(
    np.isinf(
        feature_df[
            feature_columns
        ].to_numpy(
            dtype=float
        )
    ).sum()
)


constant_features = [

    col

    for col in feature_columns

    if feature_df[
        col
    ].nunique() <= 1

]


# =========================================================
# SUMMARY
# =========================================================

print("\n==========================================")
print("FEATURE EXTRACTION PILOT COMPLETE")
print("==========================================")

print("\nImages requested:")
print(
    len(pilot_df)
)

print("\nSuccessfully extracted:")
print(
    len(feature_df)
)

print("\nFailures:")
print(
    len(failures)
)

print("\nNumber of features:")
print(
    len(feature_columns)
)

print("\nTotal NaN values:")
print(
    nan_count
)

print("\nTotal infinite values:")
print(
    inf_count
)

print("\nConstant features:")
print(
    constant_features
)


print("\nFeature ranges:\n")

print(
    feature_df[
        feature_columns
    ]
    .agg(
        [
            "min",
            "mean",
            "max"
        ]
    )
    .T
)


print("\nSaved to:")

print(
    OUTPUT_FILE
)