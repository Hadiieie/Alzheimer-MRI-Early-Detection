from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops


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
    "FULL_FEATURE_EXTRACTION"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_FILE = (
    OUTPUT_DIR /
    "full_features_39538.csv"
)

FAILURE_FILE = (
    OUTPUT_DIR /
    "feature_extraction_failures.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR /
    "feature_summary.csv"
)

CHECKPOINT_FILE = (
    OUTPUT_DIR /
    "feature_extraction_checkpoint.csv"
)


# =========================================================
# LOAD METADATA
# =========================================================

df = pd.read_csv(
    METADATA_FILE
)


print("\n==========================================")
print("FULL FEATURE EXTRACTION")
print("==========================================")

print("\nImages to process:")
print(len(df))


# =========================================================
# RESIZE ORIGINAL RAW MRI
# SAME STANDARDIZATION USED IN V4
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
# BORDER UTILITY
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
# FINAL APPROVED BRAIN MASK V4
# =========================================================

def create_brain_mask_v4(
    standardized_raw
):

    h, w = standardized_raw.shape

    BORDER = 5


    # -----------------------------------------------------
    # Gaussian smoothing
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        standardized_raw,
        (5, 5),
        0
    )


    # -----------------------------------------------------
    # Adaptive Otsu threshold
    # -----------------------------------------------------

    otsu_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )


    binary = zero_border(
        binary,
        BORDER
    )


    # -----------------------------------------------------
    # Morphological opening
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Morphological closing
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Connected components
    # -----------------------------------------------------

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
        return None, otsu_value, "NO_COMPONENT"


    image_center = np.array(
        [
            w / 2.0,
            h / 2.0
        ]
    )


    candidates = []


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


        if area < 1000:
            continue


        if (
            area_ratio < 0.15
            or
            area_ratio > 0.85
        ):
            continue


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


        score = (
            area_ratio
            -
            0.20 *
            normalized_distance
        )


        candidates.append({

            "label":
                label,

            "score":
                score

        })


    if len(candidates) == 0:
        return None, otsu_value, "NO_VALID_COMPONENT"


    best = max(
        candidates,
        key=lambda item:
            item["score"]
    )


    best_label = (
        best["label"]
    )


    component_mask = np.zeros_like(
        standardized_raw,
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
        return None, otsu_value, "NO_CONTOUR"


    contour = max(
        contours,
        key=cv2.contourArea
    )


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


    brain_ratio = float(
        np.mean(
            final_mask > 0
        )
    )


    if (
        brain_ratio < 0.20
        or
        brain_ratio > 0.85
    ):

        return (
            None,
            otsu_value,
            "FAILED_RATIO_QC"
        )


    return (
        final_mask,
        otsu_value,
        "PASS"
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
# ENTROPY
# =========================================================

def entropy_from_pixels(
    pixels
):

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


    entropy = -np.sum(
        probabilities
        *
        np.log2(
            probabilities
        )
    )


    return float(
        entropy
    )


# =========================================================
# GLCM FEATURES
# =========================================================

def extract_glcm_features(
    region
):

    quantized = (
        region.astype(
            np.float32
        )
        /
        256.0
        *
        16.0
    ).astype(
        np.uint8
    )


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

def extract_features(
    processed_image,
    brain_mask
):

    # -----------------------------------------------------
    # Brain pixels
    # -----------------------------------------------------

    brain_pixels = processed_image[
        brain_mask > 0
    ]


    if len(brain_pixels) == 0:
        return None


    # =====================================================
    # MORPHOLOGICAL FEATURES
    # =====================================================

    contours, _ = cv2.findContours(
        brain_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) == 0:
        return None


    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    # Final definition:
    # brain_area = number of pixels inside approved V4 mask
    brain_area = float(
        np.sum(
            brain_mask > 0
        )
    )


    brain_ratio = (
        brain_area
        /
        float(
            brain_mask.size
        )
    )


    brain_perimeter = float(
        cv2.arcLength(
            largest_contour,
            True
        )
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
    # WHOLE-BRAIN INTENSITY FEATURES
    # =====================================================

    mean_intensity = float(
        np.mean(
            brain_pixels
        )
    )


    std_intensity = float(
        np.std(
            brain_pixels
        )
    )


    median_intensity = float(
        np.median(
            brain_pixels
        )
    )


    entropy = entropy_from_pixels(
        brain_pixels
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


    # =====================================================
    # EDGE DENSITY
    # =====================================================

    edges = cv2.Canny(
        processed_image,
        50,
        150
    )


    edge_pixels_inside_brain = int(
        np.sum(
            (edges > 0)
            &
            (brain_mask > 0)
        )
    )


    mask_pixels = int(
        np.sum(
            brain_mask > 0
        )
    )


    edge_density = (
        edge_pixels_inside_brain
        /
        float(mask_pixels)
    )


    # =====================================================
    # CENTRAL ANALYSIS REGION
    # =====================================================

    central = get_central_region(
        brain_mask
    )


    if central is None:
        return None


    x1, y1, x2, y2 = central


    central_image = processed_image[
        y1:y2,
        x1:x2
    ]


    central_mask = brain_mask[
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


    # Dark threshold:
    # 30% of 255 ≈ 77

    central_dark_ratio = float(
        np.mean(
            central_pixels <= 77
        )
    )


    # Bright threshold:
    # 70% of 255 ≈ 179

    central_bright_ratio = float(
        np.mean(
            central_pixels >= 179
        )
    )


    # =====================================================
    # LEFT / RIGHT CENTRAL ANALYSIS
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


    if (
        len(left_pixels) == 0
        or
        len(right_pixels) == 0
    ):
        return None


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
            left_pixels <= 77
        )
    )


    right_dark_ratio = float(
        np.mean(
            right_pixels <= 77
        )
    )


    intensity_asymmetry = float(

        abs(
            left_mean -
            right_mean
        )

        /

        (
            (
                left_mean +
                right_mean
            )
            /
            2.0
            +
            1e-8
        )
    )


    dark_ratio_asymmetry = float(
        abs(
            left_dark_ratio -
            right_dark_ratio
        )
    )


    # =====================================================
    # GLCM TEXTURE
    # =====================================================

    glcm_features = (
        extract_glcm_features(
            central_image
        )
    )


    # =====================================================
    # FINAL 30 FEATURES
    # =====================================================

    features = {

        # Geometry
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


        # Whole-brain intensity
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


        # Edge
        "edge_density":
            edge_density,


        # Central region
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


        # Left/right analysis
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
# FULL EXTRACTION
# =========================================================

records = []

failures = []

total = len(df)


for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    filename = row["filename"]

    original_class = row["original_class"]


    # =====================================================
    # READ ORIGINAL RAW MRI
    # =====================================================

    raw_path = Path(
        row["path"]
    )


    raw_image = cv2.imread(
        str(raw_path),
        cv2.IMREAD_GRAYSCALE
    )


    if raw_image is None:

        failures.append({

            "filename":
                filename,

            "original_class":
                original_class,

            "stage":
                "RAW_READ",

            "reason":
                "READ_FAILURE"

        })

        continue


    standardized_raw = resize_with_padding(
        raw_image,
        target_size=220
    )


    # =====================================================
    # CREATE APPROVED V4 MASK FROM RAW MRI
    # =====================================================

    (
        brain_mask,
        otsu_threshold,
        mask_status

    ) = create_brain_mask_v4(
        standardized_raw
    )


    if brain_mask is None:

        failures.append({

            "filename":
                filename,

            "original_class":
                original_class,

            "stage":
                "BRAIN_MASK",

            "reason":
                mask_status

        })

        continue


    # =====================================================
    # READ PROCESSED MRI
    # =====================================================

    processed_path = Path(
        row["processed_path"]
    )


    processed_image = cv2.imread(
        str(processed_path),
        cv2.IMREAD_GRAYSCALE
    )


    if processed_image is None:

        failures.append({

            "filename":
                filename,

            "original_class":
                original_class,

            "stage":
                "PROCESSED_READ",

            "reason":
                "READ_FAILURE"

        })

        continue


    # =====================================================
    # SAFETY SHAPE CHECK
    # =====================================================

    if processed_image.shape != brain_mask.shape:

        failures.append({

            "filename":
                filename,

            "original_class":
                original_class,

            "stage":
                "SHAPE_CHECK",

            "reason":
                (
                    f"processed={processed_image.shape}, "
                    f"mask={brain_mask.shape}"
                )

        })

        continue


    # =====================================================
    # EXTRACT 30 FEATURES
    # =====================================================

    features = extract_features(
        processed_image,
        brain_mask
    )


    if features is None:

        failures.append({

            "filename":
                filename,

            "original_class":
                original_class,

            "stage":
                "FEATURE_EXTRACTION",

            "reason":
                "FEATURE_FAILURE"

        })

        continue


    # =====================================================
    # METADATA
    # =====================================================

    record = {

        "filename":
            filename,

        "original_class":
            original_class,

        "path":
            str(raw_path),

        "processed_path":
            str(processed_path),

        "otsu_threshold":
            float(
                otsu_threshold
            )

    }


    # Keep leakage group.
    if "leakage_group" in row.index:

        record[
            "leakage_group"
        ] = row[
            "leakage_group"
        ]


    if "study_group" in row.index:

        record[
            "study_group"
        ] = row[
            "study_group"
        ]


    record.update(
        features
    )


    records.append(
        record
    )


    # =====================================================
    # PROGRESS
    # =====================================================

    if position % 1000 == 0:

        print(
            f"Processed "
            f"{position} / {total}"
        )


    # =====================================================
    # CHECKPOINT EVERY 5000 IMAGES
    # =====================================================

    if (
        position % 5000 == 0
        and
        len(records) > 0
    ):

        checkpoint_df = pd.DataFrame(
            records
        )

        checkpoint_df.to_csv(
            CHECKPOINT_FILE,
            index=False
        )

        print(
            "Checkpoint saved:",
            len(checkpoint_df),
            "successful rows"
        )


# =========================================================
# CREATE FINAL DATAFRAME
# =========================================================

feature_df = pd.DataFrame(
    records
)


failure_df = pd.DataFrame(
    failures
)


# =========================================================
# SAVE FINAL FEATURES
# =========================================================

feature_df.to_csv(
    FEATURE_FILE,
    index=False
)


failure_df.to_csv(
    FAILURE_FILE,
    index=False
)


# =========================================================
# IDENTIFY FEATURE COLUMNS
# =========================================================

NON_FEATURE_COLUMNS = [

    "filename",
    "original_class",
    "path",
    "processed_path",
    "otsu_threshold",
    "leakage_group",
    "study_group"

]


feature_columns = [

    column

    for column in feature_df.columns

    if column
    not in NON_FEATURE_COLUMNS

]


# =========================================================
# QC
# =========================================================

numeric_features = feature_df[
    feature_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)


nan_count = int(
    numeric_features
    .isna()
    .sum()
    .sum()
)


inf_count = int(
    np.isinf(
        numeric_features.to_numpy(
            dtype=float
        )
    ).sum()
)


constant_features = [

    column

    for column in feature_columns

    if feature_df[
        column
    ].nunique(
        dropna=False
    ) <= 1

]


# =========================================================
# FEATURE SUMMARY
# =========================================================

summary_df = (
    numeric_features
    .agg(
        [
            "min",
            "mean",
            "median",
            "std",
            "max"
        ]
    )
    .T
)


summary_df.to_csv(
    SUMMARY_FILE
)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n==========================================")
print("FULL FEATURE EXTRACTION COMPLETE")
print("==========================================")


print("\nImages requested:")
print(
    total
)


print("\nSuccessfully extracted:")
print(
    len(feature_df)
)


print("\nFailures:")
print(
    len(failure_df)
)


print("\nNumber of numerical features:")
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


print("\nClass distribution:")

print(
    feature_df[
        "original_class"
    ].value_counts()
)


print("\nFeature summary:")

print(
    summary_df
)


print("\n==========================================")
print("FILES SAVED")
print("==========================================")


print("\nFull feature dataset:")
print(
    FEATURE_FILE
)


print("\nFailures:")
print(
    FAILURE_FILE
)


print("\nFeature summary:")
print(
    SUMMARY_FILE
)