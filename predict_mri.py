from pathlib import Path
import argparse
import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import cv2

from inference_core import (
    FROZEN_FEATURES,
    extract_single_mri_features,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FINAL_MODEL_DIR = (
    PROJECT_ROOT
    / "results"
    / "FINAL_MODEL"
)

FINAL_MODEL_FILE = (
    FINAL_MODEL_DIR
    / "final_model.joblib"
)

FINAL_FEATURES_FILE = (
    FINAL_MODEL_DIR
    / "final_features.txt"
)

FINAL_CONFIG_FILE = (
    FINAL_MODEL_DIR
    / "final_model_config.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "SINGLE_MRI_INFERENCE"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def load_frozen_feature_order():
    if not FINAL_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Frozen feature list not found:\n"
            f"{FINAL_FEATURES_FILE}"
        )

    with open(
        FINAL_FEATURES_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if features != FROZEN_FEATURES:
        raise RuntimeError(
            "Frozen feature list does not match "
            "the verified inference_core.py order."
        )

    return features


def load_frozen_threshold():
    if not FINAL_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Final model config not found:\n"
            f"{FINAL_CONFIG_FILE}"
        )

    with open(
        FINAL_CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    possible_keys = [
        "selected_threshold",
        "threshold",
        "frozen_threshold",
        "decision_threshold",
    ]

    for key in possible_keys:
        if key in config:
            return float(config[key]), config, key

    raise KeyError(
        "Could not find a frozen threshold in "
        f"{FINAL_CONFIG_FILE}. "
        f"Available keys: {list(config.keys())}"
    )


def load_frozen_model():
    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Frozen model not found:\n"
            f"{FINAL_MODEL_FILE}"
        )

    model = joblib.load(
        FINAL_MODEL_FILE
    )

    if not hasattr(
        model,
        "decision_function"
    ):
        raise RuntimeError(
            "Frozen model does not provide "
            "decision_function()."
        )

    return model


# ============================================================
# SINGLE MRI PREDICTION
# ============================================================

def predict_single_mri(image_path):
    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input MRI not found:\n"
            f"{image_path}"
        )

    # --------------------------------------------------------
    # 1. Load frozen artifacts
    # --------------------------------------------------------

    frozen_features = (
        load_frozen_feature_order()
    )

    (
        threshold,
        config,
        threshold_key
    ) = load_frozen_threshold()

    model = load_frozen_model()

    # --------------------------------------------------------
    # 2. Reproduce exact verified preprocessing + features
    # --------------------------------------------------------

    result = extract_single_mri_features(
        image_path
    )

    feature_dict = result[
        "features"
    ]

    feature_vector = np.array(
        [
            feature_dict[name]
            for name in frozen_features
        ],
        dtype=np.float64
    )

    # Final QC
    if feature_vector.shape != (30,):
        raise RuntimeError(
            "Expected exactly 30 features, got "
            f"{feature_vector.shape}."
        )

    if np.isnan(
        feature_vector
    ).any():
        raise RuntimeError(
            "Feature vector contains NaN."
        )

    if np.isinf(
        feature_vector
    ).any():
        raise RuntimeError(
            "Feature vector contains infinity."
        )

    # DataFrame preserves feature names/order.
    X = pd.DataFrame(
        [
            feature_vector
        ],
        columns=frozen_features
    )

    # --------------------------------------------------------
    # 3. Frozen model decision score
    # --------------------------------------------------------

    decision_score = float(
        np.ravel(
            model.decision_function(
                X
            )
        )[0]
    )

    # IMPORTANT:
    # Threshold rule is frozen from validation only.
    predicted_binary = int(
        decision_score
        >=
        threshold
    )

    predicted_label = (
        "Early_Demented"
        if predicted_binary == 1
        else "Normal"
    )

    margin = float(
        decision_score
        -
        threshold
    )

    # --------------------------------------------------------
    # 4. Save QC images
    # --------------------------------------------------------

    stem = image_path.stem

    case_dir = (
        OUTPUT_DIR
        / stem
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_path = (
        case_dir
        / "processed_220x220.png"
    )

    mask_path = (
        case_dir
        / "brain_mask_v4.png"
    )

    masked_path = (
        case_dir
        / "masked_processed_mri.png"
    )

    cv2.imwrite(
        str(processed_path),
        result[
            "processed_image"
        ]
    )

    cv2.imwrite(
        str(mask_path),
        result[
            "brain_mask"
        ]
    )

    masked = cv2.bitwise_and(
        result[
            "processed_image"
        ],
        result[
            "processed_image"
        ],
        mask=result[
            "brain_mask"
        ]
    )

    cv2.imwrite(
        str(masked_path),
        masked
    )

    # --------------------------------------------------------
    # 5. Save feature vector
    # --------------------------------------------------------

    feature_output = (
        case_dir
        / "features.csv"
    )

    feature_row = {
        "filename":
            image_path.name,
        "decision_score":
            decision_score,
        "threshold":
            threshold,
        "margin_from_threshold":
            margin,
        "prediction":
            predicted_label,
        "otsu_threshold":
            result[
                "otsu_threshold"
            ],
        "mask_status":
            result[
                "mask_status"
            ],
    }

    feature_row.update(
        feature_dict
    )

    pd.DataFrame(
        [
            feature_row
        ]
    ).to_csv(
        feature_output,
        index=False
    )

    # --------------------------------------------------------
    # 6. Save JSON report
    # --------------------------------------------------------

    report = {
        "project":
            "Alzheimer MRI Early-Detection Research Pipeline",

        "input_image":
            str(
                image_path.resolve()
            ),

        "model_file":
            str(
                FINAL_MODEL_FILE
            ),

        "model_sha256":
            sha256_file(
                FINAL_MODEL_FILE
            ),

        "number_of_features":
            len(
                frozen_features
            ),

        "feature_order":
            frozen_features,

        "threshold_source_key":
            threshold_key,

        "frozen_threshold":
            threshold,

        "decision_score":
            decision_score,

        "margin_from_threshold":
            margin,

        "prediction":
            predicted_label,

        "prediction_binary":
            predicted_binary,

        "brain_mask_status":
            result[
                "mask_status"
            ],

        "otsu_threshold":
            result[
                "otsu_threshold"
            ],

        "qc": {
            "feature_count_ok":
                len(
                    frozen_features
                ) == 30,

            "nan_count":
                int(
                    np.isnan(
                        feature_vector
                    ).sum()
                ),

            "inf_count":
                int(
                    np.isinf(
                        feature_vector
                    ).sum()
                ),
        },

        "saved_files": {
            "processed_image":
                str(
                    processed_path
                ),

            "brain_mask":
                str(
                    mask_path
                ),

            "masked_processed_mri":
                str(
                    masked_path
                ),

            "features_csv":
                str(
                    feature_output
                ),
        },

        "research_notice":
            (
                "Research-use output only. "
                "This system is not a clinical diagnosis "
                "and has not been established here as a "
                "medical device."
            ),
    }

    report_path = (
        case_dir
        / "prediction_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=4
        )

    return (
        report,
        case_dir
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Alzheimer MRI "
            "research classifier on one MRI image."
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to one input MRI image "
            "(JPG/PNG or other OpenCV-readable image)."
        )
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL SINGLE-MRI INFERENCE"
    )

    print(
        "=" * 70
    )

    print(
        "\nInput image:"
    )

    print(
        Path(
            args.image
        )
    )

    report, case_dir = (
        predict_single_mri(
            args.image
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "QC"
    )

    print(
        "=" * 70
    )

    print(
        "\nBrain Mask V4:"
    )

    print(
        report[
            "brain_mask_status"
        ]
    )

    print(
        "\nFeatures:"
    )

    print(
        f"{report['number_of_features']} / 30"
    )

    print(
        "\nNaN values:"
    )

    print(
        report[
            "qc"
        ][
            "nan_count"
        ]
    )

    print(
        "\nInfinite values:"
    )

    print(
        report[
            "qc"
        ][
            "inf_count"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FROZEN MODEL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "\nDecision score:"
    )

    print(
        f"{report['decision_score']:.12f}"
    )

    print(
        "\nFrozen threshold:"
    )

    print(
        f"{report['frozen_threshold']:.12f}"
    )

    print(
        "\nMargin from threshold:"
    )

    print(
        f"{report['margin_from_threshold']:+.12f}"
    )

    print(
        "\nFINAL PREDICTION:"
    )

    print(
        report[
            "prediction"
        ]
    )

    print(
        "\nOutput folder:"
    )

    print(
        case_dir
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Research use only — not a clinical diagnosis."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
