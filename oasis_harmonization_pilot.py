from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from inference_core import (
    FROZEN_FEATURES,
    preprocess_from_raw,
    create_brain_mask_v4,
    extract_features,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = (
    PROJECT_ROOT
    / "results"
    / "GROUP_AWARE_SPLIT"
    / "train.csv"
)

MANIFEST = (
    PROJECT_ROOT
    / "NEW MRI"
    / "OASIS_BATCH"
    / "oasis_validation_manifest.csv"
)

EXTRACTION_SUMMARY = (
    PROJECT_ROOT
    / "NEW MRI"
    / "OASIS_FREESURFER_BATCH"
    / "freesurfer_slice_extraction_summary.csv"
)

FINAL_MODEL = (
    PROJECT_ROOT
    / "results"
    / "FINAL_MODEL"
    / "final_model.joblib"
)

FINAL_CONFIG = (
    PROJECT_ROOT
    / "results"
    / "FINAL_MODEL"
    / "final_model_config.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "OASIS_HARMONIZATION_PILOT"
)

IMAGE_DIR = OUTPUT_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "harmonization_pilot_predictions.csv"
SHIFT_CSV = OUTPUT_DIR / "harmonization_feature_shift.csv"
SUMMARY_JSON = OUTPUT_DIR / "harmonization_pilot_summary.json"


# ============================================================
# HELPERS
# ============================================================

def load_threshold():
    with open(FINAL_CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    for key in (
        "selected_threshold",
        "threshold",
        "frozen_threshold",
        "decision_threshold",
    ):
        if key in config:
            return float(config[key])

    raise KeyError(
        f"No frozen threshold found in {FINAL_CONFIG}. "
        f"Keys: {list(config.keys())}"
    )


def normalize_ground_truth(value):
    text = str(value).strip().lower()

    if text in {
        "normal",
        "nondemented",
        "non_demented",
        "0",
    }:
        return "Normal", 0

    if text in {
        "early_demented",
        "early-demented",
        "verymilddemented",
        "milddemented",
        "very_mild_demented",
        "mild_demented",
        "1",
    }:
        return "Early_Demented", 1

    raise ValueError(f"Unsupported label: {value}")


def crop_to_target_occupancy(image, target_ratio):
    """
    Unsupervised spatial adapter:
    crops around the nonzero FreeSurfer brain and pads it into a square
    canvas sized so that the input foreground occupancy is approximately
    the INTERNAL-TRAIN median brain_ratio.

    No OASIS labels are used.
    """
    mask = image > 0
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError("No nonzero foreground in OASIS slice.")

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    crop = image[y1:y2, x1:x2]
    fg_pixels = int(np.count_nonzero(crop))

    if fg_pixels == 0:
        raise RuntimeError("Foreground crop is empty.")

    desired_side = int(
        np.ceil(
            np.sqrt(
                fg_pixels / max(float(target_ratio), 1e-8)
            )
        )
    )

    side = max(
        desired_side,
        crop.shape[0],
        crop.shape[1],
    )

    canvas = np.zeros(
        (side, side),
        dtype=np.uint8,
    )

    yoff = (side - crop.shape[0]) // 2
    xoff = (side - crop.shape[1]) // 2

    canvas[
        yoff:yoff + crop.shape[0],
        xoff:xoff + crop.shape[1]
    ] = crop

    return canvas


def match_processed_intensity_to_train(
    processed,
    brain_mask,
    target_p10,
    target_p90,
):
    """
    Unsupervised intensity adapter in PROCESSED image space.

    Maps current in-mask p10/p90 to the INTERNAL-TRAIN median p10/p90.
    No OASIS labels are used.
    """
    out = processed.copy().astype(np.float32)

    pixels = out[brain_mask > 0]

    if len(pixels) == 0:
        raise RuntimeError("No pixels inside brain mask.")

    src_p10 = float(np.percentile(pixels, 10))
    src_p90 = float(np.percentile(pixels, 90))

    denom = src_p90 - src_p10

    if abs(denom) < 1e-8:
        raise RuntimeError("Cannot intensity-match a zero-width p10/p90 range.")

    scale = (target_p90 - target_p10) / denom
    offset = target_p10 - scale * src_p10

    inside = brain_mask > 0
    out[inside] = out[inside] * scale + offset
    out[~inside] = 0

    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, {
        "source_p10": src_p10,
        "source_p90": src_p90,
        "target_p10": float(target_p10),
        "target_p90": float(target_p90),
        "intensity_scale": float(scale),
        "intensity_offset": float(offset),
    }


def run_variant(
    raw_image,
    variant,
    target_brain_ratio,
    target_p10,
    target_p90,
):
    if variant == "BASELINE":
        adapted_input = raw_image.copy()

    elif variant in {
        "SPATIAL_ONLY",
        "SPATIAL_PLUS_INTENSITY",
    }:
        adapted_input = crop_to_target_occupancy(
            raw_image,
            target_brain_ratio,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")

    standardized, processed = preprocess_from_raw(
        adapted_input
    )

    brain_mask, otsu, mask_status = create_brain_mask_v4(
        standardized
    )

    if brain_mask is None:
        raise RuntimeError(
            f"Brain Mask V4 failed: {mask_status}"
        )

    intensity_info = {}

    if variant == "SPATIAL_PLUS_INTENSITY":
        processed, intensity_info = (
            match_processed_intensity_to_train(
                processed,
                brain_mask,
                target_p10,
                target_p90,
            )
        )

    features = extract_features(
        processed,
        brain_mask,
    )

    if features is None:
        raise RuntimeError(
            "Feature extraction returned None."
        )

    vector = np.array(
        [
            float(features[name])
            for name in FROZEN_FEATURES
        ],
        dtype=np.float64,
    )

    if np.isnan(vector).any():
        raise RuntimeError(
            "Feature vector contains NaN."
        )

    if np.isinf(vector).any():
        raise RuntimeError(
            "Feature vector contains Inf."
        )

    return {
        "adapted_input": adapted_input,
        "standardized": standardized,
        "processed": processed,
        "brain_mask": brain_mask,
        "features": features,
        "otsu_threshold": float(otsu),
        "mask_status": mask_status,
        **intensity_info,
    }


def calculate_metrics(df):
    if len(df) == 0:
        return None

    y_true = df[
        "ground_truth_binary"
    ].astype(int).to_numpy()

    y_pred = df[
        "prediction_binary"
    ].astype(int).to_numpy()

    scores = df[
        "decision_score"
    ].astype(float).to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "n": int(len(df)),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "sensitivity": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
            if len(
                np.unique(
                    y_true
                )
            ) == 2
            else np.nan
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
            if len(
                np.unique(
                    y_true
                )
            ) == 2
            else np.nan
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return metrics


# ============================================================
# LOAD DATA / ARTIFACTS
# ============================================================

print("=" * 78)
print("OASIS HARMONIZATION PILOT")
print("=" * 78)

for required_path in (
    TRAIN_CSV,
    MANIFEST,
    EXTRACTION_SUMMARY,
    FINAL_MODEL,
    FINAL_CONFIG,
):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{required_path}"
        )

train = pd.read_csv(
    TRAIN_CSV
)

manifest = pd.read_csv(
    MANIFEST
)

extraction = pd.read_csv(
    EXTRACTION_SUMMARY
)

model = joblib.load(
    FINAL_MODEL
)

threshold = load_threshold()

missing_train = [
    feature
    for feature in FROZEN_FEATURES
    if feature not in train.columns
]

if missing_train:
    raise KeyError(
        "Training CSV is missing frozen features: "
        + ", ".join(
            missing_train
        )
    )

if "ground_truth" in manifest.columns:
    gt_col = "ground_truth"
elif "target_name" in manifest.columns:
    gt_col = "target_name"
else:
    raise KeyError(
        "Manifest must contain ground_truth or target_name."
    )

extraction = extraction.rename(
    columns={
        "status": "extraction_status",
        "output_png": "extracted_png",
    }
)

manifest["subject_id"] = (
    manifest["subject_id"]
    .astype(str)
    .str.strip()
)

extraction["subject_id"] = (
    extraction["subject_id"]
    .astype(str)
    .str.strip()
)

cases = manifest.merge(
    extraction,
    on="subject_id",
    how="left",
)

cases = cases[
    cases["extraction_status"]
    .astype(str)
    .str.upper()
    .eq("PASS")
].copy()

print("\nCases available:")
print(len(cases))

print("\nFrozen threshold:")
print(threshold)


# ============================================================
# INTERNAL-TRAIN TARGET STATISTICS
# ============================================================

train_stats = {}

for feature in FROZEN_FEATURES:
    vals = pd.to_numeric(
        train[feature],
        errors="coerce",
    ).dropna()

    train_stats[feature] = {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "median": float(vals.median()),
    }

target_brain_ratio = float(
    train["brain_ratio"].median()
)

target_p10 = float(
    train["p10"].median()
)

target_p90 = float(
    train["p90"].median()
)

print("\nInternal-train harmonization targets:")
print(
    f"brain_ratio median = "
    f"{target_brain_ratio:.6f}"
)
print(
    f"p10 median         = "
    f"{target_p10:.6f}"
)
print(
    f"p90 median         = "
    f"{target_p90:.6f}"
)


# ============================================================
# RUN THREE FIXED VARIANTS
# ============================================================

variants = [
    "BASELINE",
    "SPATIAL_ONLY",
    "SPATIAL_PLUS_INTENSITY",
]

records = []

for case_index, (_, row) in enumerate(
    cases.iterrows(),
    start=1,
):
    subject_id = str(
        row["subject_id"]
    ).strip()

    image_path = Path(
        str(
            row["extracted_png"]
        )
    )

    gt_name, gt_binary = normalize_ground_truth(
        row[gt_col]
    )

    cdr = (
        row["CDR"]
        if "CDR" in row.index
        else np.nan
    )

    print(
        f"\n[{case_index:02d}/{len(cases):02d}] "
        f"{subject_id} | CDR={cdr} | GT={gt_name}"
    )

    raw = cv2.imread(
        str(
            image_path
        ),
        cv2.IMREAD_GRAYSCALE,
    )

    if raw is None:
        for variant in variants:
            records.append({
                "subject_id": subject_id,
                "variant": variant,
                "status": "ERROR",
                "error": "Could not read extracted PNG",
            })
        print("Could not read PNG.")
        continue

    for variant in variants:
        print(
            "  ",
            variant,
            "...",
            end=" ",
        )

        try:
            result = run_variant(
                raw,
                variant,
                target_brain_ratio,
                target_p10,
                target_p90,
            )

            X = pd.DataFrame(
                [
                    {
                        name:
                            result["features"][name]
                        for name in FROZEN_FEATURES
                    }
                ],
                columns=FROZEN_FEATURES,
            )

            score = float(
                np.ravel(
                    model.decision_function(
                        X
                    )
                )[0]
            )

            pred_binary = int(
                score >= threshold
            )

            prediction = (
                "Early_Demented"
                if pred_binary == 1
                else "Normal"
            )

            correct = bool(
                pred_binary == gt_binary
            )

            variant_dir = (
                IMAGE_DIR
                / variant
                / subject_id
            )

            variant_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(
                    variant_dir
                    / "adapted_input.png"
                ),
                result[
                    "adapted_input"
                ],
            )

            cv2.imwrite(
                str(
                    variant_dir
                    / "processed.png"
                ),
                result[
                    "processed"
                ],
            )

            cv2.imwrite(
                str(
                    variant_dir
                    / "brain_mask.png"
                ),
                result[
                    "brain_mask"
                ],
            )

            record = {
                "subject_id":
                    subject_id,
                "CDR":
                    cdr,
                "ground_truth":
                    gt_name,
                "ground_truth_binary":
                    gt_binary,
                "variant":
                    variant,
                "status":
                    "PASS",
                "error":
                    "",
                "decision_score":
                    score,
                "threshold":
                    threshold,
                "margin":
                    score - threshold,
                "prediction":
                    prediction,
                "prediction_binary":
                    pred_binary,
                "correct":
                    correct,
                "mask_status":
                    result[
                        "mask_status"
                    ],
                "otsu_threshold":
                    result[
                        "otsu_threshold"
                    ],
            }

            for feature in FROZEN_FEATURES:
                record[
                    feature
                ] = float(
                    result[
                        "features"
                    ][
                        feature
                    ]
                )

            records.append(
                record
            )

            print(
                f"PASS | score={score:.4f} | "
                f"{prediction} | correct={correct}"
            )

        except Exception as e:
            records.append({
                "subject_id":
                    subject_id,
                "CDR":
                    cdr,
                "ground_truth":
                    gt_name,
                "ground_truth_binary":
                    gt_binary,
                "variant":
                    variant,
                "status":
                    "ERROR",
                "error":
                    str(e),
            })

            print(
                f"ERROR | {e}"
            )


results = pd.DataFrame(
    records
)

results.to_csv(
    RESULTS_CSV,
    index=False,
)


# ============================================================
# FEATURE-SHIFT ANALYSIS
# ============================================================

shift_rows = []

for variant in variants:
    subset = results[
        (results["variant"] == variant)
        &
        (results["status"] == "PASS")
    ].copy()

    if len(subset) == 0:
        continue

    abs_z_values = []

    for feature in FROZEN_FEATURES:
        oasis_mean = float(
            pd.to_numeric(
                subset[
                    feature
                ],
                errors="coerce",
            ).mean()
        )

        train_mean = train_stats[
            feature
        ][
            "mean"
        ]

        train_std = train_stats[
            feature
        ][
            "std"
        ]

        if train_std > 0:
            z_shift = (
                oasis_mean
                -
                train_mean
            ) / train_std
        else:
            z_shift = np.nan

        if np.isfinite(
            z_shift
        ):
            abs_z_values.append(
                abs(
                    z_shift
                )
            )

        shift_rows.append({
            "variant":
                variant,
            "feature":
                feature,
            "train_mean":
                train_mean,
            "oasis_mean":
                oasis_mean,
            "mean_z_shift":
                z_shift,
            "abs_mean_z_shift":
                (
                    abs(z_shift)
                    if np.isfinite(
                        z_shift
                    )
                    else np.nan
                ),
        })

    print(
        f"\n{variant} mean absolute "
        f"feature z-shift:",
        float(
            np.mean(
                abs_z_values
            )
        )
        if abs_z_values
        else np.nan,
    )


shift_df = pd.DataFrame(
    shift_rows
)

shift_df.to_csv(
    SHIFT_CSV,
    index=False,
)


# ============================================================
# PILOT METRICS
# ============================================================

summary = {
    "IMPORTANT":
        (
            "DEVELOPMENT/PILOT ONLY. "
            "These OASIS cases have already been inspected. "
            "Do not report post-harmonization performance "
            "as untouched external validation."
        ),

    "frozen_model_changed":
        False,

    "frozen_threshold_changed":
        False,

    "target_statistics_source":
        "INTERNAL TRAIN SET ONLY",

    "target_brain_ratio_median":
        target_brain_ratio,

    "target_p10_median":
        target_p10,

    "target_p90_median":
        target_p90,

    "variants":
        {},
}


print("\n" + "=" * 78)
print("HARMONIZATION PILOT SUMMARY")
print("=" * 78)

for variant in variants:
    subset = results[
        (results["variant"] == variant)
        &
        (results["status"] == "PASS")
    ].copy()

    metrics = calculate_metrics(
        subset
    )

    feature_subset = shift_df[
        shift_df["variant"]
        ==
        variant
    ]

    mean_abs_z_shift = (
        float(
            feature_subset[
                "abs_mean_z_shift"
            ].mean()
        )
        if len(
            feature_subset
        )
        else np.nan
    )

    summary[
        "variants"
    ][
        variant
    ] = {
        "successful_cases":
            int(
                len(
                    subset
                )
            ),

        "failed_cases":
            int(
                (
                    (
                        results[
                            "variant"
                        ]
                        ==
                        variant
                    )
                    &
                    (
                        results[
                            "status"
                        ]
                        !=
                        "PASS"
                    )
                ).sum()
            ),

        "mean_absolute_feature_z_shift":
            mean_abs_z_shift,

        "metrics":
            metrics,
    }

    print(
        f"\n{variant}"
    )

    print(
        "Successful cases:",
        len(
            subset
        )
    )

    print(
        "Mean |feature z-shift|:",
        f"{mean_abs_z_shift:.6f}"
    )

    if metrics is not None:
        print(
            "Accuracy:",
            f"{metrics['accuracy']:.6f}"
        )
        print(
            "Balanced accuracy:",
            f"{metrics['balanced_accuracy']:.6f}"
        )
        print(
            "Sensitivity:",
            f"{metrics['sensitivity']:.6f}"
        )
        print(
            "Specificity:",
            f"{metrics['specificity']:.6f}"
        )
        print(
            "ROC-AUC:",
            f"{metrics['roc_auc']:.6f}"
        )
        print(
            "TN/FP/FN/TP:",
            metrics["TN"],
            metrics["FP"],
            metrics["FN"],
            metrics["TP"],
        )


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=4,
    )


print("\n" + "=" * 78)
print("FILES SAVED")
print("=" * 78)

print(
    "\nPredictions:"
)
print(
    RESULTS_CSV
)

print(
    "\nFeature shifts:"
)
print(
    SHIFT_CSV
)

print(
    "\nSummary:"
)
print(
    SUMMARY_JSON
)

print(
    "\nAdapted/QC images:"
)
print(
    IMAGE_DIR
)

print("\nIMPORTANT:")
print(
    "This is a development harmonization pilot, "
    "NOT a new external-validation result."
)

print(
    "The frozen model and frozen threshold were NOT changed."
)

print("=" * 78)
