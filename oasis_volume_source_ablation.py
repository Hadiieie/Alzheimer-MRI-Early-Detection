from pathlib import Path
import json

import cv2
import joblib
import nibabel as nib
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

FREESURFER_ROOT = PROJECT_ROOT / "OASIS_FREESURFER"

MANIFEST = (
    PROJECT_ROOT
    / "NEW MRI"
    / "OASIS_BATCH"
    / "oasis_validation_manifest.csv"
)

TRAIN_CSV = (
    PROJECT_ROOT
    / "results"
    / "GROUP_AWARE_SPLIT"
    / "train.csv"
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
    / "OASIS_VOLUME_SOURCE_ABLATION"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "volume_source_predictions.csv"
SHIFT_CSV = OUTPUT_DIR / "volume_source_feature_shift.csv"
SUMMARY_JSON = OUTPUT_DIR / "volume_source_summary.json"


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


def find_subject_dir(subject_id):
    direct = FREESURFER_ROOT / subject_id
    if direct.exists():
        return direct

    matches = [
        p for p in FREESURFER_ROOT.rglob(subject_id)
        if p.is_dir() and p.name == subject_id
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        return None

    raise RuntimeError(
        f"Multiple subject directories found for {subject_id}: {matches}"
    )


def find_named_file(subject_dir, filename):
    # Supports both flattened extraction:
    #   subject/brain.mgz
    # and normal FreeSurfer:
    #   subject/mri/brain.mgz
    exact = subject_dir / filename
    if exact.exists():
        return exact

    mri_exact = subject_dir / "mri" / filename
    if mri_exact.exists():
        return mri_exact

    matches = list(subject_dir.rglob(filename))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        return None

    raise RuntimeError(
        f"Multiple {filename} files found in {subject_dir}: {matches}"
    )


def load_volume(path):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    orientation = nib.aff2axcodes(img.affine)

    if data.ndim != 3:
        raise RuntimeError(
            f"{path.name}: expected 3D volume, got {data.shape}"
        )

    if len(orientation) != 3 or orientation[1] not in ("I", "S"):
        raise RuntimeError(
            f"{path.name}: unexpected orientation {orientation}; "
            "second axis is not I/S."
        )

    return data, orientation


def occupied_mid_slice_index(mask_volume):
    areas = np.array([
        np.count_nonzero(mask_volume[:, i, :])
        for i in range(mask_volume.shape[1])
    ])

    nonempty = np.where(areas > 0)[0]

    if len(nonempty) == 0:
        raise RuntimeError("No nonzero brain voxels in reference mask.")

    first = int(nonempty.min())
    last = int(nonempty.max())

    index = int(
        round(
            first + 0.50 * (last - first)
        )
    )

    return index, first, last, int(areas[index])


def to_uint8_preserve_scale(sl):
    arr = np.asarray(sl)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError("Slice contains no finite values.")

    minimum = float(finite.min())
    maximum = float(finite.max())

    # FreeSurfer conformed anatomical volumes are normally stored in
    # an 8-bit-like intensity range. Preserve those values directly.
    if minimum >= 0 and maximum <= 255:
        return np.clip(arr, 0, 255).astype(np.uint8), {
            "conversion": "DIRECT_CLIP_0_255",
            "source_min": minimum,
            "source_max": maximum,
        }

    # If an unexpected range appears, do NOT silently min-max it.
    raise RuntimeError(
        f"Unexpected intensity range {minimum}..{maximum}; "
        "refusing to rescale automatically."
    )


def extract_source_slice(subject_dir, source_name):
    """
    Development ablation with fixed slice rule:
      BRAIN_MGZ  -> brain.mgz directly
      T1_MASKED  -> T1.mgz masked by brainmask.mgz
      NU_MASKED  -> nu.mgz masked by brainmask.mgz

    The 50%-of-occupied-brain-height slice index is defined by the
    brain support volume, never by class label or prediction.
    """

    brain_path = find_named_file(subject_dir, "brain.mgz")
    brainmask_path = find_named_file(subject_dir, "brainmask.mgz")

    if brain_path is None:
        raise FileNotFoundError("brain.mgz not found")

    brain, brain_orientation = load_volume(brain_path)

    if source_name == "BRAIN_MGZ":
        support = brain > 0
        source = brain
        source_path = brain_path
        mask_path = ""

    elif source_name in {"T1_MASKED", "NU_MASKED"}:
        if brainmask_path is None:
            raise FileNotFoundError("brainmask.mgz not found")

        brainmask, mask_orientation = load_volume(brainmask_path)

        if brainmask.shape != brain.shape:
            raise RuntimeError(
                f"brainmask shape {brainmask.shape} != brain shape {brain.shape}"
            )

        if mask_orientation != brain_orientation:
            raise RuntimeError(
                f"brainmask orientation {mask_orientation} "
                f"!= brain orientation {brain_orientation}"
            )

        wanted = "T1.mgz" if source_name == "T1_MASKED" else "nu.mgz"
        source_path = find_named_file(subject_dir, wanted)

        if source_path is None:
            raise FileNotFoundError(f"{wanted} not found")

        source, source_orientation = load_volume(source_path)

        if source.shape != brainmask.shape:
            raise RuntimeError(
                f"{wanted} shape {source.shape} "
                f"!= brainmask shape {brainmask.shape}"
            )

        if source_orientation != mask_orientation:
            raise RuntimeError(
                f"{wanted} orientation {source_orientation} "
                f"!= brainmask orientation {mask_orientation}"
            )

        support = brainmask > 0
        source = np.where(
            support,
            source,
            0,
        )
        mask_path = str(brainmask_path)

    else:
        raise ValueError(f"Unknown source: {source_name}")

    index, first, last, support_area = occupied_mid_slice_index(
        support
    )

    sl = np.rot90(
        source[:, index, :]
    )

    sl_uint8, conversion_info = to_uint8_preserve_scale(
        sl
    )

    return sl_uint8, {
        "source_file": str(source_path),
        "mask_file": mask_path,
        "orientation": "".join(brain_orientation),
        "selected_slice_index": index,
        "first_nonempty_axial": first,
        "last_nonempty_axial": last,
        "support_area": support_area,
        **conversion_info,
    }


def run_frozen_pipeline(raw_slice, model, threshold):
    standardized, processed = preprocess_from_raw(
        raw_slice
    )

    brain_mask, otsu, mask_status = create_brain_mask_v4(
        standardized
    )

    if brain_mask is None:
        raise RuntimeError(
            f"Brain Mask V4 failed: {mask_status}"
        )

    features = extract_features(
        processed,
        brain_mask,
    )

    vector = np.array(
        [
            float(features[name])
            for name in FROZEN_FEATURES
        ],
        dtype=np.float64,
    )

    if np.isnan(vector).any():
        raise RuntimeError("NaN in feature vector.")

    if np.isinf(vector).any():
        raise RuntimeError("Inf in feature vector.")

    X = pd.DataFrame(
        [vector],
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

    return {
        "standardized": standardized,
        "processed": processed,
        "brain_mask": brain_mask,
        "features": features,
        "decision_score": score,
        "prediction_binary": pred_binary,
        "prediction": prediction,
        "margin": score - threshold,
        "mask_status": mask_status,
        "otsu_threshold": float(otsu),
    }


def metrics_for(df):
    if len(df) == 0:
        return None

    y_true = df["ground_truth_binary"].astype(int).to_numpy()
    y_pred = df["prediction_binary"].astype(int).to_numpy()
    scores = df["decision_score"].astype(float).to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": int(len(df)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
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
            roc_auc_score(y_true, scores)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "pr_auc": float(
            average_precision_score(y_true, scores)
            if len(np.unique(y_true)) == 2
            else np.nan
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("OASIS FREESURFER VOLUME-SOURCE ABLATION")
print("=" * 80)

for path in (
    FREESURFER_ROOT,
    MANIFEST,
    TRAIN_CSV,
    FINAL_MODEL,
    FINAL_CONFIG,
):
    if not path.exists():
        raise FileNotFoundError(f"Required path missing:\n{path}")

manifest = pd.read_csv(MANIFEST)
train = pd.read_csv(TRAIN_CSV)
model = joblib.load(FINAL_MODEL)
threshold = load_threshold()

if "ground_truth" in manifest.columns:
    gt_col = "ground_truth"
elif "target_name" in manifest.columns:
    gt_col = "target_name"
else:
    raise KeyError(
        "Manifest must contain ground_truth or target_name."
    )

manifest["subject_id"] = (
    manifest["subject_id"]
    .astype(str)
    .str.strip()
)

missing_train = [
    f for f in FROZEN_FEATURES
    if f not in train.columns
]

if missing_train:
    raise KeyError(
        "Training CSV missing features: "
        + ", ".join(missing_train)
    )


# ============================================================
# INTERNAL TRAIN FEATURE STATISTICS
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
    }


# ============================================================
# RUN ABLATION
# ============================================================

sources = [
    "BRAIN_MGZ",
    "T1_MASKED",
    "NU_MASKED",
]

records = []

for case_index, (_, row) in enumerate(
    manifest.iterrows(),
    start=1,
):
    subject_id = str(row["subject_id"]).strip()
    subject_dir = find_subject_dir(subject_id)

    gt_name, gt_binary = normalize_ground_truth(
        row[gt_col]
    )

    cdr = (
        row["CDR"]
        if "CDR" in row.index
        else np.nan
    )

    print(
        f"\n[{case_index:02d}/{len(manifest):02d}] "
        f"{subject_id} | CDR={cdr} | GT={gt_name}"
    )

    if subject_dir is None:
        print("  SUBJECT MISSING")
        for source in sources:
            records.append({
                "subject_id": subject_id,
                "CDR": cdr,
                "ground_truth": gt_name,
                "ground_truth_binary": gt_binary,
                "source": source,
                "status": "MISSING",
                "error": "Subject directory not found",
            })
        continue

    for source in sources:
        print(f"  {source} ...", end=" ")

        try:
            raw_slice, source_info = extract_source_slice(
                subject_dir,
                source,
            )

            result = run_frozen_pipeline(
                raw_slice,
                model,
                threshold,
            )

            correct = bool(
                result["prediction_binary"]
                ==
                gt_binary
            )

            source_dir = (
                OUTPUT_DIR
                / "images"
                / source
                / subject_id
            )

            source_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(source_dir / "raw_slice.png"),
                raw_slice,
            )
            cv2.imwrite(
                str(source_dir / "processed.png"),
                result["processed"],
            )
            cv2.imwrite(
                str(source_dir / "brain_mask.png"),
                result["brain_mask"],
            )

            record = {
                "subject_id": subject_id,
                "CDR": cdr,
                "ground_truth": gt_name,
                "ground_truth_binary": gt_binary,
                "source": source,
                "status": "PASS",
                "error": "",
                "decision_score": result["decision_score"],
                "threshold": threshold,
                "margin": result["margin"],
                "prediction": result["prediction"],
                "prediction_binary": result["prediction_binary"],
                "correct": correct,
                "mask_status": result["mask_status"],
                "otsu_threshold": result["otsu_threshold"],
                **source_info,
            }

            for feature in FROZEN_FEATURES:
                record[feature] = float(
                    result["features"][feature]
                )

            records.append(record)

            print(
                f"PASS | score={result['decision_score']:.4f} | "
                f"{result['prediction']} | correct={correct}"
            )

        except Exception as e:
            records.append({
                "subject_id": subject_id,
                "CDR": cdr,
                "ground_truth": gt_name,
                "ground_truth_binary": gt_binary,
                "source": source,
                "status": "ERROR",
                "error": str(e),
            })

            print(f"ERROR | {e}")


results = pd.DataFrame(records)
results.to_csv(RESULTS_CSV, index=False)


# ============================================================
# FEATURE SHIFT
# ============================================================

shift_rows = []

for source in sources:
    subset = results[
        (results["source"] == source)
        &
        (results["status"] == "PASS")
    ]

    if len(subset) == 0:
        continue

    for feature in FROZEN_FEATURES:
        oasis_mean = float(
            pd.to_numeric(
                subset[feature],
                errors="coerce",
            ).mean()
        )

        train_mean = train_stats[feature]["mean"]
        train_std = train_stats[feature]["std"]

        z_shift = (
            (oasis_mean - train_mean) / train_std
            if train_std > 0
            else np.nan
        )

        shift_rows.append({
            "source": source,
            "feature": feature,
            "train_mean": train_mean,
            "oasis_mean": oasis_mean,
            "mean_z_shift": z_shift,
            "abs_mean_z_shift":
                abs(z_shift)
                if np.isfinite(z_shift)
                else np.nan,
        })

shift_df = pd.DataFrame(shift_rows)
shift_df.to_csv(SHIFT_CSV, index=False)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "IMPORTANT":
        (
            "DEVELOPMENT ABLATION ONLY. "
            "The OASIS subjects have already been inspected. "
            "Do not report the best source from this analysis as "
            "untouched external validation."
        ),
    "frozen_model_changed": False,
    "frozen_threshold_changed": False,
    "slice_rule": "50% of occupied brain height",
    "sources": {},
}

print("\n" + "=" * 80)
print("VOLUME-SOURCE ABLATION SUMMARY")
print("=" * 80)

for source in sources:
    passed = results[
        (results["source"] == source)
        &
        (results["status"] == "PASS")
    ].copy()

    metrics = metrics_for(passed)

    source_shift = shift_df[
        shift_df["source"] == source
    ]

    mean_abs_shift = (
        float(
            source_shift[
                "abs_mean_z_shift"
            ].mean()
        )
        if len(source_shift)
        else np.nan
    )

    error_count = int(
        (
            (results["source"] == source)
            &
            (results["status"] == "ERROR")
        ).sum()
    )

    missing_count = int(
        (
            (results["source"] == source)
            &
            (results["status"] == "MISSING")
        ).sum()
    )

    summary["sources"][source] = {
        "successful_cases": int(len(passed)),
        "errors": error_count,
        "missing": missing_count,
        "mean_absolute_feature_z_shift": mean_abs_shift,
        "metrics": metrics,
    }

    print(f"\n{source}")
    print("Successful cases:", len(passed))
    print("Errors:", error_count)
    print("Missing:", missing_count)
    print(
        "Mean |feature z-shift|:",
        f"{mean_abs_shift:.6f}"
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
            "PR-AUC:",
            f"{metrics['pr_auc']:.6f}"
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

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)
print("\nPredictions:")
print(RESULTS_CSV)
print("\nFeature shifts:")
print(SHIFT_CSV)
print("\nSummary:")
print(SUMMARY_JSON)
print("\nNo model, threshold, Brain Mask V4, or feature definition was changed.")
print("=" * 80)
