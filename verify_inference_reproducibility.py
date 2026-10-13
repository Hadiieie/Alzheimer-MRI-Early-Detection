from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from inference_core import FROZEN_FEATURES, extract_single_mri_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

REFERENCE_FEATURE_FILE = (
    RESULTS_DIR / "FULL_FEATURE_EXTRACTION" / "full_features_39538.csv"
)
FINAL_FEATURE_LIST = RESULTS_DIR / "FINAL_MODEL" / "final_features.txt"

OUTPUT_DIR = RESULTS_DIR / "INFERENCE_REPRODUCIBILITY"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "reproducibility_check.csv"

N_CASES = 12
ABS_TOL = 1e-8
REL_TOL = 1e-8

print("\n" + "=" * 70)
print("INFERENCE REPRODUCIBILITY CHECK")
print("=" * 70)

if not REFERENCE_FEATURE_FILE.exists():
    raise FileNotFoundError(f"Reference feature file not found:\n{REFERENCE_FEATURE_FILE}")
if not FINAL_FEATURE_LIST.exists():
    raise FileNotFoundError(f"Frozen feature list not found:\n{FINAL_FEATURE_LIST}")

df = pd.read_csv(REFERENCE_FEATURE_FILE)
print("\nReference rows:")
print(len(df))

with open(FINAL_FEATURE_LIST, "r", encoding="utf-8") as f:
    frozen_from_file = [line.strip() for line in f if line.strip()]

print("\nFrozen feature count:")
print(len(frozen_from_file))

if frozen_from_file != FROZEN_FEATURES:
    print("\nFrom final_features.txt:")
    print(frozen_from_file)
    print("\nInference-core order:")
    print(FROZEN_FEATURES)
    raise RuntimeError("Frozen feature order mismatch.")

print("Feature order: PASS")

required_columns = ["filename", "path", "processed_path"] + FROZEN_FEATURES
missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise KeyError("Reference CSV is missing: " + ", ".join(missing_columns))

n_cases = min(N_CASES, len(df))
indices = np.linspace(0, len(df) - 1, num=n_cases, dtype=int)
test_df = df.iloc[indices].copy()

records = []
all_pass = True

print("\n" + "=" * 70)
print("REPRODUCING KNOWN DATASET CASES")
print("=" * 70)

for case_number, (_, row) in enumerate(test_df.iterrows(), start=1):
    filename = row["filename"]
    raw_path = Path(row["path"])
    processed_path = Path(row["processed_path"])

    print(f"\n[{case_number:02d}/{n_cases:02d}] {filename}")

    try:
        result = extract_single_mri_features(raw_path)

        stored_processed = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
        if stored_processed is None:
            raise RuntimeError(f"Could not read stored processed image: {processed_path}")

        if stored_processed.shape != result["processed_image"].shape:
            raise RuntimeError("Processed image shape mismatch.")

        pixel_diff = np.abs(
            stored_processed.astype(np.int16) -
            result["processed_image"].astype(np.int16)
        )
        max_pixel_diff = int(pixel_diff.max())
        processed_exact = bool(max_pixel_diff == 0)

        reference_values = np.array(
            [float(row[name]) for name in FROZEN_FEATURES],
            dtype=np.float64
        )
        reproduced_values = np.array(
            [float(result["features"][name]) for name in FROZEN_FEATURES],
            dtype=np.float64
        )

        feature_diffs = np.abs(reference_values - reproduced_values)
        max_feature_diff = float(feature_diffs.max())
        feature_match = bool(
            np.allclose(
                reference_values,
                reproduced_values,
                atol=ABS_TOL,
                rtol=REL_TOL
            )
        )

        case_pass = processed_exact and feature_match
        if not case_pass:
            all_pass = False

        worst_index = int(np.argmax(feature_diffs))
        worst_feature = FROZEN_FEATURES[worst_index]

        print("Processed pixels exact match:", processed_exact)
        print("Max processed pixel difference:", max_pixel_diff)
        print("30-feature vector match:", feature_match)
        print("Maximum feature absolute difference:", f"{max_feature_diff:.12g}")
        print("Worst feature:", worst_feature)
        print("CASE STATUS:", "PASS" if case_pass else "FAIL")

        records.append({
            "filename": filename,
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "processed_exact_match": processed_exact,
            "max_processed_pixel_diff": max_pixel_diff,
            "feature_match": feature_match,
            "max_feature_abs_diff": max_feature_diff,
            "worst_feature": worst_feature,
            "mask_status": result["mask_status"],
            "otsu_threshold": result["otsu_threshold"],
            "status": "PASS" if case_pass else "FAIL",
        })

    except Exception as e:
        all_pass = False
        print("CASE STATUS: ERROR")
        print("Reason:", e)
        records.append({
            "filename": filename,
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "processed_exact_match": False,
            "max_processed_pixel_diff": np.nan,
            "feature_match": False,
            "max_feature_abs_diff": np.nan,
            "worst_feature": "",
            "mask_status": "ERROR",
            "otsu_threshold": np.nan,
            "status": "ERROR",
            "error": str(e),
        })

report_df = pd.DataFrame(records)
report_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("REPRODUCIBILITY SUMMARY")
print("=" * 70)
print("\nCases checked:")
print(len(report_df))
print("\nStatus:")
print(report_df["status"].value_counts(dropna=False))

passed = report_df[report_df["status"] == "PASS"]
if len(passed) > 0:
    print("\nLargest feature absolute difference among PASS cases:")
    print(passed["max_feature_abs_diff"].max())

print("\nReport saved to:")
print(OUTPUT_FILE)
print("\n" + "=" * 70)

if all_pass:
    print("INFERENCE REPRODUCIBILITY: PASS")
    print("\nThe new single-image inference core reproduces the frozen training pipeline.")
    print("\nNEXT STEP:")
    print("Build predict_mri.py using the frozen model and frozen threshold.")
else:
    print("INFERENCE REPRODUCIBILITY: FAIL")
    print("\nDo NOT move to new-image inference yet.")
    print("Inspect the failed case(s) in the CSV report.")

print("=" * 70)
