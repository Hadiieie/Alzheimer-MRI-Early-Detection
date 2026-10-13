from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OASIS_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "OASIS_FREESURFER_EVALUATION"
    / "oasis_freesurfer_predictions.csv"
)

INTERNAL_TEST = (
    PROJECT_ROOT
    / "results"
    / "GROUP_AWARE_SPLIT"
    / "test.csv"
)

FINAL_MODEL = (
    PROJECT_ROOT
    / "results"
    / "FINAL_MODEL"
    / "final_model.joblib"
)

FINAL_FEATURES = (
    PROJECT_ROOT
    / "results"
    / "FINAL_MODEL"
    / "final_features.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "OASIS_DOMAIN_SHIFT"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SHIFT_CSV = OUTPUT_DIR / "feature_shift_summary.csv"
SCORE_SUMMARY_CSV = OUTPUT_DIR / "score_distribution_summary.csv"
ERROR_CASES_CSV = OUTPUT_DIR / "oasis_inference_errors.csv"
REPORT_JSON = OUTPUT_DIR / "domain_shift_report.json"


# ============================================================
# LOAD FROZEN FEATURE ORDER
# ============================================================

with open(FINAL_FEATURES, "r", encoding="utf-8") as f:
    features = [line.strip() for line in f if line.strip()]

if len(features) != 30:
    raise RuntimeError(f"Expected 30 frozen features, found {len(features)}")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 76)
print("OASIS EXTERNAL DOMAIN-SHIFT ANALYSIS")
print("=" * 76)

oasis = pd.read_csv(OASIS_RESULTS)
internal = pd.read_csv(INTERNAL_TEST)
model = joblib.load(FINAL_MODEL)

print("\nOASIS rows:", len(oasis))
print("Internal test rows:", len(internal))

# ============================================================
# 1. IDENTIFY OASIS ERROR CASE(S)
# ============================================================

errors = oasis[oasis["status"].astype(str).str.upper() != "PASS"].copy()
errors.to_csv(ERROR_CASES_CSV, index=False)

print("\n" + "=" * 76)
print("1. OASIS INFERENCE ERRORS")
print("=" * 76)

print("\nError count:")
print(len(errors))

if len(errors) > 0:
    cols = [
        c for c in [
            "subject_id",
            "CDR",
            "ground_truth",
            "image_path",
            "error",
        ]
        if c in errors.columns
    ]
    print(errors[cols].to_string(index=False))
else:
    print("No inference errors.")


# ============================================================
# 2. RECOMPUTE INTERNAL TEST DECISION SCORES
# ============================================================

missing_internal = [f for f in features if f not in internal.columns]
if missing_internal:
    raise KeyError(
        "Internal test.csv is missing frozen features: "
        + ", ".join(missing_internal)
    )

X_internal = internal[features].copy()

internal_scores = np.ravel(
    model.decision_function(X_internal)
).astype(float)

# Ground-truth target column
if "target_name" in internal.columns:
    internal_gt_name = internal["target_name"].astype(str)
elif "binary_target" in internal.columns:
    internal_gt_name = internal["binary_target"].astype(str)
elif "target" in internal.columns:
    internal_gt_name = internal["target"].astype(str)
else:
    # Fall back to original_class mapping used in the project
    if "original_class" not in internal.columns:
        raise KeyError(
            "Could not identify internal test target column."
        )

    internal_gt_name = internal["original_class"].map({
        "NonDemented": "Normal",
        "VeryMildDemented": "Early_Demented",
        "MildDemented": "Early_Demented",
    })

internal_score_df = pd.DataFrame({
    "dataset": "Internal_Test",
    "ground_truth": internal_gt_name,
    "decision_score": internal_scores,
})

oasis_pass = oasis[
    oasis["status"].astype(str).str.upper() == "PASS"
].copy()

oasis_score_df = pd.DataFrame({
    "dataset": "OASIS_FreeSurfer",
    "ground_truth": oasis_pass["ground_truth"].astype(str),
    "decision_score": oasis_pass["decision_score"].astype(float),
})

all_scores = pd.concat(
    [internal_score_df, oasis_score_df],
    ignore_index=True
)


# ============================================================
# 3. SCORE DISTRIBUTION SUMMARY
# ============================================================

score_summary = (
    all_scores
    .groupby(["dataset", "ground_truth"], dropna=False)
    ["decision_score"]
    .agg(
        count="count",
        min="min",
        mean="mean",
        median="median",
        std="std",
        max="max",
    )
    .reset_index()
)

score_summary.to_csv(
    SCORE_SUMMARY_CSV,
    index=False
)

print("\n" + "=" * 76)
print("2. DECISION-SCORE DISTRIBUTIONS")
print("=" * 76)
print(score_summary.to_string(index=False))

# Explicit OASIS class separation
oasis_normal = oasis_pass.loc[
    oasis_pass["ground_truth"] == "Normal",
    "decision_score"
].astype(float)

oasis_early = oasis_pass.loc[
    oasis_pass["ground_truth"] == "Early_Demented",
    "decision_score"
].astype(float)

print("\nOASIS score shift:")
if len(oasis_normal) > 0:
    print(
        "Normal mean / median:",
        float(oasis_normal.mean()),
        "/",
        float(oasis_normal.median())
    )
if len(oasis_early) > 0:
    print(
        "Early_Demented mean / median:",
        float(oasis_early.mean()),
        "/",
        float(oasis_early.median())
    )


# ============================================================
# 4. COLLECT OASIS RAW FEATURE VECTORS
# ============================================================

oasis_feature_rows = []

for _, row in oasis_pass.iterrows():
    image_path = Path(str(row["image_path"]))
    stem = image_path.stem

    feature_file = (
        PROJECT_ROOT
        / "results"
        / "SINGLE_MRI_INFERENCE"
        / stem
        / "features.csv"
    )

    if not feature_file.exists():
        continue

    fdf = pd.read_csv(feature_file)
    if len(fdf) == 0:
        continue

    feature_row = {
        "subject_id": row["subject_id"],
        "ground_truth": row["ground_truth"],
    }

    for feature in features:
        if feature not in fdf.columns:
            raise KeyError(
                f"{feature} missing from {feature_file}"
            )
        feature_row[feature] = float(fdf.iloc[0][feature])

    oasis_feature_rows.append(feature_row)

oasis_features = pd.DataFrame(oasis_feature_rows)

print("\nOASIS feature vectors located:")
print(len(oasis_features))


# ============================================================
# 5. FEATURE DOMAIN SHIFT
# Compare OASIS vs internal test in raw feature space.
# z_shift = (OASIS mean - Internal mean) / Internal std
# ============================================================

shift_rows = []

if len(oasis_features) > 0:
    for feature in features:
        internal_values = pd.to_numeric(
            internal[feature],
            errors="coerce"
        ).dropna()

        oasis_values = pd.to_numeric(
            oasis_features[feature],
            errors="coerce"
        ).dropna()

        internal_mean = float(internal_values.mean())
        internal_std = float(internal_values.std(ddof=0))
        internal_median = float(internal_values.median())

        oasis_mean = float(oasis_values.mean())
        oasis_median = float(oasis_values.median())

        if internal_std > 0:
            z_shift = (
                oasis_mean - internal_mean
            ) / internal_std
        else:
            z_shift = np.nan

        # Robust median shift scaled by internal IQR
        q1 = float(internal_values.quantile(0.25))
        q3 = float(internal_values.quantile(0.75))
        iqr = q3 - q1

        if iqr > 0:
            robust_shift = (
                oasis_median - internal_median
            ) / iqr
        else:
            robust_shift = np.nan

        shift_rows.append({
            "feature": feature,
            "internal_mean": internal_mean,
            "internal_std": internal_std,
            "internal_median": internal_median,
            "oasis_mean": oasis_mean,
            "oasis_median": oasis_median,
            "mean_z_shift": z_shift,
            "abs_mean_z_shift": abs(z_shift) if np.isfinite(z_shift) else np.nan,
            "robust_median_shift_iqr_units": robust_shift,
            "abs_robust_median_shift_iqr_units":
                abs(robust_shift) if np.isfinite(robust_shift) else np.nan,
        })

    feature_shift = pd.DataFrame(shift_rows).sort_values(
        "abs_mean_z_shift",
        ascending=False
    )

    feature_shift.to_csv(
        FEATURE_SHIFT_CSV,
        index=False
    )

    print("\n" + "=" * 76)
    print("3. TOP RAW-FEATURE DOMAIN SHIFTS")
    print("=" * 76)

    print(
        feature_shift[
            [
                "feature",
                "internal_mean",
                "oasis_mean",
                "mean_z_shift",
                "robust_median_shift_iqr_units",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

else:
    feature_shift = pd.DataFrame()
    print("\nNo OASIS features.csv files were found.")


# ============================================================
# 6. SAVE SEPARATE PLOTS
# ============================================================

# Plot 1: Internal decision-score distribution
plt.figure(figsize=(8, 5))
for label in ["Normal", "Early_Demented"]:
    vals = internal_score_df.loc[
        internal_score_df["ground_truth"] == label,
        "decision_score"
    ]
    if len(vals) > 0:
        plt.hist(vals, bins=40, alpha=0.5, label=label)

plt.axvline(0.10872472921801979, linestyle="--", label="Frozen threshold")
plt.xlabel("Decision score")
plt.ylabel("Count")
plt.title("Internal Test Decision-Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "internal_score_distribution.png",
    dpi=170,
    bbox_inches="tight"
)
plt.close()

# Plot 2: OASIS decision-score distribution
plt.figure(figsize=(8, 5))
for label in ["Normal", "Early_Demented"]:
    vals = oasis_score_df.loc[
        oasis_score_df["ground_truth"] == label,
        "decision_score"
    ]
    if len(vals) > 0:
        plt.hist(vals, bins=12, alpha=0.5, label=label)

plt.axvline(0.10872472921801979, linestyle="--", label="Frozen threshold")
plt.xlabel("Decision score")
plt.ylabel("Count")
plt.title("OASIS FreeSurfer Decision-Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "oasis_score_distribution.png",
    dpi=170,
    bbox_inches="tight"
)
plt.close()

# Plot 3: top feature shifts
if len(feature_shift) > 0:
    top = feature_shift.head(15).sort_values(
        "abs_mean_z_shift",
        ascending=True
    )

    plt.figure(figsize=(9, 6))
    plt.barh(
        top["feature"],
        top["mean_z_shift"]
    )
    plt.xlabel("OASIS mean shift (internal-test SD units)")
    plt.ylabel("Feature")
    plt.title("Top OASIS vs Internal Raw-Feature Shifts")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "top_feature_domain_shifts.png",
        dpi=170,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 7. FINAL JSON REPORT
# ============================================================

report = {
    "oasis_total_rows":
        int(len(oasis)),

    "oasis_successful_inference":
        int(len(oasis_pass)),

    "oasis_inference_errors":
        int(len(errors)),

    "oasis_normal_score_mean":
        float(oasis_normal.mean())
        if len(oasis_normal) else None,

    "oasis_normal_score_median":
        float(oasis_normal.median())
        if len(oasis_normal) else None,

    "oasis_early_score_mean":
        float(oasis_early.mean())
        if len(oasis_early) else None,

    "oasis_early_score_median":
        float(oasis_early.median())
        if len(oasis_early) else None,

    "top_shifted_features":
        (
            feature_shift[
                [
                    "feature",
                    "mean_z_shift",
                    "robust_median_shift_iqr_units",
                ]
            ]
            .head(10)
            .to_dict(orient="records")
            if len(feature_shift) > 0
            else []
        ),

    "interpretation_note":
        (
            "Descriptive domain-shift analysis only. "
            "The frozen model and threshold are not changed."
        ),
}

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 76)
print("DOMAIN-SHIFT ANALYSIS COMPLETE")
print("=" * 76)

print("\nSaved:")
print(SCORE_SUMMARY_CSV)
print(FEATURE_SHIFT_CSV)
print(ERROR_CASES_CSV)
print(REPORT_JSON)

print("\nPlots:")
print(OUTPUT_DIR / "internal_score_distribution.png")
print(OUTPUT_DIR / "oasis_score_distribution.png")
print(OUTPUT_DIR / "top_feature_domain_shifts.png")

print("\nIMPORTANT:")
print("No model, threshold, preprocessing, Brain Mask V4, or feature set was modified.")
print("=" * 76)
