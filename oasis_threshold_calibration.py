from pathlib import Path
import json

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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "OASIS_VOLUME_SOURCE_ABLATION"
    / "volume_source_predictions.csv"
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
    / "OASIS_THRESHOLD_CALIBRATION"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SWEEP_CSV = OUTPUT_DIR / "threshold_sweep.csv"

CALIBRATION_CASES_CSV = (
    OUTPUT_DIR
    / "oasis_calibration_cases.csv"
)

CANDIDATE_JSON = (
    OUTPUT_DIR
    / "oasis_v2_candidate_threshold.json"
)

REPORT_TXT = (
    OUTPUT_DIR
    / "threshold_calibration_report.txt"
)


# ============================================================
# LOAD ORIGINAL FROZEN THRESHOLD
# ============================================================

def load_original_threshold():
    with open(
        FINAL_CONFIG,
        "r",
        encoding="utf-8",
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
            return float(config[key])

    raise KeyError(
        "Could not find original frozen threshold in "
        f"{FINAL_CONFIG}\n"
        f"Available keys: {list(config.keys())}"
    )


ORIGINAL_THRESHOLD = load_original_threshold()


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    scores,
    threshold,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    y_pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    youden_j = (
        sensitivity
        +
        specificity
        -
        1.0
    )

    sens_spec_gap = abs(
        sensitivity
        -
        specificity
    )

    return {
        "threshold":
            float(threshold),

        "accuracy":
            float(accuracy),

        "balanced_accuracy":
            float(balanced_accuracy),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "precision":
            float(precision),

        "f1":
            float(f1),

        "youden_j":
            float(youden_j),

        "sensitivity_specificity_gap":
            float(sens_spec_gap),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),
    }


# ============================================================
# LOAD DEVELOPMENT / CALIBRATION DATA
# ============================================================

print()
print("=" * 76)
print("OASIS V2 THRESHOLD CALIBRATION")
print("=" * 76)

print()
print("IMPORTANT:")
print("DEVELOPMENT / CALIBRATION SET ONLY")
print("THIS IS NOT FINAL EXTERNAL VALIDATION")
print()

print("Original V1 model:")
print("UNCHANGED / FROZEN")

print()
print("Original V1 threshold:")
print(f"{ORIGINAL_THRESHOLD:.12f}")

print()
print("TEST SET:")
print("NOT LOADED")

print()
print("Calibration source:")
print("OASIS FreeSurfer BRAIN_MGZ only")


if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)


required_columns = {
    "subject_id",
    "source",
    "status",
    "ground_truth",
    "ground_truth_binary",
    "decision_score",
}

missing = (
    required_columns
    -
    set(df.columns)
)

if missing:
    raise KeyError(
        "Missing required columns:\n"
        + "\n".join(
            sorted(missing)
        )
    )


# ============================================================
# SELECT ONLY CLEAN BRAIN_MGZ DEVELOPMENT CASES
# ============================================================

calibration = df[
    (
        df["source"]
        .astype(str)
        .str.upper()
        ==
        "BRAIN_MGZ"
    )
    &
    (
        df["status"]
        .astype(str)
        .str.upper()
        ==
        "PASS"
    )
].copy()


if len(calibration) == 0:
    raise RuntimeError(
        "No successful BRAIN_MGZ cases found."
    )


calibration[
    "ground_truth_binary"
] = pd.to_numeric(
    calibration[
        "ground_truth_binary"
    ],
    errors="raise",
).astype(int)


calibration[
    "decision_score"
] = pd.to_numeric(
    calibration[
        "decision_score"
    ],
    errors="raise",
).astype(float)


calibration.to_csv(
    CALIBRATION_CASES_CSV,
    index=False,
)


print()
print("=" * 76)
print("CALIBRATION DATASET")
print("=" * 76)

print()
print("Cases:")
print(len(calibration))

print()
print("Ground-truth distribution:")
print(
    calibration[
        "ground_truth"
    ].value_counts()
)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

normal_scores = calibration.loc[
    calibration[
        "ground_truth_binary"
    ]
    ==
    0,
    "decision_score",
].to_numpy()


early_scores = calibration.loc[
    calibration[
        "ground_truth_binary"
    ]
    ==
    1,
    "decision_score",
].to_numpy()


print()
print("=" * 76)
print("DECISION SCORE SUMMARY")
print("=" * 76)

print()
print("NORMAL")

print(
    "Count:",
    len(normal_scores),
)

print(
    "Min:",
    float(
        np.min(
            normal_scores
        )
    ),
)

print(
    "Mean:",
    float(
        np.mean(
            normal_scores
        )
    ),
)

print(
    "Median:",
    float(
        np.median(
            normal_scores
        )
    ),
)

print(
    "Max:",
    float(
        np.max(
            normal_scores
        )
    ),
)


print()
print("EARLY_DEMENTED")

print(
    "Count:",
    len(early_scores),
)

print(
    "Min:",
    float(
        np.min(
            early_scores
        )
    ),
)

print(
    "Mean:",
    float(
        np.mean(
            early_scores
        )
    ),
)

print(
    "Median:",
    float(
        np.median(
            early_scores
        )
    ),
)

print(
    "Max:",
    float(
        np.max(
            early_scores
        )
    ),
)


# ============================================================
# THRESHOLD-INDEPENDENT METRICS
# ============================================================

y_true = calibration[
    "ground_truth_binary"
].to_numpy(
    dtype=int
)

scores = calibration[
    "decision_score"
].to_numpy(
    dtype=float
)


roc_auc = roc_auc_score(
    y_true,
    scores,
)

pr_auc = average_precision_score(
    y_true,
    scores,
)


print()
print("=" * 76)
print("THRESHOLD-INDEPENDENT METRICS")
print("=" * 76)

print()
print(
    "ROC-AUC:",
    f"{roc_auc:.6f}",
)

print(
    "PR-AUC:",
    f"{pr_auc:.6f}",
)


# ============================================================
# ORIGINAL V1 THRESHOLD PERFORMANCE
# ============================================================

original_metrics = calculate_metrics(
    y_true,
    scores,
    ORIGINAL_THRESHOLD,
)


print()
print("=" * 76)
print("ORIGINAL V1 THRESHOLD ON OASIS DEVELOPMENT SET")
print("=" * 76)

for key, value in original_metrics.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# CREATE ALL NON-REDUNDANT THRESHOLD CANDIDATES
# ============================================================

unique_scores = np.sort(
    np.unique(
        scores
    )
)


threshold_candidates = []


# Threshold below all scores
epsilon = 1e-9

threshold_candidates.append(
    float(
        unique_scores[0]
        -
        epsilon
    )
)


# Midpoints between every adjacent score
for left, right in zip(
    unique_scores[:-1],
    unique_scores[1:],
):
    midpoint = (
        left
        +
        right
    ) / 2.0

    threshold_candidates.append(
        float(
            midpoint
        )
    )


# Threshold above all scores
threshold_candidates.append(
    float(
        unique_scores[-1]
        +
        epsilon
    )
)


# Also include original V1 threshold
threshold_candidates.append(
    float(
        ORIGINAL_THRESHOLD
    )
)


threshold_candidates = np.array(
    sorted(
        set(
            threshold_candidates
        )
    ),
    dtype=float,
)


print()
print("=" * 76)
print("THRESHOLD SWEEP")
print("=" * 76)

print()
print(
    "Candidates:",
    len(
        threshold_candidates
    )
)


# ============================================================
# RUN SWEEP
# ============================================================

rows = []

for threshold in threshold_candidates:

    metrics = calculate_metrics(
        y_true,
        scores,
        threshold,
    )

    metrics[
        "is_original_v1_threshold"
    ] = bool(
        np.isclose(
            threshold,
            ORIGINAL_THRESHOLD,
            rtol=0,
            atol=1e-12,
        )
    )

    rows.append(
        metrics
    )


sweep = pd.DataFrame(
    rows
)


# ============================================================
# FIND OPTIMAL BALANCED-ACCURACY REGION
# ============================================================

best_balanced_accuracy = float(
    sweep[
        "balanced_accuracy"
    ].max()
)


best_rows = sweep[
    np.isclose(
        sweep[
            "balanced_accuracy"
        ],
        best_balanced_accuracy,
        atol=1e-12,
        rtol=0,
    )
].copy()


# Balanced accuracy and Youden J are mathematically related.
# If several thresholds tie, prefer the one where sensitivity and
# specificity are most balanced.
best_gap = float(
    best_rows[
        "sensitivity_specificity_gap"
    ].min()
)


best_rows = best_rows[
    np.isclose(
        best_rows[
            "sensitivity_specificity_gap"
        ],
        best_gap,
        atol=1e-12,
        rtol=0,
    )
].copy()


# If multiple equivalent thresholds remain, use their median.
candidate_threshold = float(
    np.median(
        best_rows[
            "threshold"
        ].to_numpy(
            dtype=float
        )
    )
)


candidate_metrics = calculate_metrics(
    y_true,
    scores,
    candidate_threshold,
)


# ============================================================
# BEST YOUDEN J FOR DOCUMENTATION
# ============================================================

best_youden_j = float(
    sweep[
        "youden_j"
    ].max()
)


best_youden_rows = sweep[
    np.isclose(
        sweep[
            "youden_j"
        ],
        best_youden_j,
        atol=1e-12,
        rtol=0,
    )
].copy()


best_youden_threshold = float(
    np.median(
        best_youden_rows[
            "threshold"
        ].to_numpy(
            dtype=float
        )
    )
)


best_youden_metrics = calculate_metrics(
    y_true,
    scores,
    best_youden_threshold,
)


# ============================================================
# SAVE SWEEP SORTED FOR REVIEW
# ============================================================

sweep = sweep.sort_values(
    by=[
        "balanced_accuracy",
        "youden_j",
        "sensitivity_specificity_gap",
    ],
    ascending=[
        False,
        False,
        True,
    ],
)


sweep.to_csv(
    SWEEP_CSV,
    index=False,
)


# ============================================================
# PRINT CANDIDATE
# ============================================================

print()
print("=" * 76)
print("OASIS V2 CANDIDATE THRESHOLD")
print("=" * 76)

print()
print(
    "Candidate threshold:",
    f"{candidate_threshold:.12f}",
)

print()
print(
    "Original V1 threshold:",
    f"{ORIGINAL_THRESHOLD:.12f}",
)

print()
print(
    "Threshold shift:",
    f"{candidate_threshold - ORIGINAL_THRESHOLD:+.12f}",
)

print()
print("Candidate metrics:")

for key, value in candidate_metrics.items():
    print(
        f"{key}: {value}"
    )


print()
print("=" * 76)
print("BEST YOUDEN-J CHECK")
print("=" * 76)

print()
print(
    "Best Youden-J threshold:",
    f"{best_youden_threshold:.12f}",
)

print(
    "Best Youden J:",
    f"{best_youden_j:.6f}",
)

print(
    "Balanced accuracy:",
    f"{best_youden_metrics['balanced_accuracy']:.6f}",
)


# ============================================================
# SAVE CANDIDATE — DO NOT TOUCH V1 CONFIG
# ============================================================

candidate_payload = {

    "project_stage":
        "OASIS V2 THRESHOLD CALIBRATION",

    "status":
        "DEVELOPMENT_CALIBRATION_ONLY",

    "warning":
        (
            "This threshold was selected using the current OASIS "
            "development/calibration cases. It must NOT be reported "
            "as independently externally validated performance."
        ),

    "source":
        "BRAIN_MGZ",

    "number_of_calibration_cases":
        int(
            len(
                calibration
            )
        ),

    "normal_cases":
        int(
            np.sum(
                y_true == 0
            )
        ),

    "early_demented_cases":
        int(
            np.sum(
                y_true == 1
            )
        ),

    "original_v1_threshold":
        float(
            ORIGINAL_THRESHOLD
        ),

    "candidate_v2_threshold":
        float(
            candidate_threshold
        ),

    "threshold_shift":
        float(
            candidate_threshold
            -
            ORIGINAL_THRESHOLD
        ),

    "selection_rule":
        (
            "Maximize balanced accuracy; among tied thresholds "
            "minimize absolute sensitivity-specificity difference; "
            "if multiple thresholds remain, use their median."
        ),

    "candidate_metrics_on_calibration_set":
        candidate_metrics,

    "best_balanced_accuracy":
        float(
            best_balanced_accuracy
        ),

    "best_youden_j":
        float(
            best_youden_j
        ),

    "best_youden_threshold":
        float(
            best_youden_threshold
        ),

    "roc_auc":
        float(
            roc_auc
        ),

    "pr_auc":
        float(
            pr_auc
        ),

    "model_changed":
        False,

    "feature_set_changed":
        False,

    "brain_mask_changed":
        False,

    "original_v1_threshold_changed":
        False,

    "next_required_step":
        (
            "Freeze this as an OASIS V2 candidate threshold, then "
            "evaluate once on a NEW untouched OASIS subject set "
            "that was not used during calibration or previous "
            "development experiments."
        ),
}


with open(
    CANDIDATE_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        candidate_payload,
        f,
        indent=4,
    )


# ============================================================
# TEXT REPORT
# ============================================================

report_lines = [

    "OASIS V2 THRESHOLD CALIBRATION",
    "=" * 76,
    "",

    "STATUS:",
    "DEVELOPMENT / CALIBRATION ONLY",
    "",

    "Original V1 model:",
    "UNCHANGED",
    "",

    "Original V1 threshold:",
    f"{ORIGINAL_THRESHOLD:.12f}",
    "",

    "Number of calibration cases:",
    str(len(calibration)),
    "",

    "ROC-AUC:",
    f"{roc_auc:.6f}",
    "",

    "PR-AUC:",
    f"{pr_auc:.6f}",
    "",

    "Candidate V2 threshold:",
    f"{candidate_threshold:.12f}",
    "",

    "Candidate balanced accuracy:",
    f"{candidate_metrics['balanced_accuracy']:.6f}",
    "",

    "Candidate sensitivity:",
    f"{candidate_metrics['sensitivity']:.6f}",
    "",

    "Candidate specificity:",
    f"{candidate_metrics['specificity']:.6f}",
    "",

    "Candidate precision:",
    f"{candidate_metrics['precision']:.6f}",
    "",

    "Candidate F1:",
    f"{candidate_metrics['f1']:.6f}",
    "",

    "Confusion matrix:",
    f"TN = {candidate_metrics['TN']}",
    f"FP = {candidate_metrics['FP']}",
    f"FN = {candidate_metrics['FN']}",
    f"TP = {candidate_metrics['TP']}",
    "",

    "IMPORTANT:",
    (
        "This performance is measured on the SAME OASIS cases used "
        "to select the threshold."
    ),

    (
        "It is calibration/development performance, NOT external "
        "validation performance."
    ),

    "",

    "NEXT STEP:",
    (
        "Freeze the candidate threshold and evaluate it once on "
        "a completely new untouched OASIS test cohort."
    ),
]


REPORT_TXT.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 76)
print("FILES SAVED")
print("=" * 76)

print()
print("Calibration cases:")
print(
    CALIBRATION_CASES_CSV
)

print()
print("Full threshold sweep:")
print(
    SWEEP_CSV
)

print()
print("V2 candidate threshold:")
print(
    CANDIDATE_JSON
)

print()
print("Calibration report:")
print(
    REPORT_TXT
)


print()
print("=" * 76)
print("OASIS V2 THRESHOLD CALIBRATION COMPLETE")
print("=" * 76)

print()
print(
    "The original V1 model and original V1 threshold "
    "were NOT modified."
)

print()
print(
    "NEXT: collect a completely new untouched OASIS test set."
)