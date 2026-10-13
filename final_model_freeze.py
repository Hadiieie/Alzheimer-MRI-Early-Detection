# ============================================================
# FINAL MODEL FREEZE
# Alzheimer MRI Early Detection Project
# ============================================================
#
# Purpose:
# Freeze the exact final model configuration BEFORE opening
# or evaluating the test set.
#
# NO TRAINING
# NO VALIDATION OPTIMIZATION
# NO TEST LOADING
#
# ============================================================

from pathlib import Path
import hashlib
import json
import shutil
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

TUNING_DIR = (
    RESULTS_DIR
    / "HYPERPARAMETER_TUNING"
)

TRAINING_PREP_DIR = (
    RESULTS_DIR
    / "TRAINING_PREPARATION"
)

THRESHOLD_DIR = (
    RESULTS_DIR
    / "THRESHOLD_OPTIMIZATION"
)

FINAL_DIR = (
    RESULTS_DIR
    / "FINAL_MODEL"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SOURCE FILES
# ============================================================

SOURCE_MODEL = (
    TUNING_DIR
    / "models"
    / "FULL_30__SVM_RBF__TUNED.joblib"
)

SOURCE_FEATURES = (
    TRAINING_PREP_DIR
    / "full_feature_set.txt"
)

SOURCE_SCALER = (
    TRAINING_PREP_DIR
    / "standard_scaler_full.joblib"
)

SOURCE_THRESHOLD = (
    THRESHOLD_DIR
    / "selected_threshold.json"
)


# ============================================================
# 3. FINAL OUTPUT FILES
# ============================================================

FINAL_MODEL = (
    FINAL_DIR
    / "final_model.joblib"
)

FINAL_FEATURES = (
    FINAL_DIR
    / "final_features.txt"
)

FINAL_SCALER = (
    FINAL_DIR
    / "final_scaler.joblib"
)

FINAL_THRESHOLD_SOURCE = (
    FINAL_DIR
    / "selected_threshold.json"
)

FINAL_CONFIG = (
    FINAL_DIR
    / "final_model_config.json"
)

FINAL_MANIFEST = (
    FINAL_DIR
    / "freeze_manifest.json"
)

FINAL_REPORT = (
    FINAL_DIR
    / "model_freeze_report.txt"
)


# ============================================================
# 4. HEADER
# ============================================================

print("\n")
print("=" * 75)
print("FINAL MODEL FREEZE")
print("=" * 75)

print("\nProject:")
print("Alzheimer MRI Early Detection")

print("\nChosen model:")
print("FULL_30 + Tuned SVM RBF")

print("\nTEST SET:")
print("LOCKED / NOT LOADED")


# ============================================================
# 5. VERIFY REQUIRED FILES
# ============================================================

required_files = [
    SOURCE_MODEL,
    SOURCE_FEATURES,
    SOURCE_THRESHOLD
]


print("\n")
print("=" * 75)
print("SOURCE FILE CHECK")
print("=" * 75)


for file_path in required_files:

    print(
        f"\nChecking:\n{file_path}"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}"
        )

    print("FOUND")


if SOURCE_SCALER.exists():

    print(
        "\nScaler found:"
    )

    print(
        SOURCE_SCALER
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "Standalone FULL_30 scaler was not found."
    )


# ============================================================
# 6. LOAD FEATURE LIST
# ============================================================

with open(
    SOURCE_FEATURES,
    "r",
    encoding="utf-8"
) as file:

    features = [
        line.strip()
        for line in file
        if line.strip()
    ]


print("\n")
print("=" * 75)
print("FEATURE SET")
print("=" * 75)

print(
    "\nNumber of features:",
    len(features)
)


if len(features) != 30:

    raise ValueError(
        f"Expected 30 FULL_30 features, "
        f"but found {len(features)}."
    )


for index, feature in enumerate(
    features,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )


# ============================================================
# 7. LOAD THRESHOLD INFORMATION
# ============================================================

with open(
    SOURCE_THRESHOLD,
    "r",
    encoding="utf-8"
) as file:

    threshold_info = json.load(
        file
    )


if "threshold" not in threshold_info:

    raise ValueError(
        "Threshold field missing from "
        "selected_threshold.json"
    )


selected_threshold = float(
    threshold_info[
        "threshold"
    ]
)


print("\n")
print("=" * 75)
print("THRESHOLD")
print("=" * 75)

print(
    "\nSelected threshold:",
    selected_threshold
)

print(
    "\nSelection criterion:",
    threshold_info.get(
        "criterion",
        "UNKNOWN"
    )
)

print(
    "\nSelection dataset:",
    threshold_info.get(
        "selection_dataset",
        "UNKNOWN"
    )
)


# ============================================================
# 8. IMPORTANT VALIDATION-ONLY SAFETY CHECK
# ============================================================

selection_dataset = str(
    threshold_info.get(
        "selection_dataset",
        ""
    )
).upper()


if (
    selection_dataset
    and
    selection_dataset != "VALIDATION"
):

    raise ValueError(
        "Threshold was not marked as selected "
        "using VALIDATION."
    )


if threshold_info.get(
    "test_set_used",
    False
):

    raise ValueError(
        "Threshold configuration says TEST was used. "
        "Freeze aborted."
    )


# ============================================================
# 9. LOAD MODEL FOR STRUCTURAL CHECK
# ============================================================

model = joblib.load(
    SOURCE_MODEL
)


print("\n")
print("=" * 75)
print("MODEL STRUCTURE CHECK")
print("=" * 75)

print(
    "\nLoaded object type:"
)

print(
    type(model)
)


if not hasattr(
    model,
    "decision_function"
):

    raise ValueError(
        "Final SVM model does not expose "
        "decision_function()."
    )


print(
    "\ndecision_function(): AVAILABLE"
)


if hasattr(
    model,
    "predict"
):

    print(
        "predict(): AVAILABLE"
    )


# ============================================================
# 10. DETECT PIPELINE
# ============================================================

is_pipeline = hasattr(
    model,
    "named_steps"
)


print(
    "\nModel is sklearn Pipeline:",
    is_pipeline
)


if is_pipeline:

    print(
        "\nPipeline steps:"
    )

    for step_name in model.named_steps:

        print(
            f" - {step_name}"
        )


# ============================================================
# 11. COPY EXACT FINAL ARTIFACTS
# ============================================================

print("\n")
print("=" * 75)
print("FREEZING ARTIFACTS")
print("=" * 75)


shutil.copy2(
    SOURCE_MODEL,
    FINAL_MODEL
)


shutil.copy2(
    SOURCE_FEATURES,
    FINAL_FEATURES
)


shutil.copy2(
    SOURCE_THRESHOLD,
    FINAL_THRESHOLD_SOURCE
)


if SOURCE_SCALER.exists():

    shutil.copy2(
        SOURCE_SCALER,
        FINAL_SCALER
    )


print(
    "\nFinal model copied."
)

print(
    "Final feature list copied."
)

print(
    "Threshold configuration copied."
)


if SOURCE_SCALER.exists():

    print(
        "Scaler copied."
    )


# ============================================================
# 12. SHA256 FILE HASH FUNCTION
# ============================================================

def sha256_file(
    file_path,
    chunk_size=1024 * 1024
):

    digest = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ============================================================
# 13. CALCULATE HASHES
# ============================================================

model_hash = sha256_file(
    FINAL_MODEL
)

features_hash = sha256_file(
    FINAL_FEATURES
)

threshold_hash = sha256_file(
    FINAL_THRESHOLD_SOURCE
)


scaler_hash = None


if FINAL_SCALER.exists():

    scaler_hash = sha256_file(
        FINAL_SCALER
    )


# ============================================================
# 14. FINAL CONFIGURATION
# ============================================================

final_config = {

    "project":
        "Alzheimer MRI Early Detection",

    "status":
        "FROZEN",

    "model_name":
        "SVM_RBF",

    "model_variant":
        "FULL_30__SVM_RBF__TUNED",

    "feature_set":
        "FULL_30",

    "number_of_features":
        len(features),

    "features":
        features,

    "decision_method":
        "decision_function",

    "selected_threshold":
        selected_threshold,

    "threshold_selection_dataset":
        threshold_info.get(
            "selection_dataset",
            "VALIDATION"
        ),

    "threshold_selection_criterion":
        threshold_info.get(
            "criterion",
            "MAX_BALANCED_ACCURACY"
        ),

    "validation_accuracy":
        threshold_info.get(
            "accuracy"
        ),

    "validation_balanced_accuracy":
        threshold_info.get(
            "balanced_accuracy"
        ),

    "validation_sensitivity":
        threshold_info.get(
            "sensitivity"
        ),

    "validation_specificity":
        threshold_info.get(
            "specificity"
        ),

    "validation_precision":
        threshold_info.get(
            "precision"
        ),

    "validation_f1":
        threshold_info.get(
            "f1"
        ),

    "validation_roc_auc":
        threshold_info.get(
            "roc_auc"
        ),

    "validation_pr_auc":
        threshold_info.get(
            "pr_auc"
        ),

    "is_sklearn_pipeline":
        bool(
            is_pipeline
        ),

    "standalone_scaler_available":
        FINAL_SCALER.exists(),

    "test_set_status":
        "LOCKED_NOT_LOADED",

    "test_used_for_model_selection":
        False,

    "test_used_for_threshold_selection":
        False
}


with open(
    FINAL_CONFIG,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_config,
        file,
        indent=4
    )


# ============================================================
# 15. FREEZE MANIFEST
# ============================================================

manifest = {

    "final_model": {

        "filename":
            FINAL_MODEL.name,

        "sha256":
            model_hash
    },

    "feature_list": {

        "filename":
            FINAL_FEATURES.name,

        "sha256":
            features_hash
    },

    "threshold_file": {

        "filename":
            FINAL_THRESHOLD_SOURCE.name,

        "sha256":
            threshold_hash
    },

    "scaler": {

        "filename":
            (
                FINAL_SCALER.name
                if FINAL_SCALER.exists()
                else None
            ),

        "sha256":
            scaler_hash
    }
}


with open(
    FINAL_MANIFEST,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        indent=4
    )


# ============================================================
# 16. HUMAN-READABLE FREEZE REPORT
# ============================================================

report_lines = [

    "=" * 75,

    "FINAL MODEL FREEZE REPORT",

    "=" * 75,

    "",

    "Project:",
    "Alzheimer MRI Early Detection",

    "",

    "FINAL MODEL:",
    "Tuned SVM RBF",

    "",

    "FEATURE SET:",
    "FULL_30",

    "",

    f"NUMBER OF FEATURES: {len(features)}",

    "",

    "SELECTED THRESHOLD:",
    f"{selected_threshold:.12f}",

    "",

    "THRESHOLD SELECTION:",
    str(
        threshold_info.get(
            "criterion",
            "MAX_BALANCED_ACCURACY"
        )
    ),

    "",

    "THRESHOLD DATASET:",
    str(
        threshold_info.get(
            "selection_dataset",
            "VALIDATION"
        )
    ),

    "",

    "VALIDATION PERFORMANCE:",

    (
        "Balanced Accuracy: "
        f"{threshold_info.get('balanced_accuracy')}"
    ),

    (
        "Sensitivity: "
        f"{threshold_info.get('sensitivity')}"
    ),

    (
        "Specificity: "
        f"{threshold_info.get('specificity')}"
    ),

    (
        "ROC-AUC: "
        f"{threshold_info.get('roc_auc')}"
    ),

    (
        "PR-AUC: "
        f"{threshold_info.get('pr_auc')}"
    ),

    "",

    "MODEL SHA256:",
    model_hash,

    "",

    "FEATURE LIST SHA256:",
    features_hash,

    "",

    "THRESHOLD FILE SHA256:",
    threshold_hash,

    "",

    "SCALER SHA256:",
    (
        scaler_hash
        if scaler_hash is not None
        else "N/A"
    ),

    "",

    "TEST SET STATUS:",
    "LOCKED / NOT LOADED",

    "",

    "IMPORTANT:",
    (
        "No model selection, threshold optimization, "
        "or preprocessing decision may be changed "
        "after the test set is opened."
    ),

    "",

    "=" * 75
]


with open(
    FINAL_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# 17. VERIFY COPIED FILE HASHES
# ============================================================

if (
    sha256_file(
        SOURCE_MODEL
    )
    !=
    sha256_file(
        FINAL_MODEL
    )
):

    raise RuntimeError(
        "Model hash mismatch after copying."
    )


if (
    sha256_file(
        SOURCE_FEATURES
    )
    !=
    sha256_file(
        FINAL_FEATURES
    )
):

    raise RuntimeError(
        "Feature-list hash mismatch after copying."
    )


if SOURCE_SCALER.exists():

    if (
        sha256_file(
            SOURCE_SCALER
        )
        !=
        sha256_file(
            FINAL_SCALER
        )
    ):

        raise RuntimeError(
            "Scaler hash mismatch after copying."
        )


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("FINAL MODEL FREEZE COMPLETE")
print("=" * 75)

print("\nFinal model:")
print(
    FINAL_MODEL
)

print("\nFinal feature list:")
print(
    FINAL_FEATURES
)

if FINAL_SCALER.exists():

    print("\nFinal scaler:")
    print(
        FINAL_SCALER
    )


print("\nFinal threshold:")
print(
    f"{selected_threshold:.12f}"
)


print("\nFinal configuration:")
print(
    FINAL_CONFIG
)


print("\nFreeze manifest:")
print(
    FINAL_MANIFEST
)


print("\nFreeze report:")
print(
    FINAL_REPORT
)


print("\nModel SHA256:")
print(
    model_hash
)


print("\nTEST SET:")
print(
    "LOCKED / NOT LOADED"
)


print("\nNEXT STEP:")
print(
    "ONE-TIME FINAL TEST EVALUATION"
)


print("\nIMPORTANT:")
print(
    "Do not modify the model, feature set, scaler, "
    "or threshold after this point."
)


print("=" * 75)