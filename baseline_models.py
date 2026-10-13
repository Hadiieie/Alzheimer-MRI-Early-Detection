from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "TRAINING_PREPARATION"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "BASELINE_MODELS"
)

MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# IMPORTANT
# =========================================================
#
# TEST SET IS NOT LOADED HERE.
#
# It remains locked until:
# - feature-set decision
# - model selection
# - hyperparameter tuning
# - final model decision
#
# =========================================================


DATASETS = {

    "FULL_30": {
        "train":
            DATA_DIR / "train_full_scaled.csv",

        "validation":
            DATA_DIR / "validation_full_scaled.csv"
    },

    "REDUCED_21": {
        "train":
            DATA_DIR / "train_reduced_scaled.csv",

        "validation":
            DATA_DIR / "validation_reduced_scaled.csv"
    }
}


# =========================================================
# METADATA COLUMNS
# =========================================================

NON_FEATURE_COLUMNS = [
    "filename",
    "leakage_group",
    "target_name",
    "target"
]


# =========================================================
# MODELS
# =========================================================

def create_models():

    return {

        "Logistic_Regression":

            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ),


        "SVM_RBF":

            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                cache_size=2000,
                probability=False
            ),


        "Random_Forest":

            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),


        "KNN":

            KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                n_jobs=-1
            )
    }


# =========================================================
# METRIC FUNCTION
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_score
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )


    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )


    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),

        "sensitivity_recall":
            sensitivity,

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_score
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_score
            ),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp)
    }


    return metrics


# =========================================================
# SCORE EXTRACTION
# =========================================================

def get_model_score(
    model,
    X
):

    if hasattr(
        model,
        "predict_proba"
    ):

        return (
            model
            .predict_proba(X)[:, 1]
        )


    if hasattr(
        model,
        "decision_function"
    ):

        return (
            model
            .decision_function(X)
        )


    raise ValueError(
        "Model has no probability "
        "or decision-function output."
    )


# =========================================================
# RESULTS
# =========================================================

all_results = []


print("\n==========================================")
print("BASELINE MACHINE LEARNING")
print("==========================================")

print(
    "\nPositive class:"
    " Early_Demented = 1"
)

print(
    "Negative class:"
    " Normal = 0"
)

print(
    "\nTEST SET STATUS:"
    " LOCKED / NOT LOADED"
)


# =========================================================
# LOOP THROUGH FEATURE SETS
# =========================================================

for feature_set_name, paths in DATASETS.items():

    print("\n\n==========================================")
    print(
        f"FEATURE SET: {feature_set_name}"
    )
    print("==========================================")


    train_df = pd.read_csv(
        paths["train"]
    )

    val_df = pd.read_csv(
        paths["validation"]
    )


    feature_columns = [
        c
        for c in train_df.columns
        if c not in NON_FEATURE_COLUMNS
    ]


    print("\nNumber of features:")
    print(
        len(feature_columns)
    )


    print("\nTraining images:")
    print(
        len(train_df)
    )


    print("\nValidation images:")
    print(
        len(val_df)
    )


    # -----------------------------------------------------
    # DATA MATRICES
    # -----------------------------------------------------

    X_train = (
        train_df[
            feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_train = (
        train_df[
            "target"
        ]
        .to_numpy(
            dtype=int
        )
    )


    X_val = (
        val_df[
            feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_val = (
        val_df[
            "target"
        ]
        .to_numpy(
            dtype=int
        )
    )


    # -----------------------------------------------------
    # QC
    # -----------------------------------------------------

    if np.isnan(X_train).any():
        raise ValueError(
            "NaN detected in TRAIN."
        )

    if np.isnan(X_val).any():
        raise ValueError(
            "NaN detected in VALIDATION."
        )

    if np.isinf(X_train).any():
        raise ValueError(
            "Inf detected in TRAIN."
        )

    if np.isinf(X_val).any():
        raise ValueError(
            "Inf detected in VALIDATION."
        )


    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

    models = create_models()


    for model_name, model in models.items():

        print("\n------------------------------------------")
        print(
            f"{feature_set_name}"
            f" | {model_name}"
        )
        print("------------------------------------------")


        # =================================================
        # TRAIN
        # =================================================

        start_time = time.time()


        model.fit(
            X_train,
            y_train
        )


        training_time = (
            time.time()
            -
            start_time
        )


        # =================================================
        # VALIDATION PREDICTIONS
        # =================================================

        y_pred = model.predict(
            X_val
        )


        y_score = get_model_score(
            model,
            X_val
        )


        # =================================================
        # METRICS
        # =================================================

        metrics = calculate_metrics(
            y_val,
            y_pred,
            y_score
        )


        print(
            "\nAccuracy:"
            f" {metrics['accuracy']:.4f}"
        )

        print(
            "Balanced Accuracy:"
            f" {metrics['balanced_accuracy']:.4f}"
        )

        print(
            "Sensitivity:"
            f" {metrics['sensitivity_recall']:.4f}"
        )

        print(
            "Specificity:"
            f" {metrics['specificity']:.4f}"
        )

        print(
            "Precision:"
            f" {metrics['precision']:.4f}"
        )

        print(
            "F1:"
            f" {metrics['f1']:.4f}"
        )

        print(
            "ROC-AUC:"
            f" {metrics['roc_auc']:.4f}"
        )

        print(
            "PR-AUC:"
            f" {metrics['pr_auc']:.4f}"
        )


        print("\nConfusion Matrix:")

        print(
            f"TN = {metrics['TN']}"
        )

        print(
            f"FP = {metrics['FP']}"
        )

        print(
            f"FN = {metrics['FN']}"
        )

        print(
            f"TP = {metrics['TP']}"
        )


        print(
            "\nTraining time:"
            f" {training_time:.2f} sec"
        )


        # =================================================
        # SAVE RESULT
        # =================================================

        row = {

            "feature_set":
                feature_set_name,

            "model":
                model_name,

            "n_features":
                len(feature_columns),

            "training_time_sec":
                training_time,

            **metrics
        }


        all_results.append(
            row
        )


        # =================================================
        # SAVE BASELINE MODEL
        # =================================================

        model_filename = (
            f"{feature_set_name}"
            f"__{model_name}"
            f".joblib"
        )


        joblib.dump(
            model,
            MODEL_DIR
            / model_filename
        )


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(
    OUTPUT_DIR
    / "baseline_validation_results.csv",
    index=False
)


# =========================================================
# PRINT TABLE
# =========================================================

display_columns = [

    "feature_set",
    "model",
    "n_features",

    "accuracy",
    "balanced_accuracy",

    "sensitivity_recall",
    "specificity",

    "precision",
    "f1",

    "roc_auc",
    "pr_auc",

    "training_time_sec"
]


print("\n\n==========================================")
print("FINAL BASELINE COMPARISON")
print("==========================================\n")


print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# =========================================================
# SORTED VIEWS
# =========================================================

print("\n\n==========================================")
print("SORTED BY ROC-AUC")
print("==========================================\n")


print(
    results_df[
        display_columns
    ]
    .sort_values(
        "roc_auc",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print("\n\n==========================================")
print("SORTED BY BALANCED ACCURACY")
print("==========================================\n")


print(
    results_df[
        display_columns
    ]
    .sort_values(
        "balanced_accuracy",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print("\n\n==========================================")
print("BASELINE MODELING COMPLETE")
print("==========================================")

print(
    "\nTEST SET WAS NOT USED."
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)