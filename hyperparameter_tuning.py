# ============================================================
# HYPERPARAMETER TUNING - LIGHT / RESUMABLE VERSION
# ============================================================
#
# Project:
# Alzheimer MRI Early Detection
#
# IMPORTANT:
# - TRAIN is used for hyperparameter tuning.
# - Cross-validation is GROUP-AWARE.
# - Scaling for SVM happens INSIDE each CV fold.
# - VALIDATION is used only after tuning each model.
# - TEST remains completely LOCKED and is NOT loaded.
#
# Models tuned:
#   1. Random Forest
#   2. SVM RBF
#
# Feature sets:
#   1. FULL_30
#   2. REDUCED_21
#
# Total searches:
#   FULL_30    + Random Forest
#   FULL_30    + SVM
#   REDUCED_21 + Random Forest
#   REDUCED_21 + SVM
#
# ============================================================


from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedGroupKFold,
    RandomizedSearchCV
)

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


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


SPLIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "GROUP_AWARE_SPLIT"
)


PREP_DIR = (
    PROJECT_ROOT
    / "results"
    / "TRAINING_PREPARATION"
)


# We use a NEW folder so old interrupted tuning runs
# do not interfere with this version.

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "HYPERPARAMETER_TUNING_V2"
)


MODEL_DIR = (
    OUTPUT_DIR
    / "models"
)


CV_RESULTS_DIR = (
    OUTPUT_DIR
    / "cv_results"
)


PARAMETERS_DIR = (
    OUTPUT_DIR
    / "best_parameters"
)


SUMMARY_DIR = (
    OUTPUT_DIR
    / "individual_summaries"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CV_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PARAMETERS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT FILES
# ============================================================

TRAIN_FILE = (
    SPLIT_DIR
    / "train.csv"
)


VALIDATION_FILE = (
    SPLIT_DIR
    / "validation.csv"
)


FULL_FEATURE_FILE = (
    PREP_DIR
    / "full_feature_set.txt"
)


REDUCED_FEATURE_FILE = (
    PREP_DIR
    / "reduced_feature_set.txt"
)


# ============================================================
# 3. SETTINGS
# ============================================================

RANDOM_STATE = 42


# ------------------------------------------------------------
# RESUME MODE
# ------------------------------------------------------------
#
# True:
# If one complete tuning configuration was already finished,
# it will not be repeated.
#
# Example:
# FULL_30 Random Forest finished yesterday,
# but SVM was interrupted.
#
# Running this script again will skip the finished RF model.
# ------------------------------------------------------------

RESUME_COMPLETED_SEARCHES = True


# ------------------------------------------------------------
# PARALLEL JOBS
# ------------------------------------------------------------
#
# 2 = reasonable speed without excessive load.
#
# If the laptop becomes unstable or RAM usage becomes high:
# change to:
#
# N_JOBS = 1
# ------------------------------------------------------------

N_JOBS = 2


# ============================================================
# 4. DISPLAY HEADER
# ============================================================

print("\n")
print("=" * 70)
print("HYPERPARAMETER TUNING V2 - LIGHT VERSION")
print("=" * 70)

print("\nTEST SET STATUS:")
print("LOCKED / NOT LOADED")

print("\nCross-validation:")
print("3-fold StratifiedGroupKFold")

print("\nRandom Forest search:")
print("12 candidates x 3 folds = 36 fits per feature set")

print("\nSVM search:")
print("10 candidates x 3 folds = 30 fits per feature set")


# ============================================================
# 5. VERIFY INPUT FILES
# ============================================================

required_files = [
    TRAIN_FILE,
    VALIDATION_FILE,
    FULL_FEATURE_FILE,
    REDUCED_FEATURE_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )


# ============================================================
# 6. LOAD TRAIN + VALIDATION
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)


val_df = pd.read_csv(
    VALIDATION_FILE
)


print("\n")
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)


print("\nTRAIN images:")
print(
    len(train_df)
)


print("\nVALIDATION images:")
print(
    len(val_df)
)


print("\nTEST:")
print(
    "LOCKED / NOT LOADED"
)


# ============================================================
# 7. REQUIRED COLUMNS QC
# ============================================================

required_columns = [
    "target",
    "leakage_group"
]


for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing TRAIN column: {column}"
        )

    if column not in val_df.columns:

        raise ValueError(
            f"Missing VALIDATION column: {column}"
        )


# ============================================================
# 8. TARGET QC
# ============================================================

print("\nTRAIN target distribution:")

print(
    train_df[
        "target"
    ].value_counts(
        sort=False
    )
)


print("\nVALIDATION target distribution:")

print(
    val_df[
        "target"
    ].value_counts(
        sort=False
    )
)


train_targets = set(
    train_df[
        "target"
    ].unique()
)


val_targets = set(
    val_df[
        "target"
    ].unique()
)


if not train_targets.issubset({0, 1}):

    raise ValueError(
        "TRAIN target contains values other than 0/1."
    )


if not val_targets.issubset({0, 1}):

    raise ValueError(
        "VALIDATION target contains values other than 0/1."
    )


# ============================================================
# 9. GROUP LEAKAGE QC
# ============================================================

train_groups_set = set(
    train_df[
        "leakage_group"
    ]
)


val_groups_set = set(
    val_df[
        "leakage_group"
    ]
)


train_val_overlap = (
    train_groups_set
    &
    val_groups_set
)


print("\n")
print("=" * 70)
print("GROUP LEAKAGE QC")
print("=" * 70)


print("\nTrain <-> Validation group overlap:")

print(
    len(train_val_overlap)
)


if len(train_val_overlap) != 0:

    raise RuntimeError(
        "\nSTOPPING."
        "\nLeakage groups overlap between TRAIN and VALIDATION."
    )


print("\nGROUP LEAKAGE QC: PASS")


# ============================================================
# 10. LOAD FEATURE LISTS
# ============================================================

def load_feature_list(
    file_path
):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        features = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return features


full_features = load_feature_list(
    FULL_FEATURE_FILE
)


reduced_features = load_feature_list(
    REDUCED_FEATURE_FILE
)


print("\n")
print("=" * 70)
print("FEATURE SETS")
print("=" * 70)


print("\nFULL features:")
print(
    len(full_features)
)


print("\nREDUCED features:")
print(
    len(reduced_features)
)


if len(full_features) != 30:

    print(
        "\nWARNING:"
        f" Expected 30 full features,"
        f" found {len(full_features)}."
    )


if len(reduced_features) != 21:

    print(
        "\nWARNING:"
        f" Expected 21 reduced features,"
        f" found {len(reduced_features)}."
    )


# ============================================================
# 11. FEATURE EXISTENCE QC
# ============================================================

all_required_features = set(
    full_features
    +
    reduced_features
)


missing_train_features = [

    feature

    for feature
    in all_required_features

    if feature
    not in train_df.columns
]


missing_val_features = [

    feature

    for feature
    in all_required_features

    if feature
    not in val_df.columns
]


if missing_train_features:

    raise ValueError(
        "\nMissing TRAIN features:\n"
        +
        "\n".join(
            missing_train_features
        )
    )


if missing_val_features:

    raise ValueError(
        "\nMissing VALIDATION features:\n"
        +
        "\n".join(
            missing_val_features
        )
    )


FEATURE_SETS = {

    "FULL_30":
        full_features,

    "REDUCED_21":
        reduced_features
}


# ============================================================
# 12. TARGET + GROUP ARRAYS
# ============================================================

y_train = (
    train_df[
        "target"
    ]
    .to_numpy(
        dtype=int
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


groups_train = (
    train_df[
        "leakage_group"
    ]
    .to_numpy()
)


# ============================================================
# 13. GROUP-AWARE CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(

    n_splits=3,

    shuffle=True,

    random_state=RANDOM_STATE
)


# ============================================================
# 14. MULTIPLE SCORING METRICS
# ============================================================

SCORING = {

    "balanced_accuracy":
        "balanced_accuracy",

    "roc_auc":
        "roc_auc",

    "f1":
        "f1",

    "recall":
        "recall",

    "precision":
        "precision"
}


# ============================================================
# 15. RANDOM FOREST SEARCH
# ============================================================

def create_random_forest_search():

    pipeline = Pipeline([

        (
            "clf",

            RandomForestClassifier(

                random_state=
                    RANDOM_STATE,

                # Important:
                # CV parallelism is handled
                # by RandomizedSearchCV.
                n_jobs=1
            )
        )
    ])


    parameter_space = {

        "clf__n_estimators": [
            200,
            400,
            600
        ],

        "clf__max_depth": [
            None,
            12,
            20
        ],

        "clf__min_samples_split": [
            2,
            5,
            10
        ],

        "clf__min_samples_leaf": [
            1,
            2,
            4
        ],

        "clf__max_features": [
            "sqrt",
            "log2"
        ],

        "clf__class_weight": [
            None,
            "balanced",
            "balanced_subsample"
        ]
    }


    search = RandomizedSearchCV(

        estimator=
            pipeline,

        param_distributions=
            parameter_space,

        n_iter=
            12,

        scoring=
            SCORING,

        refit=
            "balanced_accuracy",

        cv=
            cv,

        random_state=
            RANDOM_STATE,

        n_jobs=
            N_JOBS,

        verbose=
            2,

        return_train_score=
            False,

        error_score=
            "raise"
    )


    return search


# ============================================================
# 16. SVM RBF SEARCH
# ============================================================

def create_svm_search():

    # IMPORTANT:
    # StandardScaler is INSIDE Pipeline.
    #
    # Therefore during CV:
    #
    # scaler.fit()
    #
    # sees ONLY the training part
    # of each individual CV fold.
    #
    # This prevents CV scaling leakage.

    pipeline = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "clf",

            SVC(

                kernel="rbf",

                probability=False
            )
        )
    ])


    parameter_space = {

        "clf__C": [
            0.1,
            0.5,
            1,
            2,
            5,
            10
        ],

        "clf__gamma": [
            "scale",
            0.001,
            0.005,
            0.01,
            0.05
        ],

        "clf__class_weight": [
            None,
            "balanced"
        ]
    }


    search = RandomizedSearchCV(

        estimator=
            pipeline,

        param_distributions=
            parameter_space,

        n_iter=
            10,

        scoring=
            SCORING,

        refit=
            "balanced_accuracy",

        cv=
            cv,

        random_state=
            RANDOM_STATE,

        n_jobs=
            N_JOBS,

        verbose=
            2,

        return_train_score=
            False,

        error_score=
            "raise"
    )


    return search


# ============================================================
# 17. MODEL SCORE EXTRACTION
# ============================================================

def get_model_score(
    model,
    X
):

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model
            .predict_proba(
                X
            )
        )

        return (
            probabilities[:, 1]
        )


    if hasattr(
        model,
        "decision_function"
    ):

        return (
            model
            .decision_function(
                X
            )
        )


    raise ValueError(
        "Model has neither predict_proba "
        "nor decision_function."
    )


# ============================================================
# 18. VALIDATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_score
):

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=[
            0,
            1
        ]
    )


    tn, fp, fn, tp = (
        matrix.ravel()
    )


    sensitivity = recall_score(

        y_true,

        y_pred,

        pos_label=1,

        zero_division=0
    )


    if (
        tn + fp
    ) > 0:

        specificity = (
            tn
            /
            (
                tn
                +
                fp
            )
        )

    else:

        specificity = np.nan


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


        "sensitivity":

            sensitivity,


        "specificity":

            specificity,


        "precision":

            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),


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


# ============================================================
# 19. SAFE JSON CONVERSION
# ============================================================

def make_json_safe(
    dictionary
):

    safe_dictionary = {}


    for key, value in dictionary.items():

        if isinstance(
            value,
            np.integer
        ):

            value = int(
                value
            )


        elif isinstance(
            value,
            np.floating
        ):

            value = float(
                value
            )


        elif isinstance(
            value,
            np.bool_
        ):

            value = bool(
                value
            )


        safe_dictionary[
            key
        ] = value


    return safe_dictionary


# ============================================================
# 20. RESUME HELPERS
# ============================================================

def get_summary_path(
    feature_set_name,
    model_name
):

    return (

        SUMMARY_DIR

        /

        (
            f"{feature_set_name}"
            f"__{model_name}"
            f"__summary.json"
        )
    )


def load_completed_summary(
    feature_set_name,
    model_name
):

    summary_path = get_summary_path(
        feature_set_name,
        model_name
    )


    if not summary_path.exists():

        return None


    try:

        with open(
            summary_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        if (
            data.get(
                "status"
            )
            ==
            "COMPLETE"
        ):

            return data


    except Exception:

        return None


    return None


# ============================================================
# 21. RUN SEARCHES
# ============================================================

all_results = []


TOTAL_CONFIGURATIONS = 4

configuration_number = 0


for (
    feature_set_name,
    features
) in FEATURE_SETS.items():


    print("\n\n")
    print("#" * 70)

    print(
        f"FEATURE SET: "
        f"{feature_set_name}"
    )

    print("#" * 70)


    print("\nNumber of features:")

    print(
        len(features)
    )


    # --------------------------------------------------------
    # RAW FEATURE MATRICES
    # --------------------------------------------------------
    #
    # These are intentionally NOT pre-scaled.
    #
    # SVM scaling happens within Pipeline / CV.
    # --------------------------------------------------------

    X_train = (
        train_df[
            features
        ]
        .to_numpy(
            dtype=float
        )
    )


    X_val = (
        val_df[
            features
        ]
        .to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # NaN / Inf QC
    # --------------------------------------------------------

    train_nan = (
        np.isnan(
            X_train
        )
        .sum()
    )


    val_nan = (
        np.isnan(
            X_val
        )
        .sum()
    )


    train_inf = (
        np.isinf(
            X_train
        )
        .sum()
    )


    val_inf = (
        np.isinf(
            X_val
        )
        .sum()
    )


    print("\nNumerical QC:")

    print(
        "TRAIN NaN:",
        train_nan
    )

    print(
        "VALIDATION NaN:",
        val_nan
    )

    print(
        "TRAIN Inf:",
        train_inf
    )

    print(
        "VALIDATION Inf:",
        val_inf
    )


    if (
        train_nan != 0
        or
        val_nan != 0
        or
        train_inf != 0
        or
        val_inf != 0
    ):

        raise ValueError(
            "NaN or Inf found in model features."
        )


    # --------------------------------------------------------
    # SEARCH DEFINITIONS
    # --------------------------------------------------------

    searches = {

        "Random_Forest":
            create_random_forest_search(),

        "SVM_RBF":
            create_svm_search()
    }


    # ========================================================
    # MODEL LOOP
    # ========================================================

    for (
        model_name,
        search
    ) in searches.items():


        configuration_number += 1


        print("\n\n")
        print("=" * 70)

        print(
            f"CONFIGURATION "
            f"{configuration_number}"
            f"/{TOTAL_CONFIGURATIONS}"
        )

        print("=" * 70)


        print(
            f"\n{feature_set_name}"
            f" | "
            f"{model_name}"
        )


        print("\n" + "-" * 70)


        # ====================================================
        # RESUME CHECK
        # ====================================================

        if RESUME_COMPLETED_SEARCHES:

            completed = (
                load_completed_summary(
                    feature_set_name,
                    model_name
                )
            )


            if completed is not None:

                print(
                    "\nSTATUS:"
                    " ALREADY COMPLETE"
                )

                print(
                    "Skipping this configuration."
                )


                all_results.append(
                    completed[
                        "results"
                    ]
                )


                continue


        # ====================================================
        # SEARCH INFORMATION
        # ====================================================

        if (
            model_name
            ==
            "Random_Forest"
        ):

            print(
                "\nExpected search:"
            )

            print(
                "12 candidates"
                " x "
                "3 folds"
                " = "
                "36 fits"
            )


        elif (
            model_name
            ==
            "SVM_RBF"
        ):

            print(
                "\nExpected search:"
            )

            print(
                "10 candidates"
                " x "
                "3 folds"
                " = "
                "30 fits"
            )


        print(
            "\nStarting search..."
        )


        start_time = (
            time.time()
        )


        # ====================================================
        # FIT RANDOMIZED SEARCH
        # ====================================================

        search.fit(

            X_train,

            y_train,

            groups=
                groups_train
        )


        elapsed_time = (
            time.time()
            -
            start_time
        )


        # ====================================================
        # BEST CV RESULT
        # ====================================================

        print("\n")
        print("-" * 70)
        print("SEARCH COMPLETE")
        print("-" * 70)


        print(
            "\nBest CV Balanced Accuracy:"
        )

        print(
            f"{search.best_score_:.6f}"
        )


        print(
            "\nBest parameters:"
        )


        for (
            parameter,
            value
        ) in search.best_params_.items():

            print(
                f"{parameter}"
                f": "
                f"{value}"
            )


        print(
            "\nSearch time:"
        )

        print(
            f"{elapsed_time:.2f}"
            " seconds"
        )


        print(
            f"({elapsed_time / 60:.2f}"
            " minutes)"
        )


        # ====================================================
        # SAVE COMPLETE CV TABLE
        # ====================================================

        cv_results_df = pd.DataFrame(
            search.cv_results_
        )


        cv_results_file = (

            CV_RESULTS_DIR

            /

            (
                f"{feature_set_name}"
                f"__{model_name}"
                f"__cv_results.csv"
            )
        )


        cv_results_df.to_csv(

            cv_results_file,

            index=False
        )


        # ====================================================
        # BEST MODEL
        # ====================================================

        best_model = (
            search.best_estimator_
        )


        model_file = (

            MODEL_DIR

            /

            (
                f"{feature_set_name}"
                f"__{model_name}"
                f"__TUNED.joblib"
            )
        )


        joblib.dump(

            best_model,

            model_file
        )


        # ====================================================
        # SAVE BEST PARAMETERS
        # ====================================================

        safe_params = make_json_safe(
            search.best_params_
        )


        parameter_file = (

            PARAMETERS_DIR

            /

            (
                f"{feature_set_name}"
                f"__{model_name}"
                f"__best_params.json"
            )
        )


        with open(

            parameter_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                safe_params,

                file,

                indent=4
            )


        # ====================================================
        # VALIDATION EVALUATION
        # ====================================================

        print("\n")
        print("=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)


        y_pred = (
            best_model
            .predict(
                X_val
            )
        )


        y_score = get_model_score(

            best_model,

            X_val
        )


        metrics = calculate_metrics(

            y_val,

            y_pred,

            y_score
        )


        print(
            "\nAccuracy:"
            f" {metrics['accuracy']:.6f}"
        )


        print(
            "Balanced Accuracy:"
            f" {metrics['balanced_accuracy']:.6f}"
        )


        print(
            "Sensitivity:"
            f" {metrics['sensitivity']:.6f}"
        )


        print(
            "Specificity:"
            f" {metrics['specificity']:.6f}"
        )


        print(
            "Precision:"
            f" {metrics['precision']:.6f}"
        )


        print(
            "F1:"
            f" {metrics['f1']:.6f}"
        )


        print(
            "ROC-AUC:"
            f" {metrics['roc_auc']:.6f}"
        )


        print(
            "PR-AUC:"
            f" {metrics['pr_auc']:.6f}"
        )


        print("\nConfusion Matrix:")


        print(
            f"TN = "
            f"{metrics['TN']}"
        )


        print(
            f"FP = "
            f"{metrics['FP']}"
        )


        print(
            f"FN = "
            f"{metrics['FN']}"
        )


        print(
            f"TP = "
            f"{metrics['TP']}"
        )


        # ====================================================
        # RESULT RECORD
        # ====================================================

        result_row = {

            "feature_set":
                feature_set_name,

            "model":
                model_name,

            "n_features":
                int(
                    len(features)
                ),

            "best_cv_balanced_accuracy":
                float(
                    search.best_score_
                ),

            "accuracy":
                float(
                    metrics[
                        "accuracy"
                    ]
                ),

            "balanced_accuracy":
                float(
                    metrics[
                        "balanced_accuracy"
                    ]
                ),

            "sensitivity":
                float(
                    metrics[
                        "sensitivity"
                    ]
                ),

            "specificity":
                float(
                    metrics[
                        "specificity"
                    ]
                ),

            "precision":
                float(
                    metrics[
                        "precision"
                    ]
                ),

            "f1":
                float(
                    metrics[
                        "f1"
                    ]
                ),

            "roc_auc":
                float(
                    metrics[
                        "roc_auc"
                    ]
                ),

            "pr_auc":
                float(
                    metrics[
                        "pr_auc"
                    ]
                ),

            "TN":
                int(
                    metrics[
                        "TN"
                    ]
                ),

            "FP":
                int(
                    metrics[
                        "FP"
                    ]
                ),

            "FN":
                int(
                    metrics[
                        "FN"
                    ]
                ),

            "TP":
                int(
                    metrics[
                        "TP"
                    ]
                ),

            "search_time_sec":
                float(
                    elapsed_time
                ),

            "search_time_min":
                float(
                    elapsed_time
                    /
                    60
                )
        }


        all_results.append(
            result_row
        )


        # ====================================================
        # SAVE INDIVIDUAL COMPLETION SUMMARY
        # ====================================================
        #
        # This enables safe restart.
        # ====================================================

        individual_summary = {

            "status":
                "COMPLETE",

            "feature_set":
                feature_set_name,

            "model":
                model_name,

            "best_parameters":
                safe_params,

            "results":
                result_row
        }


        summary_file = get_summary_path(

            feature_set_name,

            model_name
        )


        with open(

            summary_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                individual_summary,

                file,

                indent=4
            )


        # ====================================================
        # SAVE CURRENT GLOBAL RESULTS AFTER EACH MODEL
        # ====================================================

        partial_results_df = pd.DataFrame(
            all_results
        )


        partial_results_df.to_csv(

            OUTPUT_DIR
            /
            "tuned_validation_results_PARTIAL.csv",

            index=False
        )


        print("\nSaved completed configuration.")

        print(
            "If the script stops later,"
            " this completed model"
            " will be skipped next time."
        )


# ============================================================
# 22. FINAL RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


if results_df.empty:

    raise RuntimeError(
        "No tuning results available."
    )


# Remove accidental duplicates
# if summaries were loaded more than once.

results_df = (
    results_df
    .drop_duplicates(
        subset=[
            "feature_set",
            "model"
        ],
        keep="last"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 23. SAVE FINAL RESULTS
# ============================================================

final_results_file = (

    OUTPUT_DIR

    /

    "tuned_validation_results.csv"
)


results_df.to_csv(

    final_results_file,

    index=False
)


# ============================================================
# 24. DISPLAY FINAL COMPARISON
# ============================================================

display_columns = [

    "feature_set",

    "model",

    "n_features",

    "best_cv_balanced_accuracy",

    "accuracy",

    "balanced_accuracy",

    "sensitivity",

    "specificity",

    "precision",

    "f1",

    "roc_auc",

    "pr_auc",

    "TN",

    "FP",

    "FN",

    "TP",

    "search_time_min"
]


print("\n\n")
print("=" * 100)
print("FINAL TUNED MODEL COMPARISON")
print("=" * 100)
print("\n")


final_display = (

    results_df[
        display_columns
    ]

    .sort_values(
        by=
            "balanced_accuracy",

        ascending=
            False
    )
)


print(
    final_display
    .to_string(
        index=False
    )
)


# ============================================================
# 25. SORT BY ROC-AUC
# ============================================================

print("\n\n")
print("=" * 100)
print("TUNED MODELS SORTED BY ROC-AUC")
print("=" * 100)
print("\n")


print(

    results_df[
        display_columns
    ]

    .sort_values(
        by=
            "roc_auc",

        ascending=
            False
    )

    .to_string(
        index=False
    )
)


# ============================================================
# 26. SORT BY SENSITIVITY
# ============================================================

print("\n\n")
print("=" * 100)
print("TUNED MODELS SORTED BY SENSITIVITY")
print("=" * 100)
print("\n")


print(

    results_df[
        display_columns
    ]

    .sort_values(
        by=
            "sensitivity",

        ascending=
            False
    )

    .to_string(
        index=False
    )
)


# ============================================================
# 27. COMPLETION MESSAGE
# ============================================================

print("\n\n")
print("=" * 70)
print("HYPERPARAMETER TUNING V2 COMPLETE")
print("=" * 70)


print("\nCompleted configurations:")

print(
    len(
        results_df
    )
)


print("\nExpected configurations:")

print(
    TOTAL_CONFIGURATIONS
)


if (
    len(
        results_df
    )
    ==
    TOTAL_CONFIGURATIONS
):

    print(
        "\nSTATUS:"
        " ALL 4 CONFIGURATIONS COMPLETE"
    )

else:

    print(
        "\nSTATUS:"
        " SOME CONFIGURATIONS ARE STILL MISSING"
    )


print("\nTEST SET STATUS:")

print(
    "STILL LOCKED / NEVER LOADED"
)


print("\nFinal results saved to:")

print(
    final_results_file
)


print("\nAll outputs saved inside:")

print(
    OUTPUT_DIR
)


print("\nNEXT PROJECT STEP:")

print(
    "Compare tuned models "
    "and perform validation-based "
    "threshold optimization."
)


print("\nDO NOT OPEN THE TEST SET YET.")

print("=" * 70)