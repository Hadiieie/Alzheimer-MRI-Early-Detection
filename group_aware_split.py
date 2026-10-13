from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "PRIMARY_EARLY_DETECTION"
    / "primary_early_detection_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "GROUP_AWARE_SPLIT"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VAL_FILE   = OUTPUT_DIR / "validation.csv"
TEST_FILE  = OUTPUT_DIR / "test.csv"

FULL_SPLIT_FILE = (
    OUTPUT_DIR
    / "primary_dataset_with_split.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "split_summary.txt"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)


print("\n==========================================")
print("GROUP-AWARE TRAIN / VAL / TEST SPLIT")
print("==========================================")

print("\nTotal images:")
print(len(df))


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "filename",
    "original_class",
    "target",
    "target_name",
    "leakage_group"
]

missing = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


if df["leakage_group"].isna().any():
    raise ValueError(
        "Missing leakage_group values detected."
    )


# =========================================================
# CHECK GROUP PURITY
# =========================================================

group_target_counts = (
    df
    .groupby("leakage_group")["target"]
    .nunique()
)

mixed_groups = (
    group_target_counts[
        group_target_counts > 1
    ]
)

print("\nGroups containing multiple targets:")
print(len(mixed_groups))


if len(mixed_groups) > 0:
    raise ValueError(
        "Mixed-target leakage groups found. "
        "Stop before splitting."
    )


# =========================================================
# CREATE GROUP-LEVEL TABLE
# =========================================================

group_df = (
    df
    .groupby("leakage_group")
    .agg(
        target=("target", "first"),
        target_name=("target_name", "first"),
        group_size=("filename", "count")
    )
    .reset_index()
)


print("\nUnique leakage groups:")
print(len(group_df))


print("\nGroup-level target distribution:")
print(
    group_df[
        "target_name"
    ].value_counts()
)


# =========================================================
# FUNCTION TO CREATE A CANDIDATE SPLIT
# =========================================================

def create_candidate_split(
    group_table,
    random_state
):

    # -----------------------------------------------------
    # 70% TRAIN
    # 30% TEMP
    # -----------------------------------------------------

    train_groups, temp_groups = train_test_split(
        group_table,
        test_size=0.30,
        random_state=random_state,
        stratify=group_table["target"]
    )


    # -----------------------------------------------------
    # TEMP → 15% VALIDATION + 15% TEST
    # -----------------------------------------------------

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_groups["target"]
    )


    return (
        train_groups,
        val_groups,
        test_groups
    )


# =========================================================
# SCORE A SPLIT
# =========================================================

TARGET_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}


overall_positive_ratio = (
    df["target"].mean()
)


def split_score(
    train_groups,
    val_groups,
    test_groups
):

    group_sets = {
        "train":
            set(
                train_groups[
                    "leakage_group"
                ]
            ),

        "validation":
            set(
                val_groups[
                    "leakage_group"
                ]
            ),

        "test":
            set(
                test_groups[
                    "leakage_group"
                ]
            )
    }


    score = 0.0


    for split_name, groups in group_sets.items():

        split_df = df[
            df["leakage_group"]
            .isin(groups)
        ]


        # -------------------------------------------------
        # Difference from desired dataset size
        # -------------------------------------------------

        actual_size_ratio = (
            len(split_df)
            /
            len(df)
        )


        desired_size_ratio = (
            TARGET_SPLIT_RATIOS[
                split_name
            ]
        )


        size_error = abs(
            actual_size_ratio
            -
            desired_size_ratio
        )


        # -------------------------------------------------
        # Difference from overall target prevalence
        # -------------------------------------------------

        class_ratio = (
            split_df[
                "target"
            ].mean()
        )


        class_error = abs(
            class_ratio
            -
            overall_positive_ratio
        )


        score += (
            size_error
            +
            class_error
        )


    return score


# =========================================================
# SEARCH MANY RANDOM STATES
# =========================================================

best_score = None
best_seed = None
best_split = None


print("\nSearching for balanced group-aware split...")


for seed in range(1000):

    (
        candidate_train,
        candidate_val,
        candidate_test
    ) = create_candidate_split(
        group_df,
        seed
    )


    score = split_score(
        candidate_train,
        candidate_val,
        candidate_test
    )


    if (
        best_score is None
        or
        score < best_score
    ):

        best_score = score
        best_seed = seed

        best_split = (
            candidate_train,
            candidate_val,
            candidate_test
        )


print("\nBest random state:")
print(best_seed)

print("\nBest split score:")
print(best_score)


train_groups, val_groups, test_groups = best_split


# =========================================================
# GROUP SETS
# =========================================================

train_group_ids = set(
    train_groups[
        "leakage_group"
    ]
)

val_group_ids = set(
    val_groups[
        "leakage_group"
    ]
)

test_group_ids = set(
    test_groups[
        "leakage_group"
    ]
)


# =========================================================
# VERIFY GROUP OVERLAP = ZERO
# =========================================================

train_val_overlap = (
    train_group_ids
    &
    val_group_ids
)

train_test_overlap = (
    train_group_ids
    &
    test_group_ids
)

val_test_overlap = (
    val_group_ids
    &
    test_group_ids
)


print("\n==========================================")
print("GROUP LEAKAGE CHECK")
print("==========================================")


print(
    "\nTrain ↔ Validation overlap:",
    len(train_val_overlap)
)

print(
    "Train ↔ Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation ↔ Test overlap:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) != 0
    or
    len(train_test_overlap) != 0
    or
    len(val_test_overlap) != 0
):
    raise RuntimeError(
        "GROUP LEAKAGE DETECTED."
    )


# =========================================================
# ASSIGN SPLIT TO EACH IMAGE
# =========================================================

def assign_split(group_id):

    if group_id in train_group_ids:
        return "train"

    if group_id in val_group_ids:
        return "validation"

    if group_id in test_group_ids:
        return "test"

    raise RuntimeError(
        f"Group {group_id} was not assigned."
    )


df["split"] = (
    df["leakage_group"]
    .apply(assign_split)
)


# =========================================================
# CREATE DATAFRAMES
# =========================================================

train_df = (
    df[
        df["split"]
        ==
        "train"
    ]
    .copy()
)

val_df = (
    df[
        df["split"]
        ==
        "validation"
    ]
    .copy()
)

test_df = (
    df[
        df["split"]
        ==
        "test"
    ]
    .copy()
)


# =========================================================
# FINAL SANITY CHECK
# =========================================================

assert (
    len(train_df)
    +
    len(val_df)
    +
    len(test_df)
    ==
    len(df)
)


assert (
    set(
        train_df[
            "leakage_group"
        ]
    )
    .isdisjoint(
        set(
            val_df[
                "leakage_group"
            ]
        )
    )
)


assert (
    set(
        train_df[
            "leakage_group"
        ]
    )
    .isdisjoint(
        set(
            test_df[
                "leakage_group"
            ]
        )
    )
)


assert (
    set(
        val_df[
            "leakage_group"
        ]
    )
    .isdisjoint(
        set(
            test_df[
                "leakage_group"
            ]
        )
    )
)


# =========================================================
# SUMMARY FUNCTION
# =========================================================

def print_split_summary(
    name,
    split_df
):

    print("\n------------------------------------------")
    print(name.upper())
    print("------------------------------------------")

    print("\nImages:")
    print(len(split_df))

    print("\nPercentage of full dataset:")
    print(
        round(
            len(split_df)
            /
            len(df)
            *
            100,
            3
        )
    )

    print("\nUnique leakage groups:")
    print(
        split_df[
            "leakage_group"
        ].nunique()
    )

    print("\nBinary distribution:")
    print(
        split_df[
            "target_name"
        ].value_counts()
    )

    print("\nBinary percentages:")
    print(
        (
            split_df[
                "target_name"
            ]
            .value_counts(
                normalize=True
            )
            *
            100
        ).round(3)
    )

    print("\nOriginal-class distribution:")
    print(
        split_df[
            "original_class"
        ].value_counts()
    )


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n==========================================")
print("FINAL SPLIT SUMMARY")
print("==========================================")


print_split_summary(
    "TRAIN",
    train_df
)

print_split_summary(
    "VALIDATION",
    val_df
)

print_split_summary(
    "TEST",
    test_df
)


# =========================================================
# SAVE FILES
# =========================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

df.to_csv(
    FULL_SPLIT_FILE,
    index=False
)


# =========================================================
# SAVE TEXT SUMMARY
# =========================================================

summary_lines = [

    "GROUP-AWARE SPLIT SUMMARY",
    "=" * 60,

    f"Random state: {best_seed}",
    f"Total images: {len(df)}",
    f"Total leakage groups: {df['leakage_group'].nunique()}",

    "",

    f"Train images: {len(train_df)}",
    f"Validation images: {len(val_df)}",
    f"Test images: {len(test_df)}",

    "",

    f"Train groups: {train_df['leakage_group'].nunique()}",
    f"Validation groups: {val_df['leakage_group'].nunique()}",
    f"Test groups: {test_df['leakage_group'].nunique()}",

    "",

    f"Train-Val overlap: {len(train_val_overlap)}",
    f"Train-Test overlap: {len(train_test_overlap)}",
    f"Val-Test overlap: {len(val_test_overlap)}"
]


SUMMARY_FILE.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


print("\n==========================================")
print("GROUP-AWARE SPLIT COMPLETE")
print("==========================================")


print("\nFiles saved inside:")
print(OUTPUT_DIR)