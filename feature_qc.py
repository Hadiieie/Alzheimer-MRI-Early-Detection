from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_FILE = (
    RESULTS_DIR /
    "FULL_FEATURE_EXTRACTION" /
    "full_features_39538.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR /
    "FEATURE_QC"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD FEATURE DATASET
# =========================================================

df = pd.read_csv(
    FEATURE_FILE
)


print("\n==========================================")
print("FEATURE QC")
print("==========================================")


print("\nTotal rows:")
print(len(df))

print("\nTotal columns:")
print(len(df.columns))


# =========================================================
# DEFINE METADATA / FEATURE COLUMNS
# =========================================================

NON_FEATURE_COLUMNS = [
    "filename",
    "original_class",
    "path",
    "processed_path",
    "otsu_threshold",
    "leakage_group",
    "study_group"
]


feature_columns = [
    c
    for c in df.columns
    if c not in NON_FEATURE_COLUMNS
]


print("\nNumber of numerical features:")
print(len(feature_columns))


print("\nFeature names:")
for i, feature in enumerate(
    feature_columns,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )


# =========================================================
# FORCE NUMERIC
# =========================================================

X = df[
    feature_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)


# =========================================================
# 1. NaN / INFINITY CHECK
# =========================================================

nan_per_feature = (
    X.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


inf_per_feature = pd.Series(
    {
        col: int(
            np.isinf(
                X[col].to_numpy(
                    dtype=float
                )
            ).sum()
        )
        for col in feature_columns
    }
).sort_values(
    ascending=False
)


total_nan = int(
    nan_per_feature.sum()
)

total_inf = int(
    inf_per_feature.sum()
)


print("\n==========================================")
print("1. MISSING / INFINITE VALUES")
print("==========================================")


print("\nTotal NaN:")
print(total_nan)

print("\nTotal infinite:")
print(total_inf)


# Save
nan_per_feature.to_csv(
    OUTPUT_DIR /
    "nan_per_feature.csv",
    header=[
        "nan_count"
    ]
)

inf_per_feature.to_csv(
    OUTPUT_DIR /
    "inf_per_feature.csv",
    header=[
        "inf_count"
    ]
)


# =========================================================
# 2. CONSTANT / LOW-VARIANCE FEATURES
# =========================================================

unique_counts = (
    X.nunique(
        dropna=False
    )
    .sort_values()
)


constant_features = (
    unique_counts[
        unique_counts <= 1
    ]
    .index
    .tolist()
)


variances = (
    X.var()
    .sort_values()
)


print("\n==========================================")
print("2. CONSTANT / VARIANCE CHECK")
print("==========================================")


print("\nConstant features:")
print(
    constant_features
)


print("\nFive smallest variances:")
print(
    variances.head(5)
)


unique_counts.to_csv(
    OUTPUT_DIR /
    "feature_unique_counts.csv",
    header=[
        "unique_values"
    ]
)


variances.to_csv(
    OUTPUT_DIR /
    "feature_variances.csv",
    header=[
        "variance"
    ]
)


# =========================================================
# 3. FEATURE SUMMARY
# =========================================================

summary = pd.DataFrame({
    "min":
        X.min(),

    "q1":
        X.quantile(
            0.25
        ),

    "median":
        X.median(),

    "mean":
        X.mean(),

    "q3":
        X.quantile(
            0.75
        ),

    "std":
        X.std(),

    "max":
        X.max(),

    "unique_values":
        X.nunique()
})


summary.to_csv(
    OUTPUT_DIR /
    "feature_summary_qc.csv"
)


print("\n==========================================")
print("3. FEATURE SUMMARY")
print("==========================================")


print(summary)


# =========================================================
# 4. CLASS DISTRIBUTION
# =========================================================

print("\n==========================================")
print("4. CLASS DISTRIBUTION")
print("==========================================")


class_distribution = (
    df[
        "original_class"
    ]
    .value_counts()
)


print(
    class_distribution
)


class_distribution.to_csv(
    OUTPUT_DIR /
    "class_distribution.csv",
    header=[
        "count"
    ]
)


# =========================================================
# 5. FEATURE SUMMARY BY CLASS
# =========================================================

class_mean = (
    df
    .groupby(
        "original_class"
    )[
        feature_columns
    ]
    .mean()
)


class_median = (
    df
    .groupby(
        "original_class"
    )[
        feature_columns
    ]
    .median()
)


class_std = (
    df
    .groupby(
        "original_class"
    )[
        feature_columns
    ]
    .std()
)


class_mean.to_csv(
    OUTPUT_DIR /
    "feature_mean_by_class.csv"
)


class_median.to_csv(
    OUTPUT_DIR /
    "feature_median_by_class.csv"
)


class_std.to_csv(
    OUTPUT_DIR /
    "feature_std_by_class.csv"
)


print("\n==========================================")
print("5. FEATURE MEANS BY CLASS")
print("==========================================")


print(
    class_mean.T
)


# =========================================================
# 6. CORRELATION MATRIX
# =========================================================

corr = (
    X.corr(
        method="pearson"
    )
)


corr.to_csv(
    OUTPUT_DIR /
    "feature_correlation_matrix.csv"
)


# =========================================================
# HIGH-CORRELATION PAIRS
# =========================================================

high_corr_records = []


for i in range(
    len(feature_columns)
):

    for j in range(
        i + 1,
        len(feature_columns)
    ):

        feature_a = (
            feature_columns[i]
        )

        feature_b = (
            feature_columns[j]
        )

        r = corr.loc[
            feature_a,
            feature_b
        ]


        if pd.isna(r):
            continue


        if abs(r) >= 0.95:

            high_corr_records.append({
                "feature_a":
                    feature_a,

                "feature_b":
                    feature_b,

                "correlation":
                    float(r),

                "abs_correlation":
                    float(
                        abs(r)
                    )
            })


high_corr_df = pd.DataFrame(
    high_corr_records
)


if len(high_corr_df) > 0:

    high_corr_df = (
        high_corr_df
        .sort_values(
            "abs_correlation",
            ascending=False
        )
    )


high_corr_df.to_csv(
    OUTPUT_DIR /
    "high_correlation_pairs_095.csv",
    index=False
)


print("\n==========================================")
print("6. HIGH CORRELATION PAIRS |r| >= 0.95")
print("==========================================")


if len(high_corr_df) == 0:

    print(
        "None"
    )

else:

    print(
        high_corr_df.to_string(
            index=False
        )
    )


# =========================================================
# 7. EXACT DUPLICATE FEATURE VECTORS
# =========================================================

duplicate_feature_rows = (
    df.duplicated(
        subset=feature_columns,
        keep=False
    )
)


duplicate_count = int(
    duplicate_feature_rows.sum()
)


duplicate_vectors_df = (
    df.loc[
        duplicate_feature_rows,
        [
            "filename",
            "original_class"
        ]
        +
        feature_columns
    ]
    .copy()
)


duplicate_vectors_df.to_csv(
    OUTPUT_DIR /
    "duplicate_feature_vectors.csv",
    index=False
)


print("\n==========================================")
print("7. EXACT DUPLICATE FEATURE VECTORS")
print("==========================================")


print(
    "Rows involved:",
    duplicate_count
)


# =========================================================
# 8. CROSS-CLASS IDENTICAL FEATURE VECTORS
# =========================================================

if duplicate_count > 0:

    grouped_duplicates = (
        duplicate_vectors_df
        .groupby(
            feature_columns,
            dropna=False
        )
    )


    cross_class_records = []


    for _, group in grouped_duplicates:

        classes = (
            group[
                "original_class"
            ]
            .unique()
        )


        if len(classes) > 1:

            for _, row in group.iterrows():

                cross_class_records.append(
                    row.to_dict()
                )


    cross_class_df = pd.DataFrame(
        cross_class_records
    )

else:

    cross_class_df = pd.DataFrame()


cross_class_df.to_csv(
    OUTPUT_DIR /
    "cross_class_duplicate_feature_vectors.csv",
    index=False
)


print("\nCross-class duplicate-feature rows:")
print(
    len(cross_class_df)
)


# =========================================================
# 9. IQR OUTLIER COUNTS
# =========================================================

outlier_records = []


for feature in feature_columns:

    q1 = X[
        feature
    ].quantile(
        0.25
    )


    q3 = X[
        feature
    ].quantile(
        0.75
    )


    iqr = (
        q3 - q1
    )


    lower = (
        q1 -
        1.5 * iqr
    )


    upper = (
        q3 +
        1.5 * iqr
    )


    outlier_count = int(
        (
            (
                X[
                    feature
                ] < lower
            )
            |
            (
                X[
                    feature
                ] > upper
            )
        ).sum()
    )


    outlier_records.append({

        "feature":
            feature,

        "q1":
            q1,

        "q3":
            q3,

        "iqr":
            iqr,

        "lower_bound":
            lower,

        "upper_bound":
            upper,

        "outlier_count":
            outlier_count,

        "outlier_percentage":
            (
                outlier_count
                /
                len(X)
                *
                100.0
            )

    })


outlier_df = pd.DataFrame(
    outlier_records
)


outlier_df = (
    outlier_df
    .sort_values(
        "outlier_percentage",
        ascending=False
    )
)


outlier_df.to_csv(
    OUTPUT_DIR /
    "iqr_outlier_summary.csv",
    index=False
)


print("\n==========================================")
print("8. IQR OUTLIER SUMMARY")
print("==========================================")


print(
    outlier_df[
        [
            "feature",
            "outlier_count",
            "outlier_percentage"
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# 10. CORRELATION HEATMAP
# =========================================================

fig, ax = plt.subplots(
    figsize=(16, 14)
)


im = ax.imshow(
    corr.to_numpy(),
    aspect="auto",
    vmin=-1,
    vmax=1
)


ax.set_xticks(
    np.arange(
        len(feature_columns)
    )
)


ax.set_yticks(
    np.arange(
        len(feature_columns)
    )
)


ax.set_xticklabels(
    feature_columns,
    rotation=90,
    fontsize=7
)


ax.set_yticklabels(
    feature_columns,
    fontsize=7
)


ax.set_title(
    "Feature Correlation Matrix"
)


fig.colorbar(
    im,
    ax=ax
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR /
    "feature_correlation_heatmap.png",
    dpi=200
)


plt.close()


# =========================================================
# 11. DISTRIBUTION PLOTS
# =========================================================

PLOT_DIR = (
    OUTPUT_DIR /
    "FEATURE_DISTRIBUTIONS"
)


PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


classes = list(
    df[
        "original_class"
    ].dropna().unique()
)


for feature in feature_columns:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    for class_name in classes:

        values = (
            df.loc[
                df[
                    "original_class"
                ]
                ==
                class_name,
                feature
            ]
            .dropna()
            .to_numpy()
        )


        ax.hist(
            values,
            bins=40,
            alpha=0.35,
            label=class_name
        )


    ax.set_title(
        f"{feature} Distribution by Class"
    )


    ax.set_xlabel(
        feature
    )


    ax.set_ylabel(
        "Frequency"
    )


    ax.legend(
        fontsize=8
    )


    plt.tight_layout()


    plt.savefig(
        PLOT_DIR /
        f"{feature}_distribution.png",
        dpi=160
    )


    plt.close()


# =========================================================
# 12. GEOMETRY FEATURES — SPECIAL CHECK
# =========================================================

geometry_features = [
    "brain_area",
    "brain_ratio",
    "brain_perimeter",
    "compactness",
    "bbox_aspect_ratio"
]


existing_geometry = [
    f
    for f in geometry_features
    if f in feature_columns
]


geometry_summary = (
    df.groupby(
        "original_class"
    )[
        existing_geometry
    ]
    .agg(
        [
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
)


geometry_summary.to_csv(
    OUTPUT_DIR /
    "geometry_features_by_class.csv"
)


print("\n==========================================")
print("9. GEOMETRY FEATURES BY CLASS")
print("==========================================")


print(
    geometry_summary
)


# =========================================================
# 13. FINAL QC SUMMARY
# =========================================================

QC_SUMMARY_FILE = (
    OUTPUT_DIR /
    "feature_qc_summary.txt"
)


summary_lines = [

    "FEATURE QC SUMMARY",
    "=" * 60,

    f"Rows: {len(df)}",
    f"Numerical features: {len(feature_columns)}",

    f"Total NaN: {total_nan}",
    f"Total Inf: {total_inf}",

    (
        "Constant features: "
        +
        str(
            constant_features
        )
    ),

    (
        "High-correlation pairs "
        f"(|r| >= 0.95): "
        f"{len(high_corr_df)}"
    ),

    (
        "Rows involved in exact "
        f"duplicate feature vectors: "
        f"{duplicate_count}"
    ),

    (
        "Cross-class duplicate-feature rows: "
        f"{len(cross_class_df)}"
    )

]


QC_SUMMARY_FILE.write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8"
)


print("\n==========================================")
print("FEATURE QC COMPLETE")
print("==========================================")


for line in summary_lines:
    print(line)


print("\nResults saved inside:")
print(
    OUTPUT_DIR
)