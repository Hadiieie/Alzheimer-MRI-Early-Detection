from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

EDA_DIR = RESULTS_DIR / "EDA"

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_FILE = (
    RESULTS_DIR /
    "clean_4class_grouped_metadata.csv"
)

PRIMARY_FILE = (
    RESULTS_DIR /
    "primary_grouped_metadata.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

master = pd.read_csv(MASTER_FILE)

primary = pd.read_csv(PRIMARY_FILE)


print("\n==========================================")
print("EXPLORATORY DATA ANALYSIS")
print("==========================================\n")


print("Master dataset:")
print(len(master))


print("\nPrimary dataset:")
print(len(primary))


# =========================================================
# 1. ORIGINAL CLASS DISTRIBUTION
# =========================================================

print("\n==========================================")
print("1. ORIGINAL CLASS DISTRIBUTION")
print("==========================================")

class_counts = (
    master["original_class"]
    .value_counts()
)


print(class_counts)


plt.figure(
    figsize=(10, 6)
)

class_counts.plot(
    kind="bar"
)

plt.title(
    "Original Four-Class Distribution"
)

plt.xlabel(
    "Original Class"
)

plt.ylabel(
    "Number of Images"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    EDA_DIR /
    "01_original_class_distribution.png",
    dpi=300
)

plt.close()


# =========================================================
# 2. PRIMARY STUDY DISTRIBUTION
# =========================================================

print("\n==========================================")
print("2. PRIMARY STUDY DISTRIBUTION")
print("==========================================")

study_counts = (
    primary["study_group"]
    .value_counts()
)


print(study_counts)


print("\nPercentages:")

print(
    primary["study_group"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


plt.figure(
    figsize=(8, 6)
)

study_counts.plot(
    kind="bar"
)

plt.title(
    "Primary Early-Detection Dataset"
)

plt.xlabel(
    "Study Group"
)

plt.ylabel(
    "Number of Images"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    EDA_DIR /
    "02_primary_study_distribution.png",
    dpi=300
)

plt.close()


# =========================================================
# 3. RESOLUTION BY ORIGINAL CLASS
# =========================================================

print("\n==========================================")
print("3. RESOLUTION BY ORIGINAL CLASS")
print("==========================================")


master["resolution"] = (

    master["width"]
    .astype(str)

    + "x"

    + master["height"]
    .astype(str)
)


resolution_table = pd.crosstab(

    master["original_class"],

    master["resolution"]

)


print(resolution_table)


resolution_table.to_csv(
    EDA_DIR /
    "resolution_by_class.csv"
)


resolution_table.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Image Resolution by Original Class"
)

plt.xlabel(
    "Original Class"
)

plt.ylabel(
    "Number of Images"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    EDA_DIR /
    "03_resolution_by_class.png",
    dpi=300
)

plt.close()


# =========================================================
# 4. IMAGE MODE BY ORIGINAL CLASS
# =========================================================

print("\n==========================================")
print("4. IMAGE MODE BY ORIGINAL CLASS")
print("==========================================")

mode_table = pd.crosstab(

    master["original_class"],

    master["mode"]

)


print(mode_table)


mode_table.to_csv(
    EDA_DIR /
    "mode_by_class.csv"
)


mode_table.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Image Mode by Original Class"
)

plt.xlabel(
    "Original Class"
)

plt.ylabel(
    "Number of Images"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    EDA_DIR /
    "04_mode_by_class.png",
    dpi=300
)

plt.close()


# =========================================================
# 5. RANDOM IMAGE SAMPLES
# =========================================================

print("\n==========================================")
print("5. RANDOM IMAGE SAMPLES")
print("==========================================")


classes = [

    "NonDemented",

    "VeryMildDemented",

    "MildDemented",

    "ModerateDemented"

]


samples_per_class = 5


fig, axes = plt.subplots(

    len(classes),

    samples_per_class,

    figsize=(15, 12)

)


for row_index, class_name in enumerate(classes):

    class_df = master[
        master["original_class"]
        == class_name
    ]


    sample_df = class_df.sample(

        n=samples_per_class,

        random_state=42

    )


    for col_index, (_, row) in enumerate(
        sample_df.iterrows()
    ):

        image_path = Path(
            row["path"]
        )


        with Image.open(
            image_path
        ) as img:

            img = img.convert("L")

            axes[
                row_index,
                col_index
            ].imshow(
                img,
                cmap="gray"
            )


        axes[
            row_index,
            col_index
        ].axis("off")


        if col_index == 0:

            axes[
                row_index,
                col_index
            ].set_ylabel(
                class_name,
                fontsize=10
            )


plt.suptitle(
    "Random MRI Samples from Each Original Class"
)

plt.tight_layout()

plt.savefig(
    EDA_DIR /
    "05_random_samples.png",
    dpi=300
)

plt.close()


print(
    "Random sample figure created."
)


# =========================================================
# 6. IMAGE INTENSITY STATISTICS
# =========================================================

print("\n==========================================")
print("6. INTENSITY STATISTICS")
print("==========================================")


intensity_records = []


# Use a fixed sample to keep analysis fast
SAMPLE_SIZE_PER_CLASS = 1000


for class_name in classes:

    class_df = master[
        master["original_class"]
        == class_name
    ]


    sample_size = min(
        SAMPLE_SIZE_PER_CLASS,
        len(class_df)
    )


    sampled = class_df.sample(

        n=sample_size,

        random_state=42

    )


    print(
        f"\nAnalyzing intensity: "
        f"{class_name} "
        f"({sample_size} images)"
    )


    for _, row in sampled.iterrows():

        try:

            with Image.open(
                row["path"]
            ) as img:

                img = img.convert("L")

                array = np.array(
                    img,
                    dtype=np.float32
                )


            intensity_records.append({

                "original_class":
                    class_name,

                "filename":
                    row["filename"],

                "mean_intensity":
                    float(
                        array.mean()
                    ),

                "std_intensity":
                    float(
                        array.std()
                    ),

                "min_intensity":
                    float(
                        array.min()
                    ),

                "max_intensity":
                    float(
                        array.max()
                    )
            })


        except Exception as e:

            print(
                "Failed:",
                row["filename"]
            )


intensity_df = pd.DataFrame(
    intensity_records
)


intensity_df.to_csv(

    EDA_DIR /
    "intensity_statistics_sample.csv",

    index=False

)


# =========================================================
# 7. MEAN INTENSITY SUMMARY
# =========================================================

print("\n==========================================")
print("7. MEAN INTENSITY BY CLASS")
print("==========================================")


mean_summary = (

    intensity_df
    .groupby(
        "original_class"
    )
    [
        [
            "mean_intensity",
            "std_intensity"
        ]
    ]
    .mean()
    .round(3)

)


print(mean_summary)


mean_summary.to_csv(

    EDA_DIR /
    "mean_intensity_by_class.csv"

)


# =========================================================
# 8. MEAN INTENSITY BOXPLOT
# =========================================================

data_for_boxplot = []

labels_for_boxplot = []


for class_name in classes:

    values = intensity_df[
        intensity_df[
            "original_class"
        ]
        == class_name
    ]["mean_intensity"]


    data_for_boxplot.append(
        values.values
    )


    labels_for_boxplot.append(
        class_name
    )


plt.figure(
    figsize=(11, 6)
)


plt.boxplot(

    data_for_boxplot,

    tick_labels=labels_for_boxplot

)


plt.title(
    "Mean Image Intensity by Class"
)

plt.xlabel(
    "Original Class"
)

plt.ylabel(
    "Mean Pixel Intensity"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()


plt.savefig(

    EDA_DIR /
    "06_mean_intensity_boxplot.png",

    dpi=300

)

plt.close()


# =========================================================
# 9. INTENSITY HISTOGRAM
# =========================================================

plt.figure(
    figsize=(12, 7)
)


for class_name in classes:

    values = intensity_df[
        intensity_df[
            "original_class"
        ]
        == class_name
    ]["mean_intensity"]


    plt.hist(

        values,

        bins=40,

        alpha=0.5,

        label=class_name

    )


plt.title(
    "Distribution of Mean MRI Intensity"
)

plt.xlabel(
    "Mean Pixel Intensity"
)

plt.ylabel(
    "Frequency"
)

plt.legend()

plt.tight_layout()


plt.savefig(

    EDA_DIR /
    "07_mean_intensity_histogram.png",

    dpi=300

)

plt.close()


# =========================================================
# 10. NEAR-DUPLICATE GROUP ANALYSIS
# =========================================================

print("\n==========================================")
print("10. LEAKAGE GROUP ANALYSIS")
print("==========================================")


group_sizes = (

    master.groupby(
        "leakage_group"
    )
    .size()

)


near_groups = group_sizes[
    group_sizes > 1
]


print(
    "Near-duplicate groups:"
)

print(
    len(near_groups)
)


print(
    "\nImages in near-duplicate groups:"
)

print(
    near_groups.sum()
)


print(
    "\nLargest group:"
)

print(
    near_groups.max()
)


# =========================================================
# 11. NEAR-DUPLICATE IMAGES BY CLASS
# =========================================================

near_group_names = set(
    near_groups.index
)


near_duplicate_images = master[
    master["leakage_group"]
    .isin(
        near_group_names
    )
]


near_by_class = (

    near_duplicate_images[
        "original_class"
    ]
    .value_counts()

)


print(
    "\nNear-duplicate images by class:"
)

print(
    near_by_class
)


near_by_class.to_csv(

    EDA_DIR /
    "near_duplicate_images_by_class.csv"

)


# =========================================================
# 12. SAVE MAIN SUMMARY
# =========================================================

summary_lines = []


summary_lines.append(
    f"Master clean images: {len(master)}"
)

summary_lines.append(
    f"Primary images: {len(primary)}"
)

summary_lines.append(
    ""
)

summary_lines.append(
    "Original class distribution:"
)

summary_lines.append(
    class_counts.to_string()
)

summary_lines.append(
    ""
)

summary_lines.append(
    "Primary study distribution:"
)

summary_lines.append(
    study_counts.to_string()
)

summary_lines.append(
    ""
)

summary_lines.append(
    "Resolution by class:"
)

summary_lines.append(
    resolution_table.to_string()
)

summary_lines.append(
    ""
)

summary_lines.append(
    "Mode by class:"
)

summary_lines.append(
    mode_table.to_string()
)

summary_lines.append(
    ""
)

summary_lines.append(
    "Mean intensity summary:"
)

summary_lines.append(
    mean_summary.to_string()
)

summary_lines.append(
    ""
)

summary_lines.append(
    f"Near-duplicate groups: "
    f"{len(near_groups)}"
)

summary_lines.append(
    f"Images in near-duplicate groups: "
    f"{near_groups.sum()}"
)


with open(

    EDA_DIR /
    "EDA_summary.txt",

    "w",
    encoding="utf-8"

) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


# =========================================================
# FINISHED
# =========================================================

print("\n==========================================")
print("EDA COMPLETE")
print("==========================================")

print("\nResults saved inside:")

print(
    EDA_DIR
)