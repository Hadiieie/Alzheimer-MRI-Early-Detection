from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

MASTER_FILE = RESULTS_DIR / "clean_4class_metadata.csv"

SSIM_FILE = RESULTS_DIR / "same_class_ssim_report.csv"

OUTPUT_FILE = (
    RESULTS_DIR /
    "clean_4class_grouped_metadata.csv"
)

PRIMARY_OUTPUT_FILE = (
    RESULTS_DIR /
    "primary_grouped_metadata.csv"
)

GROUP_REPORT_FILE = (
    RESULTS_DIR /
    "near_duplicate_groups_098.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

master = pd.read_csv(MASTER_FILE)

ssim_report = pd.read_csv(SSIM_FILE)


print("\n========================================")
print("BUILDING LEAKAGE-SAFE IMAGE GROUPS")
print("========================================\n")


print("Master dataset size:")
print(len(master))


print("\nTotal SSIM pairs:")
print(len(ssim_report))


# =========================================================
# USE ONLY EXTREMELY SIMILAR PAIRS
# =========================================================

strong_pairs = ssim_report[
    ssim_report["ssim"] >= 0.98
].copy()


print("\nPairs with SSIM >= 0.98:")
print(len(strong_pairs))


print("\nStrong pairs by class:")
print(
    strong_pairs[
        "original_class"
    ].value_counts()
)


# =========================================================
# UNION-FIND
# =========================================================

parent = {}


def find(x):

    if x not in parent:
        parent[x] = x

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a


# =========================================================
# CONNECT NEAR-DUPLICATE IMAGES
# =========================================================

for _, row in strong_pairs.iterrows():

    path_a = row["path_a"]
    path_b = row["path_b"]

    union(path_a, path_b)


# =========================================================
# BUILD CONNECTED COMPONENTS
# =========================================================

components = {}


for image_path in parent:

    root = find(image_path)

    components.setdefault(
        root,
        []
    ).append(image_path)


print("\nNear-duplicate connected groups:")
print(len(components))


group_sizes = [
    len(images)
    for images in components.values()
]


print("\nImages involved in strong groups:")
print(sum(group_sizes))


print("\nLargest near-duplicate group:")
print(
    max(group_sizes)
    if group_sizes
    else 0
)


# =========================================================
# CREATE GROUP MAP
# =========================================================

group_map = {}

group_records = []


group_counter = 1


for root, images in components.items():

    group_id = (
        f"NEAR_DUP_{group_counter:05d}"
    )

    for image_path in images:

        group_map[image_path] = group_id

        group_records.append({

            "group_id": group_id,

            "path": image_path

        })

    group_counter += 1


# =========================================================
# GIVE EVERY IMAGE A LEAKAGE GROUP
# =========================================================

leakage_groups = []

singleton_counter = 1


for _, row in master.iterrows():

    path = row["path"]

    if path in group_map:

        leakage_groups.append(
            group_map[path]
        )

    else:

        leakage_groups.append(
            f"SINGLE_{singleton_counter:06d}"
        )

        singleton_counter += 1


master["leakage_group"] = leakage_groups


# =========================================================
# CHECK GROUP CLASS CONSISTENCY
# =========================================================

class_counts_per_group = (
    master.groupby(
        "leakage_group"
    )["original_class"]
    .nunique()
)


bad_groups = class_counts_per_group[
    class_counts_per_group > 1
]


print("\nGroups containing multiple classes:")
print(len(bad_groups))


# =========================================================
# GROUP SIZE REPORT
# =========================================================

group_summary = (

    master.groupby(
        "leakage_group"
    )

    .agg(

        group_size=(
            "path",
            "count"
        ),

        original_class=(
            "original_class",
            "first"
        )

    )

    .reset_index()
)


near_group_summary = group_summary[
    group_summary[
        "group_size"
    ] > 1
].copy()


near_group_summary = (
    near_group_summary
    .sort_values(
        "group_size",
        ascending=False
    )
)


# =========================================================
# PRIMARY DATASET
# =========================================================

primary = master[
    master["original_class"].isin(
        [
            "NonDemented",
            "VeryMildDemented",
            "MildDemented"
        ]
    )
].copy()


primary["study_group"] = (
    primary["original_class"]
    .map(
        {
            "NonDemented":
                "Normal",

            "VeryMildDemented":
                "Early_Demented",

            "MildDemented":
                "Early_Demented"
        }
    )
)


# =========================================================
# SAVE FILES
# =========================================================

master.to_csv(
    OUTPUT_FILE,
    index=False
)


primary.to_csv(
    PRIMARY_OUTPUT_FILE,
    index=False
)


near_group_summary.to_csv(
    GROUP_REPORT_FILE,
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n========================================")
print("FINAL LEAKAGE-GROUP SUMMARY")
print("========================================")


print("\nTotal clean images:")
print(len(master))


print("\nImages in primary dataset:")
print(len(primary))


print("\nNear-duplicate groups:")
print(len(near_group_summary))


print("\nImages belonging to near-duplicate groups:")
print(
    near_group_summary[
        "group_size"
    ].sum()
)


print("\nLargest group size:")
print(
    near_group_summary[
        "group_size"
    ].max()
)


print("\nGroups containing multiple classes:")
print(len(bad_groups))


print("\nFiles saved successfully:")
print(OUTPUT_FILE)
print(PRIMARY_OUTPUT_FILE)
print(GROUP_REPORT_FILE)