from pathlib import Path
from PIL import Image
import pandas as pd
import hashlib


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================
# ORIGINAL DATASET FOLDERS
# عدل الاسم فقط إذا اسم الفولدر عندك مختلف حرفياً
# =========================================================

CLASS_FOLDERS = {

    "NonDemented": 
        PROJECT_ROOT / "NORMAL" / "archive" / "Data" / "NonDemented",

    "VeryMildDemented": 
        PROJECT_ROOT / "ALZHEIMER" / "archive 3" / "Data" / "VeryMildDemented",


     "MildDemented": 
        PROJECT_ROOT / "ALZHEIMER" / "archive" / "Data" / "MildDemented",


    "ModerateDemented":
        PROJECT_ROOT / "ALZHEIMER" / "archive 2" / "Data" / "ModerateDemented",

}


# =========================================================
# HASH FUNCTION
# =========================================================

def calculate_hash(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        for chunk in iter(lambda: f.read(4096), b""):

            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# =========================================================
# SCAN ORIGINAL FOUR CLASSES
# =========================================================

records = []


for original_class, folder_path in CLASS_FOLDERS.items():

    print(f"\nScanning: {original_class}")

    if not folder_path.exists():

        print(f"ERROR: Folder not found:")
        print(folder_path)

        continue


    for file_path in folder_path.iterdir():

        if file_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]:
            continue


        try:

            with Image.open(file_path) as img:

                width, height = img.size
                mode = img.mode

                image_hash = calculate_hash(file_path)

                status = "OK"


        except Exception:

            width = None
            height = None
            mode = None
            image_hash = None

            status = "CORRUPTED"


        records.append({

            "filename": file_path.name,

            "original_class": original_class,

            "path": str(file_path),

            "width": width,

            "height": height,

            "mode": mode,

            "status": status,

            "hash": image_hash
        })


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(records, columns=[
    "filename", "original_class", "path", 
    "width", "height", "mode", "status", "hash"
])


print("\n========================================")
print("ORIGINAL FOUR-CLASS DATASET AUDIT")
print("========================================")


print("\nTOTAL IMAGES")
print("------------")

print(len(df))


print("\nORIGINAL CLASS DISTRIBUTION")
print("---------------------------")

print(
    df["original_class"]
    .value_counts()
)


print("\nIMAGE STATUS")
print("------------")

print(
    df["status"]
    .value_counts()
)


# =========================================================
# EXACT DUPLICATES
# =========================================================

duplicate_rows = df[
    df.duplicated(
        subset=["hash"],
        keep=False
    )
].copy()


print("\nDUPLICATE ROWS INVOLVED")
print("-----------------------")

print(len(duplicate_rows))


print("\nUNIQUE DUPLICATE GROUPS")
print("-----------------------")

print(
    duplicate_rows["hash"]
    .nunique()
)


# =========================================================
# CROSS-ORIGINAL-CLASS DUPLICATES
# =========================================================

hash_class_counts = (
    df.groupby("hash")["original_class"]
    .nunique()
)


conflict_hashes = hash_class_counts[
    hash_class_counts > 1
].index


cross_original_duplicates = df[
    df["hash"].isin(conflict_hashes)
].copy()


print("\nCROSS-ORIGINAL-CLASS DUPLICATE ROWS")
print("-----------------------------------")

print(len(cross_original_duplicates))


print("\nCROSS-ORIGINAL-CLASS DUPLICATE GROUPS")
print("-------------------------------------")

print(len(conflict_hashes))


# =========================================================
# REMOVE LABEL-CONFLICT IMAGES FROM CLEAN MANIFEST
# =========================================================

safe_df = df[
    ~df["hash"].isin(conflict_hashes)
].copy()


# =========================================================
# REMOVE REDUNDANT COPIES
# KEEP ONE IMAGE PER HASH
# =========================================================

clean_df = safe_df.drop_duplicates(
    subset=["hash"],
    keep="first"
).copy()


# =========================================================
# CREATE STUDY GROUP
# =========================================================

study_group_map = {

    "NonDemented":
        "Normal",

    "VeryMildDemented":
        "Early_Demented",

    "MildDemented":
        "Early_Demented",

    "ModerateDemented":
        "Secondary_Only"
}


clean_df["study_group"] = (
    clean_df["original_class"]
    .map(study_group_map)
)


# =========================================================
# PRIMARY STUDY DATASET
# Exclude Moderate from PRIMARY early-detection experiment
# =========================================================

primary_df = clean_df[
    clean_df["study_group"] != "Secondary_Only"
].copy()


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n========================================")
print("CLEAN ORIGINAL DATASET")
print("========================================")


print("\nClean unique images:")

print(len(clean_df))


print("\nClean original-class distribution:")

print(
    clean_df["original_class"]
    .value_counts()
)


print("\nPRIMARY EARLY-DETECTION DATASET")

print("--------------------------------")


print(
    primary_df["study_group"]
    .value_counts()
)


print("\nOriginal classes inside primary dataset:")

print(
    primary_df["original_class"]
    .value_counts()
)


# =========================================================
# SAVE FILES
# =========================================================

df.to_csv(
    RESULTS_DIR / "original_4class_metadata.csv",
    index=False
)


duplicate_rows.to_csv(
    RESULTS_DIR / "original_4class_duplicates.csv",
    index=False
)


cross_original_duplicates.to_csv(
    RESULTS_DIR / "cross_original_class_duplicates.csv",
    index=False
)


clean_df.to_csv(
    RESULTS_DIR / "clean_4class_metadata.csv",
    index=False
)


primary_df.to_csv(
    RESULTS_DIR / "primary_early_detection_metadata.csv",
    index=False
)


print("\n========================================")
print("FILES CREATED SUCCESSFULLY")
print("========================================")


print("\nSaved inside:")

print(RESULTS_DIR)