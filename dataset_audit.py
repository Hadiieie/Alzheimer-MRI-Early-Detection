from pathlib import Path
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
from PIL import Image
import pandas as pd
import hashlib


# -----------------------------
# Project paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
ALZHEIMER_DIR = DATA_DIR / "ALZHEIMER"
NORMAL_DIR = DATA_DIR / "NORMAL"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------
# Function for file hash
# Used to detect duplicates
# -----------------------------

def calculate_hash(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        for chunk in iter(lambda: f.read(4096), b""):

            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# -----------------------------
# Dataset information
# -----------------------------

records = []

classes = {

    "Normal": NORMAL_DIR,

    "Alzheimer": ALZHEIMER_DIR

}


# -----------------------------
# Scan all images
# -----------------------------

for class_name, folder_path in classes.items():

    print("\nScanning:", class_name)

    for file_path in folder_path.rglob("*"):

        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:

            continue

        try:

            with Image.open(file_path) as img:

                width, height = img.size

                mode = img.mode

                image_hash = calculate_hash(file_path)

                status = "OK"

        except Exception as e:

            width = None

            height = None

            mode = None

            image_hash = None

            status = "CORRUPTED"

        records.append({

            "filename": file_path.name,

            "class": class_name,

            "width": width,

            "height": height,

            "mode": mode,

            "status": status,

            "hash": image_hash

        })


# -----------------------------
# Create dataframe
# -----------------------------

df = pd.DataFrame(records)


# -----------------------------
# Print results
# -----------------------------

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print(f"\nNumber of images: {len(df)}")  
print(df.columns)


print("\nImage status:")

print(df["status"].value_counts())


print("\nImage dimensions:")

print(
    df.groupby(
        ["width", "height"]
    ).size()
)


# -----------------------------
# Detect duplicates
# -----------------------------

duplicates = df[
    df.duplicated(
        subset=["hash"],
        keep=False
    )
]

print("\nDuplicate images found:")

print(len(duplicates))


# -----------------------------
# Save metadata
# -----------------------------

metadata_path = RESULTS_DIR / "metadata.csv"

df.to_csv(
    metadata_path,
    index=False
)


duplicates_path = RESULTS_DIR / "duplicates.csv"

duplicates.to_csv(
    duplicates_path,
    index=False
)


print("\nFiles saved:")

print(metadata_path)

print(duplicates_path)

print("\nDataset audit completed successfully.")


