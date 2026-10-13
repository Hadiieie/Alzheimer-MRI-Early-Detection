from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "NEW MRI" / "OASIS_BATCH" / "oasis_validation_manifest.csv"
FREESURFER_ROOT = PROJECT_ROOT / "OASIS_FREESURFER"
OUTPUT_DIR = PROJECT_ROOT / "NEW MRI" / "OASIS_FREESURFER_BATCH"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUTPUT_DIR / "freesurfer_slice_extraction_summary.csv"

def subject_id_from_path(path):
    for parent in [path.parent] + list(path.parents):
        if parent.name.startswith("OAS1_") and "_MR" in parent.name:
            return parent.name
    return None

def extract_mid_axial(mgz_path, output_path):
    img = nib.load(str(mgz_path))
    data = np.asarray(img.dataobj)
    orientation = nib.aff2axcodes(img.affine)

    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D volume, got {data.shape}")

    if len(orientation) != 3 or orientation[1] not in ("I", "S"):
        raise RuntimeError(f"Unexpected orientation {orientation}")

    areas = np.array([np.count_nonzero(data[:, i, :]) for i in range(data.shape[1])])
    nonempty = np.where(areas > 0)[0]

    if len(nonempty) == 0:
        raise RuntimeError("No nonzero brain voxels found.")

    first = int(nonempty.min())
    last = int(nonempty.max())
    slice_index = int(round(first + 0.50 * (last - first)))

    sl = np.rot90(data[:, slice_index, :])
    if sl.dtype != np.uint8:
        sl = np.clip(sl, 0, 255).astype(np.uint8)
    else:
        sl = sl.copy()

    if not cv2.imwrite(str(output_path), sl):
        raise RuntimeError(f"Could not save {output_path}")

    return {
        "orientation": "".join(orientation),
        "volume_shape": "x".join(map(str, data.shape)),
        "first_nonempty_axial": first,
        "last_nonempty_axial": last,
        "selected_slice_index": slice_index,
        "selected_slice_area": int(areas[slice_index]),
        "slice_min": int(sl.min()),
        "slice_max": int(sl.max()),
    }

print("=" * 72)
print("OASIS FREESURFER BATCH MID-AXIAL EXTRACTION V2")
print("=" * 72)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found:\n{MANIFEST}")

manifest = pd.read_csv(MANIFEST)
brain_files = list(FREESURFER_ROOT.rglob("brain.mgz"))

subject_to_brain = {}
for path in brain_files:
    sid = subject_id_from_path(path)
    if sid:
        subject_to_brain.setdefault(sid, []).append(path)

print("\nSubjects requested:", len(manifest))
print("brain.mgz files discovered:", len(brain_files))
print("Subjects mapped:", len(subject_to_brain))

records = []

for i, row in manifest.iterrows():
    sid = str(row["subject_id"]).strip()
    print(f"\n[{i+1:02d}/{len(manifest):02d}] {sid}")

    matches = subject_to_brain.get(sid, [])

    if len(matches) == 0:
        print("STATUS: MISSING")
        records.append({
            "subject_id": sid,
            "status": "MISSING",
            "brain_mgz": "",
            "output_png": "",
            "error": "brain.mgz not found"
        })
        continue

    if len(matches) > 1:
        print("STATUS: ERROR - multiple brain.mgz files")
        records.append({
            "subject_id": sid,
            "status": "ERROR",
            "brain_mgz": " | ".join(map(str, matches)),
            "output_png": "",
            "error": "Multiple brain.mgz matches found"
        })
        continue

    mgz_path = matches[0]
    output_png = OUTPUT_DIR / f"{sid}.png"

    try:
        info = extract_mid_axial(mgz_path, output_png)
        records.append({
            "subject_id": sid,
            "status": "PASS",
            "brain_mgz": str(mgz_path),
            "output_png": str(output_png),
            "error": "",
            **info
        })
        print("STATUS: PASS")
        print("Selected slice:", info["selected_slice_index"])
        print("Orientation:", info["orientation"])
    except Exception as e:
        print("STATUS: ERROR")
        print("Reason:", e)
        records.append({
            "subject_id": sid,
            "status": "ERROR",
            "brain_mgz": str(mgz_path),
            "output_png": "",
            "error": str(e)
        })

summary = pd.DataFrame(records)
summary.to_csv(SUMMARY_CSV, index=False)

print("\n" + "=" * 72)
print("BATCH EXTRACTION SUMMARY")
print("=" * 72)
print(summary["status"].value_counts(dropna=False))

print("\nSaved summary:")
print(SUMMARY_CSV)
print("\nSaved PNG folder:")
print(OUTPUT_DIR)
print("=" * 72)
