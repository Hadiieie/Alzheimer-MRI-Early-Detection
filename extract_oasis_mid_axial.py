from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Extract a deterministic mid-axial slice from OASIS FreeSurfer brain.mgz."
    )
    parser.add_argument("mgz_path", help="Path to OASIS FreeSurfer brain.mgz")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Default: <subject>/inspection/<subject>_mid_axial.png",
    )
    args = parser.parse_args()

    mgz_path = Path(args.mgz_path)

    if not mgz_path.exists():
        raise FileNotFoundError(f"brain.mgz not found:\n{mgz_path}")

    img = nib.load(str(mgz_path))
    data = np.asarray(img.dataobj)
    orientation = nib.aff2axcodes(img.affine)

    if data.ndim != 3:
        raise RuntimeError(f"Expected a 3D MRI volume, got shape {data.shape}")

    # OASIS FreeSurfer volume observed as LIA:
    # axis 1 is inferior-superior, so axial slices are data[:, index, :].
    if len(orientation) != 3 or orientation[1] not in ("I", "S"):
        raise RuntimeError(
            f"Unexpected orientation {orientation}. "
            "Expected the second voxel axis to be inferior/superior."
        )

    areas = np.array(
        [np.count_nonzero(data[:, i, :]) for i in range(data.shape[1])]
    )

    nonempty = np.where(areas > 0)[0]

    if len(nonempty) == 0:
        raise RuntimeError("No nonzero brain voxels found.")

    first = int(nonempty.min())
    last = int(nonempty.max())

    # Frozen deterministic rule:
    # choose the slice at 50% of occupied brain height.
    slice_index = int(round(first + 0.50 * (last - first)))

    sl = data[:, slice_index, :]

    # Same display orientation used in the inspection montage.
    sl = np.rot90(sl)

    # Preserve FreeSurfer intensities; only convert safely to uint8 for PNG.
    if sl.dtype != np.uint8:
        sl = np.clip(sl, 0, 255).astype(np.uint8)
    else:
        sl = sl.copy()

    subject_id = mgz_path.parent.parent.name

    if args.output is None:
        out_dir = mgz_path.parent.parent / "inspection"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{subject_id}_mid_axial.png"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), sl)

    if not ok:
        raise RuntimeError(f"Could not save PNG:\n{output_path}")

    print("=" * 70)
    print("OASIS MID-AXIAL EXTRACTION")
    print("=" * 70)
    print("Subject:", subject_id)
    print("Input:", mgz_path)
    print("Shape:", data.shape)
    print("Orientation:", orientation)
    print("Occupied axial range:", first, "to", last)
    print("Frozen slice rule: 50% of occupied brain height")
    print("Selected slice index:", slice_index)
    print("Selected slice nonzero area:", int(areas[slice_index]))
    print("Output min:", int(sl.min()))
    print("Output max:", int(sl.max()))
    print("Saved:")
    print(output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
