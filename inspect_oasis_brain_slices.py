from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("mgz_path")
args = parser.parse_args()

mgz_path = Path(args.mgz_path)
img = nib.load(str(mgz_path))
data = np.asarray(img.dataobj)
orientation = nib.aff2axcodes(img.affine)

print("shape:", data.shape)
print("dtype:", data.dtype)
print("orientation:", orientation)

if orientation[1] not in ("I", "S"):
    raise RuntimeError(f"Unexpected orientation: {orientation}")

areas = np.array([np.count_nonzero(data[:, i, :]) for i in range(data.shape[1])])
nonempty = np.where(areas > 0)[0]
first, last = int(nonempty.min()), int(nonempty.max())

print("nonzero axial range:", first, "to", last)
print("max-area slice:", int(np.argmax(areas)))

indices = np.linspace(
    int(round(first + 0.20 * (last-first))),
    int(round(first + 0.80 * (last-first))),
    12,
    dtype=int
)

out_dir = mgz_path.parent.parent / "inspection"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "axial_slice_montage.png"

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.ravel()

nz = data[data > 0]
vmax = float(np.percentile(nz, 99)) if len(nz) else 1.0

for ax, idx in zip(axes, indices):
    sl = np.rot90(data[:, idx, :])
    frac = (idx-first) / max(last-first, 1)
    ax.imshow(sl, cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(f"slice {idx}\nheight={frac:.2f} area={areas[idx]}")
    ax.axis("off")

fig.suptitle(f"{mgz_path.parent.parent.name} | brain.mgz | {orientation}")
fig.tight_layout()
fig.savefig(out_path, dpi=170, bbox_inches="tight")
plt.close(fig)

print("saved:", out_path)
