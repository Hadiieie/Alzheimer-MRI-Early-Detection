from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inference_core import preprocess_image_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_RESULTS = PROJECT_ROOT / "results" / "OASIS_BATCH_VALIDATION" / "oasis_batch_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "OASIS_MASK_DIAGNOSTICS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BORDER = 5

def zero_border(img, border=5):
    out = img.copy()
    out[:border, :] = 0
    out[-border:, :] = 0
    out[:, :border] = 0
    out[:, -border:] = 0
    return out

def diagnose_case(image_path, subject_id):
    _, standardized, _ = preprocess_image_path(image_path)
    h, w = standardized.shape

    blurred = cv2.GaussianBlur(standardized, (5, 5), 0)
    otsu, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = zero_border(binary, BORDER)

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    cleaned = zero_border(cleaned, BORDER)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    rows = []
    valid = []
    center = np.array([w / 2.0, h / 2.0])

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        ratio = area / float(h * w)

        touches = (
            x <= BORDER or y <= BORDER or
            x + cw >= w - BORDER or y + ch >= h - BORDER
        )

        reasons = []
        if area < 1000:
            reasons.append("AREA_LT_1000")
        if ratio < 0.15:
            reasons.append("AREA_RATIO_LT_0.15")
        if ratio > 0.85:
            reasons.append("AREA_RATIO_GT_0.85")
        if touches:
            reasons.append("TOUCHES_EDGE")

        centroid = np.array(centroids[label])
        norm_dist = np.linalg.norm(centroid - center) / (np.linalg.norm(center) + 1e-8)
        score = ratio - 0.20 * norm_dist

        if not reasons:
            valid.append((label, score))

        rows.append({
            "subject_id": subject_id,
            "label": label,
            "area": area,
            "area_ratio": ratio,
            "x": x,
            "y": y,
            "width": cw,
            "height": ch,
            "touches_edge": touches,
            "reject_reasons": "|".join(reasons),
            "score": score,
        })

    largest = max(rows, key=lambda r: r["area"]) if rows else None

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(standardized, cmap="gray")
    axes[0].set_title("Standardized")
    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("Gaussian")
    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title(f"Otsu T={otsu:.1f}")
    axes[3].imshow(cleaned, cmap="gray")
    axes[3].set_title("Cleaned")
    axes[4].imshow(standardized, cmap="gray")
    axes[4].set_title("Components")

    for ax in axes:
        ax.axis("off")

    for r in rows:
        rect = plt.Rectangle(
            (r["x"], r["y"]), r["width"], r["height"],
            fill=False, linewidth=1.5
        )
        axes[4].add_patch(rect)
        axes[4].text(
            r["x"], max(0, r["y"] - 2),
            f"L{r['label']} r={r['area_ratio']:.3f}",
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    reason = largest["reject_reasons"] if largest else "NO_COMPONENTS"
    ratio = largest["area_ratio"] if largest else 0.0

    fig.suptitle(
        f"{subject_id} | valid={len(valid)} | largest_ratio={ratio:.3f} | {reason}"
    )
    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"{subject_id}_diagnostic.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "subject_id": subject_id,
        "image_path": str(image_path),
        "otsu_threshold": float(otsu),
        "num_components": max(0, num_labels - 1),
        "num_valid_components": len(valid),
        "largest_component_ratio": ratio,
        "largest_component_reject_reasons": reason,
        "diagnostic_image": str(fig_path),
    }, rows

print("=" * 72)
print("OASIS MASK FAILURE DIAGNOSTIC")
print("=" * 72)

if not BATCH_RESULTS.exists():
    raise FileNotFoundError(f"Missing batch results:\n{BATCH_RESULTS}")

df = pd.read_csv(BATCH_RESULTS)
failed = df[df["status"].astype(str).str.upper() == "ERROR"].copy()

print("\nFailed cases:", len(failed))

summary = []
components = []

for i, (_, row) in enumerate(failed.iterrows(), start=1):
    sid = str(row["subject_id"])
    img = Path(str(row["image_path"]))

    print(f"\n[{i:02d}/{len(failed):02d}] {sid}")

    try:
        s, c = diagnose_case(img, sid)
        summary.append(s)
        components.extend(c)

        print("Otsu threshold:", round(s["otsu_threshold"], 3))
        print("Components:", s["num_components"])
        print("Valid components:", s["num_valid_components"])
        print("Largest ratio:", round(s["largest_component_ratio"], 4))
        print("Reject reason:", s["largest_component_reject_reasons"])

    except Exception as e:
        print("Diagnostic error:", e)
        summary.append({
            "subject_id": sid,
            "image_path": str(img),
            "diagnostic_error": str(e),
        })

summary_df = pd.DataFrame(summary)
summary_csv = OUTPUT_DIR / "mask_failure_diagnostics.csv"
summary_df.to_csv(summary_csv, index=False)

components_csv = OUTPUT_DIR / "all_component_diagnostics.csv"
pd.DataFrame(components).to_csv(components_csv, index=False)

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

if "largest_component_reject_reasons" in summary_df.columns:
    print(
        summary_df["largest_component_reject_reasons"]
        .fillna("NONE")
        .value_counts()
    )

print("\nSaved summary:")
print(summary_csv)

print("\nSaved component details:")
print(components_csv)

print("\nDiagnostic PNGs:")
print(OUTPUT_DIR)

print("\nNo model, threshold, or Brain Mask V4 settings were changed.")
print("=" * 72)
