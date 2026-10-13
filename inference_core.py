from pathlib import Path
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

FROZEN_FEATURES = [
    "brain_area","brain_ratio","brain_perimeter","compactness","bbox_aspect_ratio",
    "mean_intensity","std_intensity","median_intensity","entropy","p10","p25","p75","p90",
    "edge_density","central_mean","central_std","central_median","central_entropy",
    "central_dark_ratio","central_bright_ratio","left_mean","right_mean","left_dark_ratio",
    "right_dark_ratio","intensity_asymmetry","dark_ratio_asymmetry","glcm_contrast",
    "glcm_homogeneity","glcm_energy","glcm_correlation"
]

def resize_with_padding(image, target_size=220):
    h, w = image.shape
    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def preprocess_from_raw(raw_image):
    if raw_image is None or raw_image.ndim != 2:
        raise ValueError("Expected a readable grayscale 2D MRI image.")
    standardized = resize_with_padding(raw_image, 220)
    background_mask = standardized == 0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(standardized)
    filtered = cv2.medianBlur(enhanced, 3)
    filtered[background_mask] = 0
    return standardized, filtered

def preprocess_image_path(image_path):
    image_path = Path(image_path)
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not read image: {image_path}")
    standardized, processed = preprocess_from_raw(raw)
    return raw, standardized, processed

def zero_border(image, border=5):
    output = image.copy()
    output[:border, :] = 0
    output[-border:, :] = 0
    output[:, :border] = 0
    output[:, -border:] = 0
    return output

def create_brain_mask_v4(standardized_raw):
    h, w = standardized_raw.shape
    border = 5

    blurred = cv2.GaussianBlur(standardized_raw, (5, 5), 0)
    otsu_value, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = zero_border(binary, border)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = zero_border(cleaned, border)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    if num_labels <= 1:
        return None, float(otsu_value), "NO_COMPONENT"

    image_center = np.array([w / 2.0, h / 2.0])
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_ratio = area / float(h * w)

        if area < 1000:
            continue
        if area_ratio < 0.15 or area_ratio > 0.85:
            continue

        touches_edge = (
            x <= border or y <= border or
            x + cw >= w - border or y + ch >= h - border
        )
        if touches_edge:
            continue

        centroid = np.array(centroids[label])
        distance = np.linalg.norm(centroid - image_center)
        normalized_distance = distance / (np.linalg.norm(image_center) + 1e-8)
        score = area_ratio - 0.20 * normalized_distance
        candidates.append({"label": label, "score": score})

    if not candidates:
        return None, float(otsu_value), "NO_VALID_COMPONENT"

    best_label = max(candidates, key=lambda item: item["score"])["label"]

    component_mask = np.zeros_like(standardized_raw, dtype=np.uint8)
    component_mask[labels == best_label] = 255

    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, float(otsu_value), "NO_CONTOUR"

    contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(standardized_raw, dtype=np.uint8)
    cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    final_mask = zero_border(final_mask, border)

    brain_ratio = float(np.mean(final_mask > 0))
    if brain_ratio < 0.20 or brain_ratio > 0.85:
        return None, float(otsu_value), "FAILED_RATIO_QC"

    return final_mask, float(otsu_value), "PASS"

def get_central_region(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return (
        int(x + 0.25 * w),
        int(y + 0.25 * h),
        int(x + 0.75 * w),
        int(y + 0.75 * h),
    )

def entropy_from_pixels(pixels):
    if len(pixels) == 0:
        return np.nan
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    probabilities = hist / hist.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))

def extract_glcm_features(region):
    quantized = (
        region.astype(np.float32) / 256.0 * 16.0
    ).astype(np.uint8)
    quantized = np.clip(quantized, 0, 15)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=16,
        symmetric=True,
        normed=True,
    )

    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
    }

def extract_features(processed_image, brain_mask):
    brain_pixels = processed_image[brain_mask > 0]
    if len(brain_pixels) == 0:
        return None

    contours, _ = cv2.findContours(
        brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    brain_area = float(np.sum(brain_mask > 0))
    brain_ratio = brain_area / float(brain_mask.size)
    brain_perimeter = float(cv2.arcLength(largest_contour, True))
    compactness = (
        4.0 * np.pi * brain_area / (brain_perimeter ** 2)
        if brain_perimeter > 0 else np.nan
    )

    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox_aspect_ratio = float(w) / float(h) if h > 0 else np.nan

    mean_intensity = float(np.mean(brain_pixels))
    std_intensity = float(np.std(brain_pixels))
    median_intensity = float(np.median(brain_pixels))
    entropy = entropy_from_pixels(brain_pixels)
    p10 = float(np.percentile(brain_pixels, 10))
    p25 = float(np.percentile(brain_pixels, 25))
    p75 = float(np.percentile(brain_pixels, 75))
    p90 = float(np.percentile(brain_pixels, 90))

    edges = cv2.Canny(processed_image, 50, 150)
    edge_pixels_inside_brain = int(np.sum((edges > 0) & (brain_mask > 0)))
    mask_pixels = int(np.sum(brain_mask > 0))
    edge_density = edge_pixels_inside_brain / float(mask_pixels)

    central = get_central_region(brain_mask)
    if central is None:
        return None
    x1, y1, x2, y2 = central

    central_image = processed_image[y1:y2, x1:x2]
    central_mask = brain_mask[y1:y2, x1:x2]
    central_pixels = central_image[central_mask > 0]
    if len(central_pixels) == 0:
        return None

    central_mean = float(np.mean(central_pixels))
    central_std = float(np.std(central_pixels))
    central_median = float(np.median(central_pixels))
    central_entropy = entropy_from_pixels(central_pixels)
    central_dark_ratio = float(np.mean(central_pixels <= 77))
    central_bright_ratio = float(np.mean(central_pixels >= 179))

    mid_x = central_image.shape[1] // 2
    left_image = central_image[:, :mid_x]
    right_image = central_image[:, mid_x:]
    left_mask = central_mask[:, :mid_x]
    right_mask = central_mask[:, mid_x:]
    left_pixels = left_image[left_mask > 0]
    right_pixels = right_image[right_mask > 0]

    if len(left_pixels) == 0 or len(right_pixels) == 0:
        return None

    left_mean = float(np.mean(left_pixels))
    right_mean = float(np.mean(right_pixels))
    left_dark_ratio = float(np.mean(left_pixels <= 77))
    right_dark_ratio = float(np.mean(right_pixels <= 77))

    intensity_asymmetry = float(
        abs(left_mean - right_mean) /
        (((left_mean + right_mean) / 2.0) + 1e-8)
    )
    dark_ratio_asymmetry = float(abs(left_dark_ratio - right_dark_ratio))

    glcm_features = extract_glcm_features(central_image)

    features = {
        "brain_area": brain_area,
        "brain_ratio": brain_ratio,
        "brain_perimeter": brain_perimeter,
        "compactness": compactness,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "median_intensity": median_intensity,
        "entropy": entropy,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "edge_density": edge_density,
        "central_mean": central_mean,
        "central_std": central_std,
        "central_median": central_median,
        "central_entropy": central_entropy,
        "central_dark_ratio": central_dark_ratio,
        "central_bright_ratio": central_bright_ratio,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_dark_ratio": left_dark_ratio,
        "right_dark_ratio": right_dark_ratio,
        "intensity_asymmetry": intensity_asymmetry,
        "dark_ratio_asymmetry": dark_ratio_asymmetry,
    }
    features.update(glcm_features)
    return features

def extract_single_mri_features(image_path):
    raw, standardized_raw, processed_image = preprocess_image_path(image_path)
    brain_mask, otsu_threshold, mask_status = create_brain_mask_v4(standardized_raw)

    if brain_mask is None:
        raise RuntimeError(f"Brain Mask V4 failed: {mask_status}")

    if processed_image.shape != brain_mask.shape:
        raise RuntimeError(
            f"Shape mismatch: processed={processed_image.shape}, "
            f"mask={brain_mask.shape}"
        )

    features = extract_features(processed_image, brain_mask)
    if features is None:
        raise RuntimeError("30-feature extraction failed.")

    missing = [name for name in FROZEN_FEATURES if name not in features]
    if missing:
        raise RuntimeError("Missing frozen features: " + ", ".join(missing))

    vector = np.array([features[name] for name in FROZEN_FEATURES], dtype=np.float64)

    if np.isnan(vector).any():
        raise RuntimeError("Extracted feature vector contains NaN.")
    if np.isinf(vector).any():
        raise RuntimeError("Extracted feature vector contains infinity.")

    return {
        "features": features,
        "feature_vector": vector,
        "raw": raw,
        "standardized_raw": standardized_raw,
        "processed_image": processed_image,
        "brain_mask": brain_mask,
        "otsu_threshold": float(otsu_threshold),
        "mask_status": mask_status,
    }
