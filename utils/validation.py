"""Retinal OCT image validation."""

import cv2
import numpy as np
from utils.config import INVALID_MSG


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _row_banding_score(gray: np.ndarray) -> float:
    """OCT B-scans show horizontal layer banding in row-mean profiles."""
    row_means = gray.mean(axis=1)
    row_std = np.std(np.diff(row_means))
    profile_std = np.std(row_means)
    if profile_std < 1e-3:
        return 0.0
    banding = row_std / (profile_std + 1e-6)
    return float(np.clip(banding * 2.5, 0, 1))


def _oct_heuristic_score(image_bgr: np.ndarray) -> float:
    """Score 0-1 indicating likelihood of retinal OCT scan."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0

    h, w = image_bgr.shape[:2]
    if h < 64 or w < 64:
        return 0.0

    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if len(image_bgr.shape) == 3
        else image_bgr
    )
    gray = cv2.resize(gray, (224, 224))

    sharpness = _laplacian_variance(gray)
    if sharpness < 20:
        return 0.0

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gx = np.abs(sobelx).mean()
    gy = np.abs(sobely).mean()
    layer_ratio = gy / (gx + 1e-6)

    if len(image_bgr.shape) == 3:
        b, g, r = cv2.split(image_bgr)
        color_std = np.std(r.astype(float) - g.astype(float)) + np.std(
            g.astype(float) - b.astype(float)
        )
        grayscale_score = 1.0 if color_std < 35 else max(0, 1 - color_std / 100)
    else:
        grayscale_score = 1.0

    banding = _row_banding_score(gray)

    score = 0.0
    if 3.8 < entropy < 8.0:
        score += 0.25
    if layer_ratio > 0.65:
        score += 0.25
    if grayscale_score > 0.4:
        score += 0.15
    if sharpness > 35:
        score += 0.1
    score += 0.25 * banding

    if len(image_bgr.shape) == 3:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = hsv[:, :, 1].mean()
        if sat_mean > 90:
            score *= 0.25

    return min(1.0, score)


def validate_oct_image(image_bgr: np.ndarray) -> tuple[bool, str]:
    """
    Validate retinal OCT scan.
    Returns (is_valid, message).
    """
    try:
        if image_bgr is None:
            return False, INVALID_MSG
        if image_bgr.size == 0:
            return False, INVALID_MSG

        # Corrupted / unreadable
        if np.isnan(image_bgr).any() or np.isinf(image_bgr).any():
            return False, INVALID_MSG

        gray = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if len(image_bgr.shape) == 3
            else image_bgr
        )
        if _laplacian_variance(gray) < 15:
            return False, INVALID_MSG

        score = _oct_heuristic_score(image_bgr)
        if score < 0.38:
            return False, INVALID_MSG

        return True, "Valid retinal OCT scan."
    except Exception:
        return False, INVALID_MSG
