"""OCT image preprocessing pipeline."""

import cv2
import numpy as np
from utils.config import IMAGE_SIZE


def median_denoise(img: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(img, 3)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def histogram_equalize(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.equalizeHist(img)


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def edge_enhance(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    if len(img.shape) == 3:
        enhanced = img.copy().astype(np.float32)
        edge_mask = edges.astype(np.float32) / 255.0
        for c in range(3):
            enhanced[:, :, c] += edge_mask * 0.15
        return np.clip(enhanced, 0, 1)
    return np.clip(img.astype(np.float32) + edges.astype(np.float32) / 255.0 * 0.15, 0, 1)


def preprocess_oct(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal inference preprocessing that matches training normalization.
    Returns (display_rgb_0_255, model_input_chw_normalized).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    img = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = display.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (tensor - mean) / std
    chw = np.transpose(tensor, (2, 0, 1))
    return display, chw
