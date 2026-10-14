"""Backend services for VisionCare API."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch

from database.db import get_db
from utils.gradcam import generate_gradcam
from utils.inference import get_predictor
from utils.preprocess import preprocess_oct
from utils.reports import generate_english_pdf, generate_tamil_pdf


class VisionCareService:
    def __init__(self):
        self.db = get_db()

    def gradcam_for_result(self, raw_image: np.ndarray, result: dict) -> tuple[np.ndarray, np.ndarray]:
        predictor = get_predictor()
        _, chw = preprocess_oct(raw_image)
        tensor = torch.from_numpy(chw).float().unsqueeze(0)
        try:
            class_idx = list(result["probabilities"].keys()).index(result["label"])
        except (ValueError, KeyError):
            class_idx = 0
        try:
            # generate_gradcam returns RGB overlay (display_image is already RGB from preprocess_oct)
            overlay_rgb, cam = generate_gradcam(predictor.efficient, tensor, result["display_image"], class_idx)
            # Convert RGB→BGR for consistency with OpenCV image handling in main.py
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            return overlay_bgr, cam
        except Exception as e:
            # Fallback: create simple heatmap
            print(f"⚠ GradCAM failed, using fallback: {e}")
            display_rgb = result["display_image"]  # Already RGB from preprocess_oct
            gray = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2GRAY)
            cam = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # Returns BGR
            overlay = cv2.addWeighted(display_bgr, 0.2, cam, 0.8, 0)  # BGR + BGR = BGR
            return overlay, gray

    def render_segmentation(self, image: np.ndarray, result: dict) -> tuple[np.ndarray, np.ndarray]:
        """Generate high-contrast clinical retinal layer segmentation mask and overlay."""
        h, w = image.shape[:2]
        seg_mask = result.get("seg_mask")
        if seg_mask is None:
            seg_mask = np.zeros((h, w), dtype=np.float32)
        else:
            seg_mask = cv2.resize(seg_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        label = result.get("label", "NORMAL")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        tissue = (gray > 18).astype(np.uint8)

        # Morphological tissue contour
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_clean = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel)

        # Layer boundaries (upper ILM and lower RPE)
        edges = cv2.Canny(tissue_clean * 255, 50, 150)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

        # Semantic pathology masks
        lesion_m = (seg_mask > 0.5) & (tissue_clean > 0)
        fluid_m = (seg_mask > 0.3) & (seg_mask <= 0.5) & (tissue_clean > 0)
        edema_m = (seg_mask > 0.15) & (seg_mask <= 0.3) & (tissue_clean > 0)

        # Standalone clinical segmentation mask on black background (BGR)
        seg_render = np.zeros((h, w, 3), dtype=np.uint8)

        if label == "NORMAL":
            # Normal: highlight healthy neurosensory retina & sharp, intact ILM/RPE boundaries
            seg_render[tissue_clean > 0] = [60, 40, 20]      # Soft neurosensory layer
            seg_render[edges_dilated > 0] = [80, 230, 100]    # Vibrant intact boundary (Green)
        else:
            # Retinal tissue base
            seg_render[tissue_clean > 0] = [50, 30, 20]
            seg_render[edges_dilated > 0] = [80, 200, 100]    # Green boundary contours

            if label == "DME":
                # Cystoid intraretinal fluid & edema (Cyan / Electric Blue)
                seg_render[edema_m] = [200, 180, 30]
                seg_render[fluid_m] = [255, 230, 0]           # Vibrant Cyan fluid cysts
                seg_render[lesion_m] = [220, 255, 50]
            elif label == "CNV":
                # Subretinal neovascular membrane & lesion (Amber / Coral Red)
                seg_render[edema_m] = [180, 150, 20]
                seg_render[fluid_m] = [0, 165, 255]           # Orange subretinal fluid
                seg_render[lesion_m] = [30, 70, 240]          # Vibrant Crimson lesion membrane
            elif label == "DRUSEN" or label == "AMD":
                # Nodular sub-RPE drusen deposits (Fuchsia / Violet)
                seg_render[edema_m] = [150, 70, 180]
                seg_render[fluid_m] = [200, 50, 220]
                seg_render[lesion_m] = [240, 60, 210]         # Vivid magenta drusen mounds
            else:
                seg_render[fluid_m] = [240, 200, 0]
                seg_render[lesion_m] = [0, 140, 255]

        # Clinical overlay onto base OCT
        base_bgr = image.copy()
        overlay = base_bgr.copy()
        mask_active = (seg_render.sum(axis=2) > 0)
        alpha = 0.48
        overlay[mask_active] = cv2.addWeighted(base_bgr[mask_active], 1.0 - alpha, seg_render[mask_active], alpha, 0)
        overlay[edges_dilated > 0] = cv2.addWeighted(base_bgr[edges_dilated > 0], 0.2, seg_render[edges_dilated > 0], 0.8, 0)

        return seg_render, overlay

    def generate_reports(self, result: dict, scan_id: int, patient_name: str = "Patient") -> dict:
        en = generate_english_pdf(result, scan_id, patient_name)
        ta = generate_tamil_pdf(result, scan_id, patient_name)
        self.db.add_report(scan_id, "english", str(en))
        self.db.add_report(scan_id, "tamil", str(ta))
        return {"english": en, "tamil": ta}


_service: Optional[VisionCareService] = None


def get_service() -> VisionCareService:
    global _service
    if _service is None:
        _service = VisionCareService()
    return _service
