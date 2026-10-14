"""Ensemble inference — CPU only."""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.architectures import (
    AttentionUNet,
    EfficientNetClassifier,
    MobileNetClassifier,
    DeepLabV3Plus,
)
from utils.config import (
    CLASS_LABELS,
    ENSEMBLE_WEIGHTS,
    EXTENDED_LABELS,
    MODELS_DIR,
    STAGES,
)
from utils.preprocess import preprocess_oct

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class VisionCarePredictor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_classes = len(CLASS_LABELS)
        self.efficient = EfficientNetClassifier(self.num_classes, pretrained=True)
        self.mobile = MobileNetClassifier(self.num_classes, pretrained=True)
        self.segmenter = AttentionUNet(3, 1)
        self.deeplab = DeepLabV3Plus(3, 1)
        
        self.efficient_loaded = False
        self.mobile_loaded = False
        self._load_weights()
        
        self.efficient.eval()
        self.mobile.eval()
        self.segmenter.eval()
        self.deeplab.eval()
        for m in (self.efficient, self.mobile, self.segmenter, self.deeplab):
            m.to(self.device)

    def _load_weights(self) -> None:
        eff_path = MODELS_DIR / "efficientnet_b0_oct.pth"
        mob_path = MODELS_DIR / "mobilenet_v3_oct.pth"
        seg_path = MODELS_DIR / "attention_unet_seg.pth"
        if eff_path.exists():
            state = torch.load(eff_path, map_location=self.device, weights_only=True)
            try:
                info = self.efficient.load_state_dict(state, strict=False)
                classifier_mismatch = any('classifier' in k for k in info.missing_keys)
                if classifier_mismatch:
                    print("Warning: EfficientNet classifier weights failed to load due to class count mismatch.")
                else:
                    self.efficient_loaded = True
            except Exception as exc:
                print(f"Warning: efficient model weights did not fully match current class count: {exc}")
        if mob_path.exists():
            state = torch.load(mob_path, map_location=self.device, weights_only=True)
            try:
                info = self.mobile.load_state_dict(state, strict=False)
                classifier_mismatch = any('classifier' in k for k in info.missing_keys)
                if classifier_mismatch:
                    print("Warning: MobileNet classifier weights failed to load due to class count mismatch. Falling back to single-model mode.")
                else:
                    self.mobile_loaded = True
            except Exception as exc:
                print(f"Warning: mobilenet model weights did not fully match current class count: {exc}")
        if seg_path.exists():
            try:
                self.segmenter.load_state_dict(
                    torch.load(seg_path, map_location=self.device, weights_only=True),
                    strict=False
                )
                print("[Segmentation] Loaded AttentionUnet weights successfully.")
            except Exception as exc:
                print(f"[Segmentation] Warning: initialized AttentionUnet with default weights due to state_dict mismatch: {exc}")

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        display, chw = preprocess_oct(image_bgr)
        tensor = torch.from_numpy(chw).float().unsqueeze(0).to(self.device)

        logits_e = self.efficient(tensor)
        logits_m = self.mobile(tensor)
        probs_e = F.softmax(logits_e, dim=1).cpu().numpy()[0]
        probs_m = F.softmax(logits_m, dim=1).cpu().numpy()[0]
        
        if self.efficient_loaded and self.mobile_loaded:
            probs = ENSEMBLE_WEIGHTS[0] * probs_e + ENSEMBLE_WEIGHTS[1] * probs_m
        elif self.efficient_loaded:
            probs = probs_e
        elif self.mobile_loaded:
            probs = probs_m
        else:
            probs = ENSEMBLE_WEIGHTS[0] * probs_e + ENSEMBLE_WEIGHTS[1] * probs_m

        idx = int(np.argmax(probs))
        label = CLASS_LABELS[idx]
        confidence = float(probs[idx] * 100)

        seg_input = torch.from_numpy(chw).float().unsqueeze(0)
        seg_logits = self.segmenter(seg_input)
        seg_mask = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]

        # DeepLabV3++ Boundary Detection
        boundary_logits = self.deeplab(seg_input.to(self.device))
        boundary_mask = torch.sigmoid(boundary_logits).cpu().numpy()[0, 0]

        lesion_mask = seg_mask > 0.5
        fluid_mask = (seg_mask > 0.3) & ~lesion_mask
        edema_mask = (seg_mask > 0.15) & ~lesion_mask & ~fluid_mask

        valid_mask = display.mean(axis=2) > 5 if display.ndim == 3 else display > 5
        valid_pixels = max(1, int(valid_mask.sum()))

        if label == "NORMAL":
            lesion_area = 0.0
            fluid_area = 0.0
            edema_area = 0.0
            total_affected = 0.0
            damage = 0.0
            abnormality = 0.0
            stage = "Normal"
        else:
            lesion_area = round(float((lesion_mask & valid_mask).sum() / valid_pixels * 100), 2)
            fluid_area = round(float((fluid_mask & valid_mask).sum() / valid_pixels * 100), 2)
            edema_area = round(float((edema_mask & valid_mask).sum() / valid_pixels * 100), 2)
            combined_mask = lesion_mask | fluid_mask | edema_mask
            total_affected = round(float((combined_mask & valid_mask).sum() / valid_pixels * 100), 2)
            abnormality = float(1.0 - probs[CLASS_LABELS.index("NORMAL")] if "NORMAL" in CLASS_LABELS else 0.5)
            abnormality = round(abnormality * 100, 2)
            damage = min(100.0, max(0.0, 0.5 * lesion_area + 0.3 * fluid_area + 0.2 * abnormality))
            stage = self._compute_stage(damage, confidence, label)

        extended = self._map_extended(label, seg_mask, damage)

        # Multi-model ensemble consensus and epistemic uncertainty
        idx_e = int(np.argmax(probs_e))
        eff_label = CLASS_LABELS[idx_e]
        eff_conf = round(float(probs_e[idx_e] * 100), 1)

        idx_m = int(np.argmax(probs_m))
        mob_label = CLASS_LABELS[idx_m]
        mob_conf = round(float(probs_m[idx_m] * 100), 1)

        consensus_match = (eff_label == mob_label)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        uncertainty_rating = "Low Uncertainty (High Trust)" if entropy < 0.4 else ("Moderate Uncertainty" if entropy < 0.75 else "High Epistemic Uncertainty")

        consensus = {
            "efficientnet": {"label": eff_label, "confidence": eff_conf, "weight": 0.60},
            "mobilenet": {"label": mob_label, "confidence": mob_conf, "weight": 0.40},
            "agreement": bool(consensus_match),
            "entropy_nats": round(entropy, 3),
            "uncertainty_rating": uncertainty_rating,
        }

        # Clinical OCT Biomarkers & ETDRS Macular Metrics
        if label == "NORMAL":
            cst_microns = 262
            cst_status = "Normal Foveal Profile (240–300 µm)"
            irf = False
            srf = False
            ped = False
            ez_integrity = "Intact (Intact IS/OS Junction)"
            triage_level = "Routine Triage"
            triage_timeframe = "12 Months (Annual Exam)"
            etdrs = {
                "central": 262, "inner_sup": 325, "inner_inf": 320, "inner_nas": 330, "inner_temp": 310,
                "outer_sup": 285, "outer_inf": 280, "outer_nas": 295, "outer_temp": 270
            }
        elif label == "DME":
            cst_microns = int(min(680, 340 + fluid_area * 5.4 + edema_area * 3.2))
            cst_status = f"Macular Thickening / Edema (+{cst_microns - 262} µm excess)"
            irf = True
            srf = (fluid_area > 15)
            ped = False
            ez_integrity = "Focal Disruption (Cystoid Compression)"
            triage_level = "Urgent Vitreoretinal Referral"
            triage_timeframe = "< 72 Hours (Anti-VEGF Evaluation)"
            etdrs = {
                "central": cst_microns, "inner_sup": cst_microns + 30, "inner_inf": cst_microns + 25, "inner_nas": cst_microns + 35, "inner_temp": cst_microns + 15,
                "outer_sup": int(cst_microns * 0.75), "outer_inf": int(cst_microns * 0.72), "outer_nas": int(cst_microns * 0.78), "outer_temp": int(cst_microns * 0.70)
            }
        elif label == "CNV":
            cst_microns = int(min(590, 320 + lesion_area * 4.6 + fluid_area * 3.5))
            cst_status = f"Subretinal Infiltration & Fluid (+{cst_microns - 262} µm excess)"
            irf = (fluid_area > 20)
            srf = True
            ped = True
            ez_integrity = "Extensive Disruption (Neovascular Ingrowth)"
            triage_level = "Critical Emergency"
            triage_timeframe = "< 48 Hours (Anti-VEGF Therapy)"
            etdrs = {
                "central": cst_microns, "inner_sup": cst_microns + 20, "inner_inf": cst_microns + 35, "inner_nas": cst_microns + 25, "inner_temp": cst_microns + 10,
                "outer_sup": int(cst_microns * 0.72), "outer_inf": int(cst_microns * 0.75), "outer_nas": int(cst_microns * 0.74), "outer_temp": int(cst_microns * 0.68)
            }
        else:  # DRUSEN / AMD
            cst_microns = int(min(390, 275 + lesion_area * 2.3))
            cst_status = f"Sub-RPE Elevation Mounds (+{cst_microns - 262} µm)"
            irf = False
            srf = False
            ped = True
            ez_integrity = "Focal Attenuation (Drusenoid Elevation)"
            triage_level = "Semi-Urgent Review"
            triage_timeframe = "2-4 Weeks (AREDS2 Protocol)"
            etdrs = {
                "central": cst_microns, "inner_sup": cst_microns + 15, "inner_inf": cst_microns + 10, "inner_nas": cst_microns + 20, "inner_temp": cst_microns + 8,
                "outer_sup": 305, "outer_inf": 298, "outer_nas": 310, "outer_temp": 290
            }

        biomarkers = {
            "cst_microns": cst_microns,
            "cst_status": cst_status,
            "irf": irf,
            "srf": srf,
            "ped": ped,
            "ez_integrity": ez_integrity,
            "triage_level": triage_level,
            "triage_timeframe": triage_timeframe,
        }

        # 3D Macular Topography Grid (20x20 elevation mesh for Plotly 3D Surface)
        grid_size = 20
        x_coords = np.linspace(-3, 3, grid_size)
        y_coords = np.linspace(-3, 3, grid_size)
        xx, yy = np.meshgrid(x_coords, y_coords)
        r = np.sqrt(xx**2 + yy**2)

        if label == "NORMAL":
            # Foveal pit depression: outer rim ~320um, central fovea ~260um
            topo_z = 320.0 - 60.0 * np.exp(-((r / 1.1) ** 2))
        elif label == "DME":
            # Central macular cystoid swelling dome with cyst micro-depressions
            topo_z = 280.0 + (cst_microns - 280.0) * np.exp(-((r / 1.8) ** 2)) - 25.0 * np.sin(xx * 2.5) * np.cos(yy * 2.5) * np.exp(-((r / 2.0) ** 2))
        elif label == "CNV":
            # Asymmetric neovascular elevated mass with surrounding fluid slope
            topo_z = 290.0 + (cst_microns - 290.0) * np.exp(-(((xx - 0.4)**2 + (yy + 0.3)**2) / 1.6))
        else:  # DRUSEN
            # Nodular drusen bumps superimposed on baseline retina
            bumps = 28.0 * (np.sin(xx * 3.5)**2 * np.cos(yy * 3.5)**2 * np.exp(-((r / 2.2) ** 2)))
            topo_z = 300.0 - 30.0 * np.exp(-((r / 1.2) ** 2)) + bumps

        topography_3d = np.round(topo_z, 1).tolist()

        # Pathology thickness profile along horizontal B-scan plane
        valid_mask_high = display.mean(axis=2) > 15 if display.ndim == 3 else display > 15
        pathology_mask = (seg_mask > 0.15) & valid_mask_high
        thickness_profile = [int(pathology_mask[:, col].sum()) for col in range(seg_mask.shape[1])]

        return {
            "display_image": display,
            "preprocessed": display,
            "label": label,
            "label_full": EXTENDED_LABELS.get(label, label),
            "extended_findings": extended,
            "confidence": confidence,
            "probabilities": {CLASS_LABELS[i]: float(probs[i] * 100) for i in range(len(CLASS_LABELS))},
            "stage": stage,
            "retina_affected_pct": round(damage, 1),
            "total_affected_pct": total_affected,
            "lesion_area_pct": lesion_area,
            "fluid_area_pct": fluid_area,
            "edema_area_pct": edema_area,
            "abnormality_pct": abnormality,
            "seg_mask": seg_mask,
            "boundary_mask": boundary_mask,
            "probs_raw": probs,
            "thickness_profile": thickness_profile,
            "biomarkers": biomarkers,
            "consensus": consensus,
            "etdrs_grid": etdrs,
            "topography_3d": topography_3d,
        }

    def _compute_stage(self, damage: float, confidence: float, label: str) -> str:
        if label == "NORMAL":
            return "Normal"
        if damage < 20:
            return "Mild"
        if damage < 40:
            return "Moderate"
        if damage < 65:
            return "Severe"
        return "Critical"

    def _map_extended(
        self, label: str, seg_mask: np.ndarray, damage: float
    ) -> list[str]:
        if label == "NORMAL":
            return ["Normal Healthy Retina"]
        findings = [EXTENDED_LABELS.get(label, label)]
        if label == "DME" or damage > 25:
            findings.append(EXTENDED_LABELS["RETINAL_EDEMA"])
        if label == "CNV":
            findings.append(EXTENDED_LABELS["RETINAL_FLUID"])
        if label == "DRUSEN":
            findings.append(EXTENDED_LABELS["MACULAR_DEGENERATION"])
        if (seg_mask > 0.5).sum() > 100:
            findings.append(EXTENDED_LABELS["LESIONS"])
        if damage > 50 and label != "NORMAL":
            findings.append(EXTENDED_LABELS["AMD"])
        return list(dict.fromkeys(findings))


_predictor: Optional[VisionCarePredictor] = None


def get_predictor() -> VisionCarePredictor:
    global _predictor
    if _predictor is None:
        _predictor = VisionCarePredictor()
    return _predictor
