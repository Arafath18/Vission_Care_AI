"""Grad-CAM explainability for OCT scans."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inp, out):
            self.activations = out.detach()

        def backward_hook(_module, _gin, gout):
            self.gradients = gout[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        self.model.zero_grad()
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        score = output[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay(
        self, image_rgb: np.ndarray, cam: np.ndarray
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Smooth activation map for continuous clinical gradient contours
        cam_smooth = cv2.GaussianBlur(cam_resized, (9, 9), 0)
        c_min, c_max = cam_smooth.min(), cam_smooth.max()
        if c_max > c_min:
            cam_norm = (cam_smooth - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_smooth)

        # Detect active retinal tissue to prevent background artifacts
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
        tissue_mask = (gray > 12).astype(np.float32)
        tissue_mask = cv2.GaussianBlur(tissue_mask, (5, 5), 0)

        # Suppress low-activation background noise and highlight focal pathological lesions
        cam_active = np.maximum(0.0, (cam_norm - 0.12) / 0.88)
        cam_boosted = np.power(cam_active, 1.15) * tissue_mask
        cam_uint8 = np.uint8(255 * np.clip(cam_boosted, 0.0, 1.0))

        # JET colormap: Cold blue/cyan -> Green -> Warm Amber/Yellow -> Fiery Red
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)

        # Dynamic opacity: hot spots get strong 0.60 opacity, low background drops to 0
        alpha = np.clip(cam_boosted * 0.60, 0.0, 0.60)[:, :, np.newaxis]
        overlay = (alpha * heatmap_rgb + (1.0 - alpha) * image_rgb.astype(np.float32)).astype(np.uint8)
        return overlay


def generate_gradcam(
    efficient_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    display_bgr: np.ndarray,
    class_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM heatmap overlay."""
    target = efficient_model.backbone.features[-1]
    gradcam = GradCAM(efficient_model, target)
    efficient_model.train()
    for p in efficient_model.parameters():
        p.requires_grad = True
    cam = gradcam.generate(input_tensor, class_idx)
    efficient_model.eval()
    for p in efficient_model.parameters():
        p.requires_grad = False

    # display_bgr is actually RGB (output of preprocess_oct which converts BGR→RGB).
    # Pass it directly to overlay() which expects an RGB image.
    overlay = gradcam.overlay(display_bgr, cam)
    cam_uint8 = np.uint8(255 * cv2.resize(cam, (display_bgr.shape[1], display_bgr.shape[0])))
    return overlay, cam_uint8
