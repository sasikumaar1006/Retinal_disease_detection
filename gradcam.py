import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

        if class_idx is None:
            class_idx = int(torch.argmax(probs).item())

        confidence = float(probs[class_idx].item())
        target_score = logits[0, class_idx]
        target_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam, class_idx, confidence, probs.detach().cpu().numpy()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def make_overlay(rgb_image_01: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w, _ = rgb_image_01.shape
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip((1.0 - alpha) * rgb_image_01 + alpha * heatmap_rgb, 0.0, 1.0)
    return overlay