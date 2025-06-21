# Funções utilitárias para TopoDepthApp
import numpy as np
import cv2

def update_metrics(depth_map_raw, metrics_label):
    if depth_map_raw is None:
        metrics_label.config(text="Mín: -  Máx: -  Média: -")
        return
    mini = np.min(depth_map_raw)
    maxi = np.max(depth_map_raw)
    mean = np.mean(depth_map_raw)
    metrics_label.config(text=f"Mín: {mini:.3f}  Máx: {maxi:.3f}  Média: {mean:.3f}")

def apply_adjustments(depth_map, brightness, contrast, colormap_name):
    adjusted = cv2.convertScaleAbs(depth_map, alpha=contrast, beta=(brightness - 1)*127)
    cmap_dict = {
        "Magma": cv2.COLORMAP_MAGMA,
        "Inferno": cv2.COLORMAP_INFERNO,
        "Plasma": cv2.COLORMAP_PLASMA,
        "Viridis": cv2.COLORMAP_VIRIDIS,
        "Cividis": cv2.COLORMAP_CIVIDIS,
        "Jet": cv2.COLORMAP_JET,
        "Turbo": cv2.COLORMAP_TURBO,
        "Hot": cv2.COLORMAP_HOT,
    }
    colormap = cmap_dict.get(colormap_name, cv2.COLORMAP_MAGMA)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(adjusted, -1, sharpen_kernel)
    adjusted_colored = cv2.applyColorMap(sharpened, colormap)
    return adjusted_colored
