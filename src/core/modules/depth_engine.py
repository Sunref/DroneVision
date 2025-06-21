# Engine de processamento de profundidade
import torch
import cv2
import numpy as np
import logging

class DepthEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_strong = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.model_strong.to(self.device).eval()
        self.transform_strong = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        self.model_fast = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model_fast.to(self.device).eval()
        self.transform_fast = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def run_midas(self, img_bgr, use_strong=False):
        try:
            if img_bgr is None or img_bgr.size == 0:
                raise ValueError("Imagem inválida ou vazia")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            model = self.model_strong if use_strong else self.model_fast
            transform = self.transform_strong if use_strong else self.transform_fast

            h, w = img_rgb.shape[:2]
            if h * w > 1920 * 1080 and use_strong:
                logging.warning("Imagem grande detectada, pode consumir muita memória")

            img_input = transform(img_rgb)
            if img_input.dim() == 3:
                img_input = img_input.unsqueeze(0)
            input_tensor = img_input.to(self.device)

            with torch.no_grad():
                try:
                    prediction = model(input_tensor)
                    if torch.isnan(prediction).any():
                        raise RuntimeError("O modelo gerou valores inválidos (NaN)")

                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img_rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        raise torch.cuda.OutOfMemoryError("GPU sem memória suficiente")
                    raise

            depth = prediction.cpu().numpy()
            if np.isnan(depth).any():
                raise RuntimeError("Dados de profundidade inválidos")

            depth_min = depth.min()
            depth_max = depth.max()
            if depth_min == depth_max:
                raise ValueError("Mapa de profundidade sem variação")

            depth_norm = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

            return depth_colored, depth

        except cv2.error as e:
            logging.error(f"Erro OpenCV: {str(e)}")
            raise RuntimeError(f"Erro no processamento da imagem: {str(e)}")
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            logging.error(f"Erro no MiDaS: {str(e)}")
            raise RuntimeError(f"Erro no processamento de profundidade: {str(e)}")
