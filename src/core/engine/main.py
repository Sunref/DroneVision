import cv2
import numpy as np
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class TopoDepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Profundidade com MiDaS")
        self.root.geometry("1720x775")

        self.image = None
        self.processed_image = None
        self.depth_map_raw = None
        self.cap = None
        self.running = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelos separados para imagem (mais forte) e vídeo (mais leve)
        self.model_strong = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.model_strong.to(self.device).eval()
        self.transform_strong = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

        self.model_fast = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model_fast.to(self.device).eval()
        self.transform_fast = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        btn_row = ttk.Frame(button_frame)
        btn_row.pack(fill='x', pady=5)

        ttk.Button(btn_row, text="Carregar Imagem", command=self.load_image, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Processar", command=self.process_image, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Salvar Resultado", command=self.save_result, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Iniciar Câmera", command=self.start_camera, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Parar Câmera", command=self.stop_camera, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Visualização 3D", command=self.visualizar_3d, bootstyle=INFO).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Curvas de Nível", command=self.visualizar_contornos, bootstyle=INFO).pack(side='left', padx=5)

        ttk.Label(btn_row, text="Colormap:").pack(side='left', padx=(20, 5))
        self.colormap_var = ttk.Combobox(btn_row, values=[
            "Magma", "Inferno", "Plasma", "Viridis", "Cividis", "Jet", "Turquoise", "Hot"
        ], state="readonly", width=12)
        self.colormap_var.current(0)
        self.colormap_var.pack(side='left', padx=5)

        ttk.Label(btn_row, text="Brilho:").pack(side='left', padx=(20, 5))
        self.brightness_scale = ttk.Scale(btn_row, from_=0.5, to=2.0, value=1.0, command=self.update_processed_image, length=100)
        self.brightness_scale.pack(side='left', padx=5)
        ttk.Label(btn_row, text="Contraste:").pack(side='left', padx=(10, 5))
        self.contrast_scale = ttk.Scale(btn_row, from_=0.5, to=3.0, value=1.0, command=self.update_processed_image, length=100)
        self.contrast_scale.pack(side='left', padx=5)

        self.metrics_label = ttk.Label(main_frame, text="Mín: -  Máx: -  Média: -", bootstyle=INFO)
        self.metrics_label.pack(fill='x', pady=(0, 10))

        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill='both', expand=True)

        frame_pre = ttk.LabelFrame(image_frame, text="Imagem Pré-processamento", bootstyle=INFO)
        frame_pre.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.label_orig = ttk.Label(frame_pre)
        self.label_orig.pack(fill='both', expand=True)

        frame_post = ttk.LabelFrame(image_frame, text="Imagem Pós-processamento", bootstyle=SUCCESS)
        frame_post.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.label_proc = ttk.Label(frame_post)
        self.label_proc.pack(fill='both', expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.jpg *.png *.jpeg *.bmp *.tiff")]
        )
        if path:
            self.image = cv2.imread(path)
            self.processed_image = None
            self.depth_map_raw = None
            self.show_image(self.image, processed=False)

    def process_image(self):
        if self.image is None:
            return
        depth_map, depth_raw = self.run_midas(self.image, use_strong=True)
        self.depth_map_raw = depth_raw
        self.processed_image = self.apply_adjustments(depth_map)
        self.update_metrics()
        self.show_image(self.image, processed=False)
        self.show_image(self.processed_image, processed=True)

    def apply_adjustments(self, depth_map):
        brightness = self.brightness_scale.get()
        contrast = self.contrast_scale.get()
        adjusted = cv2.convertScaleAbs(depth_map, alpha=contrast, beta=(brightness - 1)*127)
        colormap_name = self.colormap_var.get()
        cmap_dict = {
            "Magma": cv2.COLORMAP_MAGMA,
            "Inferno": cv2.COLORMAP_INFERNO,
            "Plasma": cv2.COLORMAP_PLASMA,
            "Viridis": cv2.COLORMAP_VIRIDIS,
            "Cividis": cv2.COLORMAP_CIVIDIS,
            "Jet": cv2.COLORMAP_JET,
            "Turquoise": cv2.COLORMAP_TURBO,
            "Hot": cv2.COLORMAP_HOT,
        }
        colormap = cmap_dict.get(colormap_name, cv2.COLORMAP_MAGMA)
        # Aplica nitidez antes do colormap
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(adjusted, -1, sharpen_kernel)

        adjusted_colored = cv2.applyColorMap(sharpened, colormap)
        return adjusted_colored

    def update_processed_image(self, _=None):
        if self.depth_map_raw is None:
            return
        depth_min = self.depth_map_raw.min()
        depth_max = self.depth_map_raw.max()
        depth_norm = (255 * (self.depth_map_raw - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        depth_colored = self.apply_adjustments(depth_norm)
        self.processed_image = depth_colored
        self.update_metrics()
        self.show_image(depth_colored, processed=True)

    def update_metrics(self):
        if self.depth_map_raw is None:
            self.metrics_label.config(text="Mín: -  Máx: -  Média: -")
            return
        mini = np.min(self.depth_map_raw)
        maxi = np.max(self.depth_map_raw)
        mean = np.mean(self.depth_map_raw)
        self.metrics_label.config(text=f"Mín: {mini:.3f}  Máx: {maxi:.3f}  Média: {mean:.3f}")

    def run_midas(self, img_bgr, use_strong=False):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        model = self.model_strong if use_strong else self.model_fast
        transform = self.transform_strong if use_strong else self.transform_fast
        img_input = transform(img_rgb)
        if img_input.dim() == 3:
            img_input = img_input.unsqueeze(0)
        input_tensor = img_input.to(self.device)

        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        depth_norm = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        return depth_colored, depth

    def show_image(self, img, processed=False):
        if img is None:
            return

        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dimensões originais da imagem
        h, w = img_rgb.shape[:2]

        # Defina tamanho máximo para exibição (em pixels)
        max_width, max_height = 900, 600

        # Calcula escala proporcional para caber no espaço sem distorcer
        scale = min(max_width / w, max_height / h)

        new_w, new_h = int(w * scale), int(h * scale)

        # Redimensiona com essa escala proporcional
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(img_resized)
        tk_image = ImageTk.PhotoImage(pil_image)

        # Atualiza a label correta no Tkinter com a imagem redimensionada
        if processed:
            self.label_proc.config(image=tk_image)
            self.tk_image_proc = tk_image  # Guarda referência para não ser coletada
        else:
            self.label_orig.config(image=tk_image)
            self.tk_image_orig = tk_image

    def visualizar_3d(self):
        if self.depth_map_raw is None:
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(self.depth_map_raw.shape[1]), np.arange(self.depth_map_raw.shape[0]))
        ax.plot_surface(X, Y, self.depth_map_raw, cmap='inferno')
        ax.set_title("Mapa 3D de Profundidade")
        plt.tight_layout()
        plt.show()

    def visualizar_contornos(self):
        if self.depth_map_raw is None:
            return
        plt.figure()
        plt.contourf(self.depth_map_raw, levels=30, cmap='viridis')
        plt.colorbar(label='Profundidade')
        plt.title("Curvas de Nível da Profundidade")
        plt.tight_layout()
        plt.show()

    def save_result(self):
        if self.processed_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        if path:
            cv2.imwrite(path, self.processed_image)

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return
        self.running = True
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (320, 240))
            result, raw = self.run_midas(frame, use_strong=False)
            self.depth_map_raw = raw

            self.root.after(0, lambda f=frame, r=result: (self.show_image(f, processed=False), self.show_image(r, processed=True)))

        if self.cap:
            self.cap.release()
            self.cap = None

def main():
    root = ttk.Window(themename="journal")
    app = TopoDepthApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
