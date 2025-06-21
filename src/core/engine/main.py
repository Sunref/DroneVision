import cv2
import numpy as np
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import open3d as o3d

class TopoDepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Profundidade com MiDaS e Visualização 3D com Open3D")
        self.root.geometry("1720x775")

        self.image = None
        self.image_for_3d = None # Armazena a imagem original para a nuvem de pontos
        self.processed_image = None
        self.depth_map_raw = None
        self.cap = None
        self.running = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # Modelos separados para imagem (mais forte) e vídeo (mais leve)
        print("Carregando modelos MiDaS... Isso pode levar um momento.")
        self.model_strong = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.model_strong.to(self.device).eval()
        self.transform_strong = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

        self.model_fast = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model_fast.to(self.device).eval()
        self.transform_fast = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        print("Modelos carregados.")

        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        btn_row = ttk.Frame(button_frame)
        btn_row.pack(fill='x', pady=5)

        ttk.Button(btn_row, text="Carregar Imagem", command=self.load_image, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Processar", command=self.process_image, bootstyle=SUCCESS).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Salvar Resultado", command=self.save_result, bootstyle=INFO).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Iniciar Câmera", command=self.start_camera, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Parar Câmera", command=self.stop_camera, bootstyle=DANGER).pack(side='left', padx=5)
        ttk.Button(btn_row, text="Visualizar em 3D (Open3D)", command=self.visualize_3d_open3d, bootstyle=INFO).pack(side='left', padx=5)

        ttk.Label(btn_row, text="Colormap:").pack(side='left', padx=(20, 5))
        self.colormap_var = ttk.Combobox(btn_row, values=[
            "Magma", "Inferno", "Plasma", "Viridis", "Cividis", "Jet", "Turbo", "Hot"
        ], state="readonly", width=12)
        self.colormap_var.set("Plasma")
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

        frame_pre = ttk.LabelFrame(image_frame, text="Imagem Original", bootstyle=INFO)
        frame_pre.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.label_orig = ttk.Label(frame_pre)
        self.label_orig.pack(fill='both', expand=True)

        frame_post = ttk.LabelFrame(image_frame, text="Mapa de Profundidade", bootstyle=SUCCESS)
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
            self.image_for_3d = self.image.copy()
            self.processed_image = None
            self.depth_map_raw = None
            self.show_image(self.image, processed=False)
            # Limpa a imagem processada anterior
            self.label_proc.config(image='')

    def process_image(self):
        if self.image is None:
            return
        # Usa um thread para não congelar a GUI durante o processamento
        threading.Thread(target=self._process_image_thread, daemon=True).start()

    def _process_image_thread(self):
        print("Processando imagem com modelo forte...")
        depth_map, depth_raw = self.run_midas(self.image, use_strong=True)
        self.depth_map_raw = depth_raw
        self.processed_image = self.apply_adjustments(depth_map)
        self.root.after(0, self.update_metrics)
        self.root.after(0, lambda: self.show_image(self.image, processed=False))
        self.root.after(0, lambda: self.show_image(self.processed_image, processed=True))
        print("Processamento concluído.")

    def apply_adjustments(self, depth_map):
        brightness = self.brightness_scale.get()
        contrast = self.contrast_scale.get()
        adjusted = cv2.convertScaleAbs(depth_map, alpha=contrast, beta=(brightness - 1) * 127)
        colormap_name = self.colormap_var.get()
        cmap_dict = {
            "Magma": cv2.COLORMAP_MAGMA, "Inferno": cv2.COLORMAP_INFERNO,
            "Plasma": cv2.COLORMAP_PLASMA, "Viridis": cv2.COLORMAP_VIRIDIS,
            "Cividis": cv2.COLORMAP_CIVIDIS, "Jet": cv2.COLORMAP_JET,
            "Turbo": cv2.COLORMAP_TURBO, "Hot": cv2.COLORMAP_HOT,
        }
        colormap = cmap_dict.get(colormap_name, cv2.COLORMAP_PLASMA)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
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
        self.show_image(depth_colored, processed=True)

    def update_metrics(self):
        if self.depth_map_raw is None:
            self.metrics_label.config(text="Mín: -  Máx: -  Média: -")
            return
        mini, maxi, mean = np.min(self.depth_map_raw), np.max(self.depth_map_raw), np.mean(self.depth_map_raw)
        self.metrics_label.config(text=f"Mín: {mini:.3f}  Máx: {maxi:.3f}  Média: {mean:.3f}")

    def run_midas(self, img_bgr, use_strong=False):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        model = self.model_strong if use_strong else self.model_fast
        transform = self.transform_strong if use_strong else self.transform_fast
        
        with torch.no_grad():
            img_input = transform(img_rgb).to(self.device)
            prediction = model(img_input)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img_rgb.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 0:
            depth_norm = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        else:
            depth_norm = np.zeros(depth.shape, dtype=np.uint8)
            
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
        return depth_colored, depth

    def show_image(self, img, processed=False):
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img_rgb.shape[:2]
        max_width, max_height = 800, 600
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(img_resized)
        tk_image = ImageTk.PhotoImage(pil_image)

        if processed:
            self.label_proc.config(image=tk_image); self.tk_image_proc = tk_image
        else:
            self.label_orig.config(image=tk_image); self.tk_image_orig = tk_image

    def visualize_3d_open3d(self):
        if self.image_for_3d is None or self.depth_map_raw is None:
            print("ERRO: Carregue e processe uma imagem antes de gerar a visualização 3D.")
            return
        threading.Thread(target=self._create_and_show_pcd, daemon=True).start()

    def _create_and_show_pcd(self):
        print("Iniciando criação da nuvem de pontos 3D com Open3D...")
        color_img_rgb = cv2.cvtColor(self.image_for_3d, cv2.COLOR_BGR2RGB)
        
        # O MiDaS fornece profundidade inversa relativa. Valores maiores = mais perto.
        # Precisamos converter para um mapa de profundidade onde valores maiores = mais longe.
        depth_map_raw = self.depth_map_raw

        # Normaliza o mapa de profundidade inversa para o intervalo [0, 255]
        depth_min = depth_map_raw.min()
        depth_max = depth_map_raw.max()
        if depth_max - depth_min > 0:
            inverse_depth_normalized = (255 * (depth_map_raw - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        else:
            inverse_depth_normalized = np.zeros(depth_map_raw.shape, dtype=np.uint8)

        # Inverte o mapa para obter profundidade real (valores maiores = mais longe)
        # Agora, 0 é o mais próximo e 255 é o mais distante.
        depth_map_normalized = 255 - inverse_depth_normalized

        # Converte para o formato de imagem do Open3D
        color_o3d = o3d.geometry.Image(color_img_rgb)
        depth_o3d = o3d.geometry.Image(depth_map_normalized.astype(np.float32))

        h, w = color_img_rgb.shape[:2]
        # Estimativa dos parâmetros intrínsecos da câmera (heurística).
        # Um comprimento focal de ~1.2*largura é uma heurística comum para câmaras de telemóvel.
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx=w * 1.2, fy=w * 1.2, cx=w / 2, cy=h / 2)

        # Mapeia o intervalo de profundidade [0, 255] para uma profundidade métrica.
        # Por exemplo, para mapear para uma cena de 5 metros de profundidade:
        # profundidade_real = valor_pixel / depth_scale => 5 = 255 / depth_scale => depth_scale = 51
        depth_scale = 50.0  # Ajuste este valor para aumentar/diminuir a profundidade da cena
        depth_trunc = 5.0   # Trunca a profundidade em metros (deve ser 255 / depth_scale)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        
        # Vira de cabeça para baixo e corrige o espelhamento
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        print("Limpando a nuvem de pontos (removendo outliers)...")
        # Usa uma subamostragem de voxel para tornar a remoção mais rápida e robusta
        # O tamanho do voxel depende da escala da cena; 0.02 = 2cm
        voxel_size = max(pcd.get_max_bound() - pcd.get_min_bound()) / 100
        if voxel_size <= 0:
             print("ERRO: Tamanho do voxel inválido. A nuvem de pontos pode não ter profundidade.")
             voxel_size = 0.01
        
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        cleaned_pcd = downsampled_pcd.select_by_index(ind)

        if not cleaned_pcd.has_points():
            print("ERRO: A nuvem de pontos ficou vazia após a limpeza. A nuvem original será mostrada.")
            if not downsampled_pcd.has_points():
                 if not pcd.has_points():
                      print("ERRO: Não há pontos para visualizar.")
                      return
                 cleaned_pcd = pcd
            else:
                 cleaned_pcd = downsampled_pcd


        print(f"Visualizando a nuvem de pontos com {len(cleaned_pcd.points)} pontos.")
        # Centraliza o sistema de coordenadas no centro da nuvem de pontos
        axis_size = np.mean(cleaned_pcd.get_max_bound() - cleaned_pcd.get_min_bound()) * 0.2
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size if axis_size > 0 else 0.1, 
            origin=cleaned_pcd.get_center()
        )
        o3d.visualization.draw_geometries([cleaned_pcd, coord_frame], window_name="Nuvem de Pontos 3D (Open3D)")


    def save_result(self):
        if self.processed_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path: cv2.imwrite(path, self.processed_image)

    def start_camera(self):
        if self.running: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a câmera"); return
        self.running = True
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        self.running = False
        if self.cap: self.cap.release(); self.cap = None

    def camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: break
            self.image_for_3d = frame.copy()
            result, raw = self.run_midas(frame, use_strong=False)
            self.depth_map_raw = raw
            self.root.after(0, lambda f=frame, r=result: (self.show_image(f, False), self.show_image(r, True)))
        if self.cap: self.cap.release(); self.cap = None

def main():
    root = ttk.Window(themename="cyborg")
    app = TopoDepthApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()