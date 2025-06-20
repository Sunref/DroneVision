import cv2
import numpy as np
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
import json
from scipy.ndimage import gaussian_filter

class TopoDepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Profundidade com MiDaS")
        self.root.geometry("1720x900")  # Aumentei a altura para acomodar o histograma

        self.image = None
        self.processed_image = None
        self.depth_map_raw = None
        self.cap = None
        self.running = False

        # Variáveis para análise
        self.measuring_points = []
        self.selection_start = None
        self.selection_end = None
        self.selected_area = None
        self.measuring = False
        self.selecting_area = False

        # Variáveis para histograma
        self.histogram_frame = None
        self.histogram_canvas = None

        # Barra de progresso
        self.progress_var = ttk.IntVar()

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

        # Frame superior para botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))

        # Primeira linha de botões
        btn_row1 = ttk.Frame(button_frame)
        btn_row1.pack(fill='x', pady=5)

        # Botões com tooltips
        load_btn = ttk.Button(btn_row1, text="Carregar Imagem", command=self.load_image, bootstyle=PRIMARY)
        load_btn.pack(side='left', padx=5)
        ToolTip(load_btn, "Carregar uma nova imagem para processamento")

        process_btn = ttk.Button(btn_row1, text="Processar", command=self.process_image, bootstyle=PRIMARY)
        process_btn.pack(side='left', padx=5)
        ToolTip(process_btn, "Processar a imagem usando MiDaS")

        save_btn = ttk.Button(btn_row1, text="Salvar Resultado", command=self.save_result, bootstyle=PRIMARY)
        save_btn.pack(side='left', padx=5)
        ToolTip(save_btn, "Salvar a imagem processada")



        # Botões originais
        ttk.Button(btn_row1, text="Iniciar Câmera", command=self.start_camera, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Parar Câmera", command=self.stop_camera, bootstyle=PRIMARY).pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Visualização 3D", command=self.visualizar_3d, bootstyle=INFO).pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Curvas de Nível", command=self.visualizar_contornos, bootstyle=INFO).pack(side='left', padx=5)

        # Segunda linha de botões
        btn_row2 = ttk.Frame(button_frame)
        btn_row2.pack(fill='x', pady=5)

        measure_btn = ttk.Button(btn_row2, text="Medir Distância", command=self.toggle_measurement, bootstyle=INFO)
        measure_btn.pack(side='left', padx=5)
        ToolTip(measure_btn, "Medir distância relativa entre pontos")

        select_area_btn = ttk.Button(btn_row2, text="Selecionar Área", command=self.toggle_area_selection, bootstyle=INFO)
        select_area_btn.pack(side='left', padx=5)
        ToolTip(select_area_btn, "Selecionar área para análise estatística")

        clear_btn = ttk.Button(btn_row2, text="Limpar Seleção", command=self.clear_selection, bootstyle=WARNING)
        clear_btn.pack(side='left', padx=5)
        ToolTip(clear_btn, "Limpar seleção atual")

        # Frame de filtros
        filter_frame = ttk.LabelFrame(btn_row2, text="Filtros", bootstyle=PRIMARY)
        filter_frame.pack(side='left', padx=20)

        ttk.Label(filter_frame, text="Suavização:").pack(side='left', padx=5)
        self.smooth_var = ttk.DoubleVar(value=0)
        smooth_scale = ttk.Scale(filter_frame, from_=0, to=5, variable=self.smooth_var,
                               command=self.apply_filters, length=100)
        smooth_scale.pack(side='left', padx=5)
        ToolTip(smooth_scale, "Ajustar nível de suavização")

        self.edge_var = ttk.BooleanVar(value=False)
        edge_check = ttk.Checkbutton(filter_frame, text="Realce de Bordas",
                                   variable=self.edge_var,
                                   command=self.apply_filters,
                                   bootstyle="round-toggle")
        edge_check.pack(side='left', padx=5)
        ToolTip(edge_check, "Ativar/Desativar realce de bordas")

        # Colormap e ajustes originais
        ttk.Label(btn_row2, text="Colormap:").pack(side='left', padx=(20, 5))
        self.colormap_var = ttk.Combobox(btn_row2, values=[
            "Magma", "Inferno", "Plasma", "Viridis", "Cividis", "Jet", "Turbo", "Hot"
        ], state="readonly", width=12)
        self.colormap_var.current(0)
        self.colormap_var.pack(side='left', padx=5)

        ttk.Label(btn_row2, text="Brilho:").pack(side='left', padx=(20, 5))
        self.brightness_scale = ttk.Scale(btn_row2, from_=0.5, to=2.0, value=1.0,
                                        command=self.update_processed_image, length=100)
        self.brightness_scale.pack(side='left', padx=5)

        ttk.Label(btn_row2, text="Contraste:").pack(side='left', padx=(10, 5))
        self.contrast_scale = ttk.Scale(btn_row2, from_=0.5, to=3.0, value=1.0,
                                      command=self.update_processed_image, length=100)
        self.contrast_scale.pack(side='left', padx=5)

        # Barra de progresso
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                          maximum=100, bootstyle="success-striped")
        self.progress_bar.pack(fill='x', pady=5)

        self.metrics_label = ttk.Label(main_frame, text="Mín: -  Máx: -  Média: -", bootstyle=INFO)
        self.metrics_label.pack(fill='x', pady=(0, 10))

        # Frame principal para imagens e histograma
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)

        # Frame para imagens (lado a lado)
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(fill='both', expand=True, side='left')

        frame_pre = ttk.LabelFrame(image_frame, text="Imagem Pré-processamento", bootstyle=INFO)
        frame_pre.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.label_orig = ttk.Label(frame_pre)
        self.label_orig.pack(fill='both', expand=True)

        frame_post = ttk.LabelFrame(image_frame, text="Imagem Pós-processamento", bootstyle=SUCCESS)
        frame_post.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.label_proc = ttk.Label(frame_post)
        self.label_proc.pack(fill='both', expand=True)

        # Frame para histograma e estatísticas
        self.analysis_frame = ttk.Frame(content_frame)
        self.analysis_frame.pack(fill='both', expand=True, side='right', padx=(10, 0))
        self.setup_histogram()

    def setup_histogram(self):
        """Configura o frame do histograma"""
        fig, self.ax_hist = plt.subplots(figsize=(6, 4))
        self.histogram_canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
        self.histogram_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Frame para estatísticas
        self.stats_frame = ttk.LabelFrame(self.analysis_frame, text="Estatísticas", bootstyle=INFO)
        self.stats_frame.pack(fill='x', pady=5)
        self.stats_label = ttk.Label(self.stats_frame, text="")
        self.stats_label.pack(pady=5)

    def update_histogram(self):
        """Atualiza o histograma com os dados atuais"""
        if self.depth_map_raw is None:
            return

        self.ax_hist.clear()
        self.ax_hist.hist(self.depth_map_raw.flatten(), bins=50, color='blue', alpha=0.7)
        self.ax_hist.set_title("Distribuição de Profundidade")
        self.ax_hist.set_xlabel("Profundidade")
        self.ax_hist.set_ylabel("Frequência")
        self.histogram_canvas.draw()

        # Atualiza estatísticas
        stats = {
            "Média": np.mean(self.depth_map_raw),
            "Mediana": np.median(self.depth_map_raw),
            "Desvio Padrão": np.std(self.depth_map_raw),
            "Mínimo": np.min(self.depth_map_raw),
            "Máximo": np.max(self.depth_map_raw)
        }
        stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
        self.stats_label.config(text=stats_text)



    def apply_filters(self, _=None):
        """Aplica filtros ao mapa de profundidade"""
        if self.depth_map_raw is None:
            return

        # Aplica suavização gaussiana
        smoothed = gaussian_filter(self.depth_map_raw, sigma=self.smooth_var.get())

        # Aplica realce de bordas
        if self.edge_var.get():
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            smoothed = cv2.filter2D(smoothed, -1, kernel)

        # Normaliza e converte para visualização
        depth_norm = ((smoothed - smoothed.min()) / (smoothed.max() - smoothed.min()) * 255).astype(np.uint8)
        self.processed_image = cv2.applyColorMap(depth_norm,
            getattr(cv2, f'COLORMAP_{self.colormap_var.get().upper()}'))

        self.show_image(self.processed_image, processed=True)
        self.update_histogram()

    def toggle_measurement(self):
        """Ativa/desativa modo de medição de distância"""
        self.measuring = not getattr(self, 'measuring', False)
        self.selecting_area = False
        if self.measuring:
            self.measuring_points = []
            messagebox.showinfo("Medição",
                "Clique em dois pontos na imagem processada para medir a distância relativa")

    def toggle_area_selection(self):
        """Ativa/desativa modo de seleção de área"""
        self.selecting_area = not getattr(self, 'selecting_area', False)
        self.measuring = False
        if self.selecting_area:
            self.selection_start = None
            self.selection_end = None
            messagebox.showinfo("Seleção",
                "Clique e arraste para selecionar uma área para análise")

    def clear_selection(self):
        """Limpa a seleção atual"""
        self.selection_start = None
        self.selection_end = None
        self.measuring_points = []
        self.measuring = False
        self.selecting_area = False
        if self.processed_image is not None:
            self.show_image(self.processed_image, processed=True)

    def on_click(self, event):
        """Manipula eventos de clique do mouse"""
        if self.depth_map_raw is None:
            return

        x, y = event.x, event.y

        if self.measuring:
            self.measuring_points.append((x, y))
            if len(self.measuring_points) == 2:
                self.calculate_distance()
                self.measuring_points = []
        elif self.selecting_area:
            self.selection_start = (x, y)
            self.selection_end = None

    def on_drag(self, event):
        """Manipula eventos de arrasto do mouse"""
        if self.selecting_area and self.selection_start:
            self.selection_end = (event.x, event.y)
            self.update_selection()

    def on_release(self, event):
        """Manipula eventos de liberação do mouse"""
        if self.selecting_area and self.selection_start and self.selection_end:
            self.analyze_selected_area()

    def calculate_distance(self):
        """Calcula a distância relativa entre dois pontos"""
        if len(self.measuring_points) != 2:
            return

        # Obtém as coordenadas relativas ao widget
        x1, y1 = self.measuring_points[0]
        x2, y2 = self.measuring_points[1]

        # Obtém o tamanho atual do widget
        widget_width = self.label_proc.winfo_width()
        widget_height = self.label_proc.winfo_height()

        # Converte para coordenadas proporcionais
        x1_prop = x1 / widget_width
        y1_prop = y1 / widget_height
        x2_prop = x2 / widget_width
        y2_prop = y2 / widget_height

        # Converte para coordenadas da imagem original
        x1 = int(x1_prop * self.depth_map_raw.shape[1])
        y1 = int(y1_prop * self.depth_map_raw.shape[0])
        x2 = int(x2_prop * self.depth_map_raw.shape[1])
        y2 = int(y2_prop * self.depth_map_raw.shape[0])

        depth1 = self.depth_map_raw[y1, x1]
        depth2 = self.depth_map_raw[y2, x2]

        # Calcula distância euclidiana considerando profundidade
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (depth2-depth1)**2)

        messagebox.showinfo("Medição",
            f"Distância relativa: {distance:.2f}\n"
            f"Diferença de profundidade: {abs(depth2-depth1):.2f}")

    def update_selection(self):
        """Atualiza a visualização da área selecionada"""
        if self.processed_image is None or not self.selection_start or not self.selection_end:
            return

        # Obtém o tamanho atual do widget
        widget_width = self.label_proc.winfo_width()
        widget_height = self.label_proc.winfo_height()

        # Calcula a escala entre o widget e a imagem processada
        h, w = self.processed_image.shape[:2]
        scale_x = w / widget_width
        scale_y = h / widget_height

        # Converte coordenadas do widget para coordenadas da imagem
        x1 = int(min(self.selection_start[0], self.selection_end[0]) * scale_x)
        y1 = int(min(self.selection_start[1], self.selection_end[1]) * scale_y)
        x2 = int(max(self.selection_start[0], self.selection_end[0]) * scale_x)
        y2 = int(max(self.selection_start[1], self.selection_end[1]) * scale_y)

        # Garante que as coordenadas estão dentro dos limites da imagem
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))

        img_copy = self.processed_image.copy()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.show_image(img_copy, processed=True)

    def analyze_selected_area(self):
        """Analisa a área selecionada"""
        if not self.selection_start or not self.selection_end:
            return

        # Obtém o tamanho atual do widget
        widget_width = self.label_proc.winfo_width()
        widget_height = self.label_proc.winfo_height()

        # Converte para coordenadas proporcionais
        x1_prop = min(self.selection_start[0], self.selection_end[0]) / widget_width
        y1_prop = min(self.selection_start[1], self.selection_end[1]) / widget_height
        x2_prop = max(self.selection_start[0], self.selection_end[0]) / widget_width
        y2_prop = max(self.selection_start[1], self.selection_end[1]) / widget_height

        # Converte para coordenadas da imagem original
        x1 = int(x1_prop * self.depth_map_raw.shape[1])
        y1 = int(y1_prop * self.depth_map_raw.shape[0])
        x2 = int(x2_prop * self.depth_map_raw.shape[1])
        y2 = int(y2_prop * self.depth_map_raw.shape[0])

        # Limita as coordenadas aos limites da imagem
        x1 = max(0, min(x1, self.depth_map_raw.shape[1]-1))
        y1 = max(0, min(y1, self.depth_map_raw.shape[0]-1))
        x2 = max(0, min(x2, self.depth_map_raw.shape[1]-1))
        y2 = max(0, min(y2, self.depth_map_raw.shape[0]-1))

        selected_region = self.depth_map_raw[y1:y2+1, x1:x2+1]

        stats = {
            "Média": np.mean(selected_region),
            "Mediana": np.median(selected_region),
            "Desvio Padrão": np.std(selected_region),
            "Mínimo": np.min(selected_region),
            "Máximo": np.max(selected_region)
        }

        stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
        messagebox.showinfo("Análise da Área Selecionada", stats_text)

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

        def process():
            self.progress_var.set(0)
            self.root.update()

            # Processamento principal
            depth_map, depth_raw = self.run_midas(self.image, use_strong=True)
            self.progress_var.set(50)
            self.root.update()

            self.depth_map_raw = depth_raw
            self.processed_image = self.apply_adjustments(depth_map)
            self.progress_var.set(100)

            # Atualiza visualizações
            self.show_image(self.image, processed=False)
            self.show_image(self.processed_image, processed=True)
            self.update_histogram()

        threading.Thread(target=process, daemon=True).start()

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
            "Turbo": cv2.COLORMAP_TURBO,
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

        h, w = img_rgb.shape[:2]
        max_width, max_height = 900, 600
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_image = Image.fromarray(img_resized)
        tk_image = ImageTk.PhotoImage(pil_image)

        if processed:
            self.label_proc.config(image=tk_image)
            self.tk_image_proc = tk_image
            self.processed_size = (new_w, new_h)

            # Bind mouse events for processed image
            self.label_proc.bind('<Button-1>', self.on_click)
            self.label_proc.bind('<B1-Motion>', self.on_drag)
            self.label_proc.bind('<ButtonRelease-1>', self.on_release)
        else:
            self.label_orig.config(image=tk_image)
            self.tk_image_orig = tk_image

    def visualizar_3d(self):
        if self.depth_map_raw is None:
            return
        # Reduz a resolução para melhor performance
        scale = 8  # Aumentado para melhor performance
        h, w = self.depth_map_raw.shape
        reduced_depth = self.depth_map_raw[::scale, ::scale]
        y, x = np.mgrid[0:h:scale, 0:w:scale]

        plt.close('all')  # Fecha plots anteriores
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Configurações otimizadas para rotação suave
        surf = ax.plot_surface(x, y, reduced_depth, cmap='inferno',
                              rstride=3, cstride=3,  # Aumentado para mais performance
                              linewidth=0,
                              antialiased=False,  # Desativado para performance
                              shade=True)

        # Ajustes de visualização
        ax.view_init(elev=30, azim=45)  # Ângulo inicial melhor
        ax.set_title("Mapa 3D de Profundidade")

        plt.tight_layout()
        fig.canvas.draw_idle()  # Força uma renderização inicial
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
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (320, 240))
                result, raw = self.run_midas(frame, use_strong=False)
                self.depth_map_raw = raw

                self.root.after(0, lambda f=frame, r=result: (
                    self.show_image(f, processed=False),
                    self.show_image(r, processed=True)
                ))

            if self.cap:
                self.cap.release()
                self.cap = None
        except:
            pass

def main():
    root = ttk.Window(themename="journal")
    app = TopoDepthApp(root)
    try:
        root.mainloop()
    finally:
        # Limpa recursos ao fechar
        if hasattr(app, 'cap') and app.cap:
            app.cap.release()
        plt.close('all')

if __name__ == "__main__":
    main()
