import cv2
import numpy as np
import torch
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
import json
from scipy.ndimage import gaussian_filter
import logging
import sys
import traceback
import os
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('depth_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class TopoDepthApp:
    def __init__(self, root):
        self.root = root
        self.auto_detect = False
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
        self.measuring = False

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
        load_btn = ttk.Button(btn_row1, text="Carregar Imagem", command=self.load_image, bootstyle="success-outline")
        load_btn.pack(side='left', padx=5)
        ToolTip(load_btn, "Carregar uma nova imagem para processamento")

        process_btn = ttk.Button(btn_row1, text="Processar", command=self.process_image, bootstyle="success-outline")
        process_btn.pack(side='left', padx=5)
        ToolTip(process_btn, "Processar a imagem usando MiDaS")

        save_btn = ttk.Button(btn_row1, text="Salvar Resultado", command=self.save_result, bootstyle="success-outline")
        save_btn.pack(side='left', padx=5)
        ToolTip(save_btn, "Salvar a imagem processada")

        # Botões originais
        ttk.Button(btn_row1, text="Iniciar Câmera", command=self.start_camera, bootstyle="success-outline").pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Parar Câmera", command=self.stop_camera, bootstyle="danger-outline").pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Visualização 3D", command=self.visualizar_3d, bootstyle="info-outline").pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Curvas de Nível", command=self.visualizar_contornos, bootstyle="info-outline").pack(side='left', padx=5)

        # Segunda linha de botões
        btn_row2 = ttk.Frame(button_frame)
        btn_row2.pack(fill='x', pady=5)

        measure_btn = ttk.Button(btn_row2, text="Medir Distância", command=self.toggle_measurement, bootstyle="info-outline")
        measure_btn.pack(side='left', padx=5)
        ToolTip(measure_btn, "Medir distância relativa entre pontos")

        capture_btn = ttk.Button(btn_row2, text="Capturar Câmera", command=self.capture_camera, bootstyle="warning-outline")
        capture_btn.pack(side='left', padx=5)
        ToolTip(capture_btn, "Capturar e salvar imagem atual da câmera")

        clear_line_btn = ttk.Button(btn_row2, text="Limpar Linha", command=self.clear_measurement_line, bootstyle="warning-outline")
        clear_line_btn.pack(side='left', padx=5)
        ToolTip(clear_line_btn, "Limpar linha de medição")

        # Frame de filtros
        filter_frame = ttk.LabelFrame(btn_row2, text="Filtros", bootstyle="success")
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
                                          maximum=100, bootstyle="success")
        self.progress_bar.pack(fill='x', pady=5)

        self.metrics_label = ttk.Label(main_frame, text="Mín: -  Máx: -  Média: -", bootstyle="info")
        self.metrics_label.pack(fill='x', pady=(0, 10))

        # Frame principal para imagens e histograma
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)

        # Frame para imagens (lado a lado)
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(fill='both', expand=True, side='left')

        frame_pre = ttk.LabelFrame(image_frame, text="Imagem Pré-processamento", bootstyle="info")
        frame_pre.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.label_orig = ttk.Label(frame_pre)
        self.label_orig.pack(fill='both', expand=True)

        frame_post = ttk.LabelFrame(image_frame, text="Imagem Pós-processamento", bootstyle="success")
        frame_post.pack(side='left', fill='both', expand=True, padx=5, pady=5)

        # Usar Canvas para captura precisa de coordenadas
        self.canvas_proc = Canvas(frame_post, bg='black')
        self.canvas_proc.pack(fill='both', expand=True)

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

        if self.measuring:
            self.measuring_points = []
            messagebox.showinfo("Medição",
                "Clique em dois pontos na imagem processada para medir a distância relativa")

    def on_click(self, event):
        """Manipula eventos de clique do mouse"""
        if self.depth_map_raw is None:
            return

        # Ajusta coordenadas considerando o offset da imagem centralizada
        if hasattr(self, 'image_offset') and hasattr(self, 'image_size'):
            offset_x, offset_y = self.image_offset
            img_w, img_h = self.image_size

            # Coordenadas relativas à imagem
            x = event.x - offset_x
            y = event.y - offset_y

            # Verifica se clicou dentro da imagem
            if x < 0 or x >= img_w or y < 0 or y >= img_h:
                return  # Clique fora da imagem, ignora
        else:
            x, y = event.x, event.y

        if self.measuring:
            self.measuring_points.append((x, y))
            if len(self.measuring_points) == 2:
                self.calculate_distance()
                self.measuring_points = []

    def calculate_distance(self):
        """Calcula e visualiza a distância relativa entre dois pontos"""
        if len(self.measuring_points) != 2:
            return

        # Coordenadas do clique (já são as coordenadas exatas da imagem no Canvas)
        x1_img, y1_img = self.measuring_points[0]
        x2_img, y2_img = self.measuring_points[1]

        print(f"Coordenadas diretas do Canvas: ({x1_img}, {y1_img}), ({x2_img}, {y2_img})")

        # Obtém dimensões da imagem processada
        img_h, img_w = self.processed_image.shape[:2]

        # Garante que as coordenadas estão dentro dos limites
        x1_img = max(0, min(x1_img, img_w - 1))
        y1_img = max(0, min(y1_img, img_h - 1))
        x2_img = max(0, min(x2_img, img_w - 1))
        y2_img = max(0, min(y2_img, img_h - 1))

        # Converte para coordenadas do depth map
        depth_h, depth_w = self.depth_map_raw.shape
        x1_depth = int((x1_img / img_w) * depth_w)
        y1_depth = int((y1_img / img_h) * depth_h)
        x2_depth = int((x2_img / img_w) * depth_w)
        y2_depth = int((y2_img / img_h) * depth_h)

        # Garante limites do depth map
        x1_depth = max(0, min(x1_depth, depth_w - 1))
        y1_depth = max(0, min(y1_depth, depth_h - 1))
        x2_depth = max(0, min(x2_depth, depth_w - 1))
        y2_depth = max(0, min(y2_depth, depth_h - 1))

        depth1 = self.depth_map_raw[y1_depth, x1_depth]
        depth2 = self.depth_map_raw[y2_depth, x2_depth]

        # Calcula distância
        distance = np.sqrt((x2_depth-x1_depth)**2 + (y2_depth-y1_depth)**2 + (depth2-depth1)**2)

        # Limpa Canvas e redesenha imagem com pontos
        self.canvas_proc.delete("measurement")

        # Ajusta coordenadas para desenho considerando offset
        if hasattr(self, 'image_offset'):
            offset_x, offset_y = self.image_offset
            draw_x1 = x1_img + offset_x
            draw_y1 = y1_img + offset_y
            draw_x2 = x2_img + offset_x
            draw_y2 = y2_img + offset_y
        else:
            draw_x1, draw_y1 = x1_img, y1_img
            draw_x2, draw_y2 = x2_img, y2_img

        # Desenha pontos e linha no Canvas
        self.canvas_proc.create_oval(draw_x1-5, draw_y1-5, draw_x1+5, draw_y1+5,
                                   fill='red', outline='red', tags="measurement")
        self.canvas_proc.create_oval(draw_x2-5, draw_y2-5, draw_x2+5, draw_y2+5,
                                   fill='red', outline='red', tags="measurement")
        self.canvas_proc.create_line(draw_x1, draw_y1, draw_x2, draw_y2,
                                   fill='red', width=3, tags="measurement")

        # Texto no meio da linha
        text_x = (draw_x1 + draw_x2) // 2
        text_y = (draw_y1 + draw_y2) // 2
        self.canvas_proc.create_text(text_x, text_y, text=f"D: {distance:.1f}",
                                   fill='red', font=("Arial", 12, "bold"), tags="measurement")

        # Mensagem
        msg = f"Distância relativa: {distance:.2f}\nDiferença de profundidade: {abs(depth2-depth1):.2f}"
        messagebox.showinfo("Medição", msg)

    def clear_measurement_line(self):
        """Limpa a linha de medição do Canvas"""
        if hasattr(self, 'canvas_proc'):
            self.canvas_proc.delete("measurement")
        self.measuring_points = []
        self.measuring = False

    def on_closing(self):
        """Método chamado quando a janela é fechada"""
        try:
            # Para a câmera se estiver rodando
            self.running = False
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()

            # Fecha todas as figuras do matplotlib
            plt.close('all')

            # Limpa cache do torch se disponível
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Erro ao fechar recursos: {e}")
        finally:
            # Força fechamento da janela
            self.root.quit()
            self.root.destroy()

    def load_image(self):
        try:
            path = filedialog.askopenfilename(
                title="Selecione uma imagem",
                filetypes=[("Imagens", "*.jpg *.png *.jpeg *.bmp *.tiff")]
            )
            if path:
                self.image = cv2.imread(path)
                if self.image is None:
                    messagebox.showerror("Erro", "Não foi possível carregar a imagem. O arquivo pode estar corrompido ou em um formato não suportado.")
                    return
                self.processed_image = None
                self.depth_map_raw = None
                self.show_image(self.image, processed=False)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")
            logging.error(f"Erro ao carregar imagem: {str(e)}")

    def process_image(self):
        if self.image is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro.")
            return

        def process():
            try:
                self.progress_var.set(0)
                self.root.update()

                # Verifica se a imagem não é muito grande
                if self.image.size > 50000000:  # ~50MB
                    logging.warning("Imagem muito grande, pode consumir muita memória")

                # Processamento principal
                depth_map, depth_raw = self.run_midas(self.image, use_strong=True)
                self.progress_var.set(50)
                self.root.update()

                if depth_map is None or depth_raw is None:
                    raise RuntimeError("Falha ao gerar mapa de profundidade")

                self.depth_map_raw = depth_raw
                self.processed_image = self.apply_adjustments(depth_map)
                self.progress_var.set(100)

                # Atualiza visualizações
                self.show_image(self.image, processed=False)
                self.show_image(self.processed_image, processed=True)
                self.update_histogram()

            except MemoryError as e:
                self.progress_var.set(0)
                messagebox.showerror("Erro de Memória", "Memória insuficiente para processar a imagem. Tente uma imagem menor.")
                logging.error(f"Erro de memória: {str(e)}")
            except torch.cuda.OutOfMemoryError:
                self.progress_var.set(0)
                messagebox.showerror("Erro de GPU", "Memória GPU insuficiente. Tente uma imagem menor ou use CPU.")
                logging.error("Erro de memória GPU")
            except Exception as e:
                self.progress_var.set(0)
                messagebox.showerror("Erro", f"Erro durante o processamento: {str(e)}")
                logging.error(f"Erro no processamento: {str(e)}")

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
        try:
            if img_bgr is None or img_bgr.size == 0:
                raise ValueError("Imagem inválida ou vazia")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            model = self.model_strong if use_strong else self.model_fast
            transform = self.transform_strong if use_strong else self.transform_fast

            # Verifica dimensões da imagem
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

    def show_image(self, img, processed=False):
        if img is None:
            return

        # Cria uma cópia para não modificar a imagem original
        img_display = img.copy()

        # Converte para RGB se necessário
        if len(img_display.shape) == 2:
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        if processed:
            # Para imagem processada, usar Canvas com tamanho exato da imagem
            h, w = img_rgb.shape[:2]

            # Redimensiona apenas se muito grande
            max_width, max_height = 800, 600
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_rgb
                new_w, new_h = w, h

            # Configura Canvas com tamanho maior para centralização
            canvas_width, canvas_height = 900, 700
            self.canvas_proc.config(width=canvas_width, height=canvas_height)

            # Calcula posição central
            center_x = (canvas_width - new_w) // 2
            center_y = (canvas_height - new_h) // 2

            # Converte para formato Tkinter
            pil_image = Image.fromarray(img_resized)
            self.tk_image_proc = ImageTk.PhotoImage(pil_image)

            # Salva offset para cálculo de coordenadas
            self.image_offset = (center_x, center_y)
            self.image_size = (new_w, new_h)

            # Limpa Canvas e desenha imagem centralizada
            self.canvas_proc.delete("all")
            self.canvas_proc.create_image(center_x, center_y, anchor='nw', image=self.tk_image_proc)

            # Bind clique no Canvas
            self.canvas_proc.bind("<Button-1>", self.on_click)

        else:
            # Para imagem original, usar Label normal
            h, w = img_rgb.shape[:2]
            max_width, max_height = 600, 400
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            pil_image = Image.fromarray(img_resized)
            tk_image = ImageTk.PhotoImage(pil_image)
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

        # Salvar automaticamente o gráfico 3D
        output_dir = os.path.join("src", "assets", "output")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"3d_visualization_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Sucesso", f"Visualização 3D salva automaticamente em:\n{filepath}")

        fig.canvas.draw_idle()  # Força uma renderização inicial
        plt.show()

    def visualizar_contornos(self):
        if self.depth_map_raw is None:
            return
        plt.close('all')  # Fecha plots anteriores
        plt.figure(figsize=(10, 8))
        plt.contourf(self.depth_map_raw, levels=30, cmap='viridis')
        plt.colorbar(label='Profundidade')
        plt.title("Curvas de Nível da Profundidade")
        plt.tight_layout()

        # Salvar automaticamente o gráfico de contornos
        output_dir = os.path.join("src", "assets", "output")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"contour_visualization_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Sucesso", f"Visualização de contornos salva automaticamente em:\n{filepath}")

        plt.show()

    def _save_image_to_output(self, image, prefix="image"):
        """Método auxiliar para salvar imagens padronizadamente na pasta output"""
        if image is None:
            return False

        # Criar o diretório de saída se não existir
        output_dir = os.path.join("src", "assets", "output")
        os.makedirs(output_dir, exist_ok=True)

        # Gerar nome único baseado na data/hora
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        path = os.path.join(output_dir, filename)

        # Salvar a imagem
        success = cv2.imwrite(path, image)
        if success:
            messagebox.showinfo("Sucesso", f"Imagem salva automaticamente em:\n{path}")
            return True
        else:
            messagebox.showerror("Erro", "Falha ao salvar a imagem")
            return False

    def save_result(self):
        if self.processed_image is None:
            return
        self._save_image_to_output(self.processed_image, "depth_result")

    def capture_camera(self):
        """Captura a imagem atual da câmera e salva na pasta output"""
        if not self.running or not hasattr(self, 'cap') or not self.cap:
            messagebox.showwarning("Aviso", "Câmera não está ativa")
            return

        # Capturar frame atual
        ret, frame = self.cap.read()
        if ret:
            self._save_image_to_output(frame, "camera_capture")
        else:
            messagebox.showerror("Erro", "Falha ao capturar imagem da câmera")

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
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        # Aguarda um pouco para a thread finalizar
        time.sleep(0.1)

    def camera_loop(self):
        try:
            while self.running and hasattr(self, 'cap') and self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (320, 240))
                result, raw = self.run_midas(frame, use_strong=False)
                self.depth_map_raw = raw

                # Verifica se ainda está rodando antes de atualizar
                if self.running:
                    self.root.after(0, lambda f=frame, r=result: (
                        self.show_image(f, processed=False),
                        self.show_image(r, processed=True)
                    ))
                else:
                    break

        except Exception as e:
            logging.error(f"Erro na câmera: {e}")
        finally:
            # Garante que a câmera seja liberada
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None

def main():
    root = ttk.Window(themename="minty")
    app = TopoDepthApp(root)

    # Configura o protocolo de fechamento da janela
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    try:
        root.mainloop()
    finally:
        # Garante limpeza de recursos
        try:
            app.on_closing()
        except:
            pass

if __name__ == "__main__":
    main()
