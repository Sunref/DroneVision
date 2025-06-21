# Interface gráfica do TopoDepthApp

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import numpy as np
import cv2

from .depth_engine import DepthEngine
from .utils import update_metrics, apply_adjustments

class TopoDepthApp:
    def __init__(self, root):
        self.root = root
        self.auto_detect = False
        self.root.title("Detecção de Profundidade com MiDaS")
        self.root.geometry("1720x900")

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

        # Engine de profundidade
        self.engine = DepthEngine()

        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        btn_row1 = ttk.Frame(button_frame)
        btn_row1.pack(fill='x', pady=5)
        load_btn = ttk.Button(btn_row1, text="Carregar Imagem", command=self.load_image, bootstyle="success-outline")
        load_btn.pack(side='left', padx=5)
        ToolTip(load_btn, "Carregar uma nova imagem para processamento")
        process_btn = ttk.Button(btn_row1, text="Processar", command=self.process_image, bootstyle="success-outline")
        process_btn.pack(side='left', padx=5)
        ToolTip(process_btn, "Processar a imagem usando MiDaS")
        save_btn = ttk.Button(btn_row1, text="Salvar Resultado", command=self.save_result, bootstyle="success-outline")
        save_btn.pack(side='left', padx=5)
        ToolTip(save_btn, "Salvar a imagem processada")
        ttk.Button(btn_row1, text="Iniciar Câmera", command=self.start_camera, bootstyle="success-outline").pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Parar Câmera", command=self.stop_camera, bootstyle="danger-outline").pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Visualização 3D", command=self.visualizar_3d, bootstyle="info-outline").pack(side='left', padx=5)
        ttk.Button(btn_row1, text="Curvas de Nível", command=self.visualizar_contornos, bootstyle="info-outline").pack(side='left', padx=5)
        btn_row2 = ttk.Frame(button_frame)
        btn_row2.pack(fill='x', pady=5)
        measure_btn = ttk.Button(btn_row2, text="Medir Distância", command=self.toggle_measurement, bootstyle="info-outline")
        measure_btn.pack(side='left', padx=5)
        ToolTip(measure_btn, "Medir distância relativa entre pontos")
        clear_line_btn = ttk.Button(btn_row2, text="Limpar Linha", command=self.clear_measurement_line, bootstyle="warning-outline")
        clear_line_btn.pack(side='left', padx=5)
        ToolTip(clear_line_btn, "Limpar linha de medição")
        filter_frame = ttk.LabelFrame(btn_row2, text="Filtros", bootstyle="success")
        filter_frame.pack(side='left', padx=20)
        ttk.Label(filter_frame, text="Suavização:").pack(side='left', padx=5)
        self.smooth_var = ttk.DoubleVar(value=0)
        smooth_scale = ttk.Scale(filter_frame, from_=0, to=5, variable=self.smooth_var, command=self.apply_filters, length=100)
        smooth_scale.pack(side='left', padx=5)
        ToolTip(smooth_scale, "Ajustar nível de suavização")
        self.edge_var = ttk.BooleanVar(value=False)
        edge_check = ttk.Checkbutton(filter_frame, text="Realce de Bordas", variable=self.edge_var, command=self.apply_filters, bootstyle="round-toggle")
        edge_check.pack(side='left', padx=5)
        ToolTip(edge_check, "Ativar/Desativar realce de bordas")
        ttk.Label(btn_row2, text="Colormap:").pack(side='left', padx=(20, 5))
        self.colormap_var = ttk.Combobox(btn_row2, values=["Magma", "Inferno", "Plasma", "Viridis", "Cividis", "Jet", "Turbo", "Hot"], state="readonly", width=12)
        self.colormap_var.current(0)
        self.colormap_var.pack(side='left', padx=5)
        ttk.Label(btn_row2, text="Brilho:").pack(side='left', padx=(20, 5))
        self.brightness_scale = ttk.Scale(btn_row2, from_=0.5, to=2.0, value=1.0, command=self.update_processed_image, length=100)
        self.brightness_scale.pack(side='left', padx=5)
        ttk.Label(btn_row2, text="Contraste:").pack(side='left', padx=(10, 5))
        self.contrast_scale = ttk.Scale(btn_row2, from_=0.5, to=3.0, value=1.0, command=self.update_processed_image, length=100)
        self.contrast_scale.pack(side='left', padx=5)
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100, bootstyle="success")
        self.progress_bar.pack(fill='x', pady=5)
        self.metrics_label = ttk.Label(main_frame, text="Mín: -  Máx: -  Média: -", bootstyle="info")
        self.metrics_label.pack(fill='x', pady=(0, 10))
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(fill='both', expand=True, side='left')
        frame_pre = ttk.LabelFrame(image_frame, text="Imagem Pré-processamento", bootstyle="info")
        frame_pre.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.label_orig = ttk.Label(frame_pre)
        self.label_orig.pack(fill='both', expand=True)
        frame_post = ttk.LabelFrame(image_frame, text="Imagem Pós-processamento", bootstyle="success")
        frame_post.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.canvas_proc = Canvas(frame_post, bg='black')
        self.canvas_proc.pack(fill='both', expand=True)
        self.analysis_frame = ttk.Frame(content_frame)
        self.analysis_frame.pack(fill='both', expand=True, side='right', padx=(10, 0))
        self.setup_histogram()

    def setup_histogram(self):
        fig, self.ax_hist = plt.subplots(figsize=(6, 4))
        self.histogram_canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
        self.histogram_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.stats_frame = ttk.LabelFrame(self.analysis_frame, text="Estatísticas", bootstyle=ttk.INFO)
        self.stats_frame.pack(fill='x', pady=5)
        self.stats_label = ttk.Label(self.stats_frame, text="")
        self.stats_label.pack(pady=5)

    def update_histogram(self):
        if self.depth_map_raw is None:
            return
        self.ax_hist.clear()
        self.ax_hist.hist(self.depth_map_raw.flatten(), bins=50, color='blue', alpha=0.7)
        self.ax_hist.set_title("Distribuição de Profundidade")
        self.ax_hist.set_xlabel("Profundidade")
        self.ax_hist.set_ylabel("Frequência")
        self.histogram_canvas.draw()
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
        if self.depth_map_raw is None:
            return
        smoothed = cv2.GaussianBlur(self.depth_map_raw, (0, 0), self.smooth_var.get())
        if self.edge_var.get():
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            smoothed = cv2.filter2D(smoothed, -1, kernel)
        depth_norm = ((smoothed - smoothed.min()) / (smoothed.max() - smoothed.min()) * 255).astype(np.uint8)
        self.processed_image = cv2.applyColorMap(depth_norm, getattr(cv2, f'COLORMAP_{self.colormap_var.get().upper()}'))
        self.show_image(self.processed_image, processed=True)
        self.update_histogram()

    def toggle_measurement(self):
        self.measuring = not getattr(self, 'measuring', False)
        if self.measuring:
            self.measuring_points = []
            messagebox.showinfo("Medição", "Clique em dois pontos na imagem processada para medir a distância relativa")

    def on_click(self, event):
        if self.depth_map_raw is None:
            return
        if hasattr(self, 'image_offset') and hasattr(self, 'image_size'):
            offset_x, offset_y = self.image_offset
            img_w, img_h = self.image_size
            x = event.x - offset_x
            y = event.y - offset_y
            if x < 0 or x >= img_w or y < 0 or y >= img_h:
                return
        else:
            x, y = event.x, event.y
        if self.measuring:
            self.measuring_points.append((x, y))
            if len(self.measuring_points) == 2:
                self.calculate_distance()
                self.measuring_points = []

    def calculate_distance(self):
        if len(self.measuring_points) != 2:
            return
        x1_img, y1_img = self.measuring_points[0]
        x2_img, y2_img = self.measuring_points[1]
        img_h, img_w = self.processed_image.shape[:2]
        x1_img = max(0, min(x1_img, img_w - 1))
        y1_img = max(0, min(y1_img, img_h - 1))
        x2_img = max(0, min(x2_img, img_w - 1))
        y2_img = max(0, min(y2_img, img_h - 1))
        depth_h, depth_w = self.depth_map_raw.shape
        x1_depth = int((x1_img / img_w) * depth_w)
        y1_depth = int((y1_img / img_h) * depth_h)
        x2_depth = int((x2_img / img_w) * depth_w)
        y2_depth = int((y2_img / img_h) * depth_h)
        x1_depth = max(0, min(x1_depth, depth_w - 1))
        y1_depth = max(0, min(y1_depth, depth_h - 1))
        x2_depth = max(0, min(x2_depth, depth_w - 1))
        y2_depth = max(0, min(y2_depth, depth_h - 1))
        depth1 = self.depth_map_raw[y1_depth, x1_depth]
        depth2 = self.depth_map_raw[y2_depth, x2_depth]
        distance = np.sqrt((x2_depth-x1_depth)**2 + (y2_depth-y1_depth)**2 + (depth2-depth1)**2)
        self.canvas_proc.delete("measurement")
        if hasattr(self, 'image_offset'):
            offset_x, offset_y = self.image_offset
            draw_x1 = x1_img + offset_x
            draw_y1 = y1_img + offset_y
            draw_x2 = x2_img + offset_x
            draw_y2 = y2_img + offset_y
        else:
            draw_x1, draw_y1 = x1_img, y1_img
            draw_x2, draw_y2 = x2_img, y2_img
        self.canvas_proc.create_oval(draw_x1-5, draw_y1-5, draw_x1+5, draw_y1+5, fill='red', outline='red', tags="measurement")
        self.canvas_proc.create_oval(draw_x2-5, draw_y2-5, draw_x2+5, draw_y2+5, fill='red', outline='red', tags="measurement")
        self.canvas_proc.create_line(draw_x1, draw_y1, draw_x2, draw_y2, fill='red', width=3, tags="measurement")
        text_x = (draw_x1 + draw_x2) // 2
        text_y = (draw_y1 + draw_y2) // 2
        self.canvas_proc.create_text(text_x, text_y, text=f"D: {distance:.1f}", fill='red', font=("Arial", 12, "bold"), tags="measurement")
        msg = f"Distância relativa: {distance:.2f}\nDiferença de profundidade: {abs(depth2-depth1):.2f}"
        messagebox.showinfo("Medição", msg)

    def clear_measurement_line(self):
        if hasattr(self, 'canvas_proc'):
            self.canvas_proc.delete("measurement")
        self.measuring_points = []
        self.measuring = False

    def on_closing(self):
        try:
            self.running = False
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            plt.close('all')
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        finally:
            self.root.quit()
            self.root.destroy()

    def load_image(self):
        try:
            path = filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg *.png *.jpeg *.bmp *.tiff")])
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

    def process_image(self):
        if self.image is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro.")
            return
        def process():
            try:
                self.progress_var.set(0)
                self.root.update()
                if self.image.size > 50000000:
                    pass
                depth_map, depth_raw = self.engine.run_midas(self.image, use_strong=True)
                self.progress_var.set(50)
                self.root.update()
                if depth_map is None or depth_raw is None:
                    raise RuntimeError("Falha ao gerar mapa de profundidade")
                self.depth_map_raw = depth_raw
                self.processed_image = apply_adjustments(depth_map, self.brightness_scale.get(), self.contrast_scale.get(), self.colormap_var.get())
                self.progress_var.set(100)
                self.show_image(self.image, processed=False)
                self.show_image(self.processed_image, processed=True)
                self.update_histogram()
            except Exception as e:
                self.progress_var.set(0)
                messagebox.showerror("Erro", f"Erro durante o processamento: {str(e)}")
        threading.Thread(target=process, daemon=True).start()

    def update_processed_image(self, _=None):
        if self.depth_map_raw is None:
            return
        depth_min = self.depth_map_raw.min()
        depth_max = self.depth_map_raw.max()
        depth_norm = (255 * (self.depth_map_raw - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        depth_colored = apply_adjustments(depth_norm, self.brightness_scale.get(), self.contrast_scale.get(), self.colormap_var.get())
        self.processed_image = depth_colored
        update_metrics(self.depth_map_raw, self.metrics_label)
        self.show_image(depth_colored, processed=True)

    def show_image(self, img, processed=False):
        if img is None:
            return
        img_display = img.copy()
        if len(img_display.shape) == 2:
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        if processed:
            h, w = img_rgb.shape[:2]
            max_width, max_height = 800, 600
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_rgb
                new_w, new_h = w, h
            canvas_width, canvas_height = 900, 700
            self.canvas_proc.config(width=canvas_width, height=canvas_height)
            center_x = (canvas_width - new_w) // 2
            center_y = (canvas_height - new_h) // 2
            pil_image = Image.fromarray(img_resized)
            self.tk_image_proc = ImageTk.PhotoImage(pil_image)
            self.image_offset = (center_x, center_y)
            self.image_size = (new_w, new_h)
            self.canvas_proc.delete("all")
            self.canvas_proc.create_image(center_x, center_y, anchor='nw', image=self.tk_image_proc)
            self.canvas_proc.bind("<Button-1>", self.on_click)
        else:
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
        scale = 8
        h, w = self.depth_map_raw.shape
        reduced_depth = self.depth_map_raw[::scale, ::scale]
        y, x = np.mgrid[0:h:scale, 0:w:scale]
        plt.close('all')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, reduced_depth, cmap='inferno', rstride=3, cstride=3, linewidth=0, antialiased=False, shade=True)
        ax.view_init(elev=30, azim=45)
        ax.set_title("Mapa 3D de Profundidade")
        plt.tight_layout()
        fig.canvas.draw_idle()
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
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
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
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        time.sleep(0.1)

    def camera_loop(self):
        try:
            while self.running and hasattr(self, 'cap') and self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (320, 240))
                result, raw = self.engine.run_midas(frame, use_strong=False)
                self.depth_map_raw = raw
                if self.running:
                    self.root.after(0, lambda f=frame, r=result: (self.show_image(f, processed=False), self.show_image(r, processed=True)))
                else:
                    break
        except Exception as e:
            print(f"Erro na câmera: {e}")
        finally:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
