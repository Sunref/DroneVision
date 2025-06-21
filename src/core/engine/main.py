from ttkbootstrap import Window
from src.core.modules.ui import TopoDepthApp

def main():
    root = Window(themename="minty")
    app = TopoDepthApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    try:
        root.mainloop()
    finally:
        try:
            app.on_closing()
        except:
            pass

if __name__ == "__main__":
    main()
