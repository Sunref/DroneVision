# MiDepth - Depth Estimation System

## Overview
This project is a depth estimation system that processes images and videos using MiDaS (Mixed Dense Architecture for Single-Image Depth Estimation). The system analyzes input media and generates depth maps, providing a comprehensive understanding of spatial relationships in the visual content.

## Features
- Real-time depth estimation for videos
- Integration with MiDaS for accurate depth perception
- Support for various input formats
- Visual representation of depth information

## Technologies
- Python
- MiDaS (Mixed Dense Architecture for Single-Image Depth Estimation)
- Computer Vision libraries

## Installation

### 1. Set up Python virtual environment

#### Linux (Debian) / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows (cmd):
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

> **Note:** Make sure `python3-tk` is installed on your system:
> 
> ```bash - for Debian
> sudo apt update
> sudo apt upgrade
> sudo apt install python3-tk
> ```

---

### 2. Upgrade pip and install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python
pip install pillow
pip install ttkbootstrap
pip install matplotlib
pip install timm
```

---

After this, run the main scripts as described below.

## Usage

- Run the desired Python script for image or video depth processing.
- Example:
  ```bash
  python main.py 
  ```
  
## Examples

[Coming soon: Example outputs and visualizations]

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).  
See the [LICENSE](LICENSE) file for details.

Copyright (C) 2025 Sunref

## Acknowledgments

- MiDaS team for their excellent work on depth estimation
- Contributors and maintainers of related computer vision libraries
