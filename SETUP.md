# ⚡ Setup

## 🚀 Configuration

# 1. Create virtual environment
python3 -m venv venv                # Linux/macOS
# python -m venv venv               # Windows

# 2. Activate environment
source venv/bin/activate            # Linux/macOS
# venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

## 🔄 Daily Usage

### Activate environment:
```bash
# Linux/macOS
source venv/bin/activate

# venv\Scripts\activate             # Windows
```

### Run application:
```bash
python src/core/engine/main.py
```

### Deactivate environment:
```bash
deactivate
```

## 📂 Final Structure

```
DroneVision/
├── venv/                           # Virtual environment (NOT for Git)
├── src/assets/output/              # Images saved automatically
├── requirements.txt                # Dependencies (GOES to Git)
├── setup_new_computer.py           # Automatic setup
├── test_environment.py             # Environment test
├── activate_dronevision.sh         # Linux/macOS script
└── activate_dronevision.bat        # Windows script
```

## ⚠️ Important

- **NEVER** add `venv/` to Git
- **ALWAYS** update `requirements.txt` after installing new packages
- **TEST** the environment after each sync
