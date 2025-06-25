# ⚡ Setup

## 🚀 Configuration

# 1. Create virtual environment
```bash
python3 -m venv venv                # Linux/macOS

python -m venv venv                 # Windows
```

# 2. Activate environment
```bash
source venv/bin/activate            # Linux/macOS

venv\Scripts\activate               # Windows
```

# 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔄 Daily Usage

### Activate environment:
```bash
source venv/bin/activate            # Linux/macOS
venv\Scripts\activate               # Windows
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
MiDepth/
├── venv/                           # Virtual environment
├── src/core/engine/main.py
├── requirements.txt                # Dependencies
```
