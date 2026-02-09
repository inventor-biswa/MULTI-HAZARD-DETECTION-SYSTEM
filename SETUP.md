# 🚀 Setup Guide - Multi-Hazard Detection System

Quick setup instructions after cloning this repository.

---

## 📋 Prerequisites

- **Python 3.8+** installed
- **Git** installed
- **USB Camera** (for testing detection)

---

## ⚡ Quick Setup (Windows)

### 1. Clone the Repository

```powershell
git clone git@github.com:inventor-biswa/MULTI-HAZARD-DETECTION-SYSTEM.git
cd MULTI-HAZARD-DETECTION-SYSTEM
```

### 2. Create Virtual Environment

```powershell
python -m venv hazard_env
.\hazard_env\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install opencv-python numpy torch torchvision pillow matplotlib tqdm scikit-learn onnxruntime onnx
```

### 4. Run Detection (if model exists)

```powershell
python scripts/detect_hazard.py --camera 0
```

---

## 🎯 Full Training Pipeline

If you need to train the model from scratch:

### 1. Record Training Videos

Record 5 videos (5-10 min each) and save to `videos/` folder:
- `NO_LEAKAGE.mp4` - Normal state
- `WATER_LEAKAGE.mp4` - Water dripping
- `OIL_LEAKAGE.mp4` - Oil/dark liquid
- `STEAM_LEAKAGE.mp4` - Steam/fog
- `FIRE.mp4` - Flame/fire

### 2. Extract Frames

```powershell
python scripts/extract_frames.py
```

### 3. Train Model

```powershell
python scripts/train_model.py --epochs 50
```

### 4. Export to ONNX

```powershell
python scripts/export_model.py
```

### 5. Test Detection

```powershell
python scripts/detect_hazard.py --camera 0
```

---

## 🍓 Raspberry Pi Deployment

See [raspberry_pi/RASPBERRY_PI_SETUP.md](raspberry_pi/RASPBERRY_PI_SETUP.md) for deployment instructions.

**Quick copy command:**
```powershell
scp models/hazard_detector.onnx pi@<PI_IP>:~/multi_hazard_detection/models/
scp models/hazard_detector.onnx.data pi@<PI_IP>:~/multi_hazard_detection/models/
```

---

## 📁 Project Structure

```
MULTI-HAZARD-DETECTION-SYSTEM/
├── models/                 # Trained models (ONNX)
├── scripts/                # Python scripts
│   ├── extract_frames.py   # Video → Frames
│   ├── train_model.py      # Training
│   ├── export_model.py     # ONNX export
│   └── detect_hazard.py    # Real-time detection
├── raspberry_pi/           # Pi deployment files
├── videos/                 # Training videos (not in repo)
├── dataset/                # Extracted frames (not in repo)
└── hazard_env/             # Virtual env (not in repo)
```

---

## ⚠️ Notes

- **Large files excluded:** Videos, dataset, and virtual environment are in `.gitignore`
- **Model included:** ONNX model is included for immediate testing
- **Training optional:** Only needed if you want to retrain with your own data

---

## 🔑 Keyboard Shortcuts (During Detection)

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `F` | Toggle fullscreen |

---

**Questions?** Check the [README.md](README.md) for more details.
