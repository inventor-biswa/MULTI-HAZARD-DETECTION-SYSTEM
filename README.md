# Multi-Hazard Video Analytics System

## GMR 2025 Expo Demo

A 5-class video analytics system for industrial hazard detection, designed for Raspberry Pi deployment.

---

## 🎯 Detection Classes

| Class | Description | Visual Indicator |
|-------|-------------|------------------|
| `normal` | No hazard detected | Green border |
| `water_leak` | Clear water dripping | Blue border |
| `oil_leak` | Dark brown/black liquid | Orange border |
| `steam_leak` | White/gray fog/steam | Gray border |
| `fire` | Flame/fire detected | **Flashing red border** |

---

## 📁 Project Structure

```
MULTI_HAZARD_DETECTION/
├── videos/                      # Raw training videos
│   ├── FIRE.mp4
│   ├── NO_LEAKAGE.mp4
│   ├── OIL_LEAKAGE.mp4
│   ├── STEAM_LEAKAGE.mp4
│   └── WATER_LEAKAGE.mp4
├── dataset/                     # Extracted frames (auto-generated)
│   ├── train/
│   ├── val/
│   └── test/
├── models/                      # Trained models
│   ├── hazard_detector.pth      # PyTorch checkpoint
│   ├── hazard_detector.onnx     # ONNX for deployment
│   └── class_labels.json        # Class mapping
├── scripts/
│   ├── extract_frames.py        # Video → Frames
│   ├── train_model.py           # Train model
│   ├── export_model.py          # Export to ONNX
│   └── detect_hazard.py         # Real-time inference
├── hazard_env/                  # Virtual environment
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment

```powershell
cd MULTI_HAZARD_DETECTION
python -m venv hazard_env
.\hazard_env\Scripts\Activate.ps1
pip install opencv-python numpy torch torchvision pillow matplotlib tqdm scikit-learn onnxruntime onnx
```

### 2. Extract Frames from Videos

```powershell
python scripts/extract_frames.py
```

This extracts frames from all 5 training videos and splits them into train/val/test sets.

### 3. Train the Model

```powershell
python scripts/train_model.py
```

Options:
- `--epochs 50` - Number of training epochs (default: 50)
- `--batch-size 32` - Batch size (default: 32)
- `--lr 0.001` - Learning rate (default: 0.001)

### 4. Export to ONNX

```powershell
python scripts/export_model.py
```

This creates `hazard_detector.onnx` for Raspberry Pi deployment.

### 5. Run Detection

```powershell
# From USB camera
python scripts/detect_hazard.py --camera 0

# From video file
python scripts/detect_hazard.py --video path/to/video.mp4

# With frame skipping for better FPS
python scripts/detect_hazard.py --camera 0 --skip 2
```

---

## 🍓 Raspberry Pi Deployment

### Files to Copy

```
models/hazard_detector.onnx
models/class_labels.json
scripts/detect_hazard.py
```

### Installation on Pi

```bash
pip install onnxruntime opencv-python numpy
```

### Run

```bash
python detect_hazard.py --camera 0 --skip 2 --delay 150
```

---

## ⌨️ Keyboard Shortcuts (Detection Mode)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `f` | Toggle fullscreen |

---

## 📊 Model Specifications

- **Architecture**: MobileNetV2 (pretrained on ImageNet)
- **Input Size**: 224x224 pixels
- **Classes**: 5
- **Target Accuracy**: >95%
- **Expected FPS on Pi**: 10-15

---

## ✅ Success Criteria

- [ ] Model achieves >95% test accuracy
- [ ] Real-time inference at 10+ FPS on Raspberry Pi
- [ ] Correctly distinguishes all 5 classes
- [ ] Clear visual differentiation between water and oil leaks
- [ ] Responsive demo for GMR Expo

---

**Created:** February 2026  
**Project:** GMR 2025 Expo - Video Analytics Demo
