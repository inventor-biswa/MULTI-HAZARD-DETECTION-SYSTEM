# Multi-Hazard Video Analytics System
## GMR 2025 Expo - Implementation Plan

---

## 🎯 Project Overview

Build a **single multi-class video analytics system** to detect industrial hazards using a controlled demo environment deployed on Raspberry Pi with USB camera.

### Detection Classes (5 classes)
| Class | Visual Trigger | Color/Appearance |
|-------|---------------|------------------|
| `normal` | Empty rig, no activity | Static background |
| `water_leak` | Dripping from pipe | **Clear/transparent** liquid |
| `oil_leak` | Dripping from pipe | **Dark brown/black** colored liquid |
| `steam_leak` | Fogger/smoke machine | White/gray diffuse cloud |
| `fire` | Lighter/candle | Orange/yellow flame |

> **Note:** Audio detection (abnormal sound) will be added as a separate module later.

---

## 🛠️ Hardware Requirements

| Component | Purpose | Qty |
|-----------|---------|-----|
| Raspberry Pi 4 (8GB+) | Edge inference | 1 |
| USB Camera | Video capture | 1 |
| PVC Pipe Frame | Water/Oil leak demo | 1 |
| Fogger/Smoke Machine | Steam simulation | 1 |
| Lighter/Candle | Fire simulation | 1 |
| Brown/Black Food Coloring | Oil simulation | 1 |
| Cardboard Background | Consistent backdrop | 1 |
| Container/Tray | Collect liquids | 1 |

---

## 📁 Project Folder Structure

```
MULTI_HAZARD_DETECTION/
├── videos/                      # Raw training videos
│   ├── normal.mp4              # 5-10 min empty rig
│   ├── water_leak.mp4          # 5-10 min clear water dripping
│   ├── oil_leak.mp4            # 5-10 min brown/black liquid
│   ├── steam_leak.mp4          # 5-10 min fogger running
│   └── fire.mp4                # 5-10 min lighter/candle flame
├── dataset/                     # Extracted frames
│   ├── train/
│   │   ├── normal/
│   │   ├── water_leak/
│   │   ├── oil_leak/
│   │   ├── steam_leak/
│   │   └── fire/
│   ├── val/
│   └── test/
├── models/                      # Trained models
│   ├── hazard_detector.pth     # PyTorch model
│   ├── hazard_detector.onnx    # ONNX for Raspberry Pi
│   └── class_labels.json       # Class mapping
├── scripts/
│   ├── extract_frames.py       # Video → Frames
│   ├── train_model.py          # Train multi-class model
│   ├── export_model.py         # Export to ONNX
│   └── detect_hazard.py        # Real-time inference
├── hazard_env/                  # Virtual environment
└── README.md
```

---

## 🎬 Video Recording Guidelines

### General Rules
- **Duration:** 5-10 minutes per class
- **Camera position:** Fixed, consistent angle
- **Background:** Same cardboard/backdrop for all
- **Lighting:** Consistent indoor lighting
- **Resolution:** 640x480 or 1280x720

### Per-Class Recording Instructions

| Class | Setup | Tips |
|-------|-------|------|
| `normal` | Empty rig, no activity | Record from different times of day for lighting variation |
| `water_leak` | Clear water dripping from pipe | Vary drip speed (slow, medium, fast) |
| `oil_leak` | Brown/black colored water | Use food coloring, make it visibly different from clear water |
| `steam_leak` | Fogger running behind/near pipe | Capture steam at different densities |
| `fire` | Lighter held in frame | Move flame slightly, capture from different angles |

---

## 🔧 Technical Specifications

### Model Architecture
- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Classifier:** 5-class output (normal, water_leak, oil_leak, steam_leak, fire)
- **Input Size:** 224x224 pixels
- **Framework:** PyTorch → ONNX for deployment

### Training Parameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 30-50 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Train/Val/Test Split | 70/15/15 |

### Data Augmentation
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast)
- Random affine transforms

---

## 📺 Display Design

Single screen showing all detections with color-coded alerts:

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│   ┌──────────────────────────────────────────────────────┐ │
│   │                                                      │ │
│   │                  LIVE CAMERA FEED                    │ │
│   │                                                      │ │
│   │                                                      │ │
│   └──────────────────────────────────────────────────────┘ │
│                                                            │
│   ┌────────────────────────────────────────────────────┐   │
│   │  STATUS: ⚠️ FIRE DETECTED!          Confidence: 98% │   │
│   └────────────────────────────────────────────────────┘   │
│                                                            │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐            │
│   │ ✓    │ │  ○   │ │  ○   │ │  ○   │ │ ⚠️   │            │
│   │Normal│ │Water │ │ Oil  │ │Steam │ │ Fire │            │
│   └──────┘ └──────┘ └──────┘ └──────┘ └──────┘            │
│                                                            │
│                          FPS: 15.2                         │
└────────────────────────────────────────────────────────────┘
```

### Alert Colors
| Detection | Color | Border |
|-----------|-------|--------|
| Normal | Green | None |
| Water Leak | Blue | Blue border |
| Oil Leak | Brown/Orange | Orange border |
| Steam Leak | Gray/White | Gray border |
| Fire | Red | **Flashing red border** |

---

## 📋 Implementation Steps

### Phase 1: Environment Setup
```powershell
mkdir MULTI_HAZARD_DETECTION
cd MULTI_HAZARD_DETECTION
python -m venv hazard_env
.\hazard_env\Scripts\Activate.ps1
pip install opencv-python numpy torch torchvision pillow matplotlib tqdm scikit-learn onnxruntime onnx onnxscript
```

### Phase 2: Video Recording
1. Set up demo rig with consistent background
2. Record 5 videos (one per class, 5-10 min each)
3. Save to `videos/` folder

### Phase 3: Frame Extraction
1. Create `extract_frames.py` with 5 video inputs
2. Extract frames (every 15th frame)
3. Auto-split into train/val/test

### Phase 4: Model Training
1. Create `train_model.py` for 5-class classification
2. Train for 30-50 epochs
3. Target: >95% test accuracy

### Phase 5: Export & Deployment
1. Export to ONNX format
2. Create `detect_hazard.py` with multi-class display
3. Test on Raspberry Pi with USB camera

---

## 🍓 Raspberry Pi Deployment

### Files to Copy
```
models/hazard_detector.onnx
models/hazard_detector.onnx.data
models/class_labels.json
scripts/detect_hazard.py
```

### Installation
```bash
pip install onnxruntime opencv-python numpy
```

### Run Detection
```bash
python detect_hazard.py --camera 0 --skip 2 --delay 150
```

---

## 🔮 Future Additions

### Audio Detection Module (Phase 2)
- Sound sensor connected to Raspberry Pi
- Threshold-based clap/loud sound detection
- OR ML-based audio classification
- Separate process running alongside video detection

### Potential Enhancements
- Alert logging to file/database
- MQTT/HTTP alerts for IoT integration
- Web dashboard for remote monitoring
- Multiple camera support

---

## ⏱️ Estimated Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Setup & Environment | 30 min |
| 2 | Video Recording (all 5 classes) | 1-2 hours |
| 3 | Frame Extraction | 30 min |
| 4 | Model Training | 1-2 hours |
| 5 | Testing & Deployment | 1 hour |
| **Total** | | **4-6 hours** |

---

## ✅ Success Criteria

- [ ] Model achieves >95% accuracy on test set
- [ ] Real-time inference at 10+ FPS on Raspberry Pi
- [ ] Correctly distinguishes all 5 classes
- [ ] Clear visual differentiation between water (clear) and oil (brown)
- [ ] Responsive demo for GMR Expo

---

## 📞 Quick Start Commands

```powershell
# 1. Setup
cd MULTI_HAZARD_DETECTION
.\hazard_env\Scripts\Activate.ps1

# 2. Extract frames (after recording videos)
python scripts/extract_frames.py

# 3. Train model
python scripts/train_model.py

# 4. Export to ONNX
python scripts/export_model.py

# 5. Run detection
python scripts/detect_hazard.py --camera 0
```

---

**Created:** February 2026  
**Project:** GMR 2025 Expo - Video Analytics Demo  
**Author:** AI-Assisted Development
