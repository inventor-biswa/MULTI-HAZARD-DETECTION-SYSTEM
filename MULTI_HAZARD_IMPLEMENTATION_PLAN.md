# Multi-Hazard Video Analytics System
## GMR 2025 Expo - Implementation Plan

---

## 🎯 Project Overview

Build a **multi-class video + audio analytics system** to detect industrial hazards using a controlled demo environment deployed on Raspberry Pi with USB camera and microphone.

### Video Detection Classes (5 classes)
| Class | Visual Trigger | Color/Appearance |
|-------|---------------|------------------|
| `normal` | Empty rig, no activity | Static background |
| `water_leak` | Dripping from pipe | **Clear/transparent** liquid |
| `oil_leak` | Dripping from pipe | **Dark brown/black** colored liquid |
| `steam_leak` | Fogger/smoke machine | White/gray diffuse cloud |
| `fire` | Lighter/candle | Orange/yellow flame |

### Audio Detection Categories
| Category | Trigger | Alert Level |
|----------|---------|-------------|
| `loud_bang` | Impact / explosion sounds | **Critical** |
| `alarm` | Siren / alarm sounds | **Critical** |
| `grinding` | Mechanical grinding | Warning |
| `hissing` | Gas leak / steam sounds | Warning |
| `abnormal` | Any sound above threshold | Warning |

---

## 🛠️ Hardware Requirements

| Component | Purpose | Qty |
|-----------|---------|-----|
| Raspberry Pi 4/5 | Edge inference | 1 |
| USB Camera (e.g., Logitech C270) | Video capture + Microphone | 1 |
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
│   ├── normal.mp4
│   ├── water_leak.mp4
│   ├── oil_leak.mp4
│   ├── steam_leak.mp4
│   └── fire.mp4
├── dataset/                     # Extracted frames
│   ├── train/
│   ├── val/
│   └── test/
├── models/                      # Trained models
│   ├── hazard_detector.pth     # PyTorch model
│   ├── hazard_detector.onnx    # ONNX for Raspberry Pi
│   ├── hazard_detector.onnx.data # Model weights
│   └── class_labels.json       # Class mapping
├── scripts/
│   ├── extract_frames.py       # Video → Frames
│   ├── train_model.py          # Train multi-class model
│   ├── export_model.py         # Export to ONNX
│   ├── detect_hazard.py        # Video-only real-time inference
│   ├── detect_sound.py         # Audio abnormal sound detection
│   └── detect_combined.py      # Combined video + audio detection
├── raspberry_pi/
│   ├── setup_pi.sh             # Automated Pi setup script
│   └── RASPBERRY_PI_SETUP.md   # Setup guide
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

Combined screen showing video + audio detections with color-coded alerts:

```
┌────────────────────────────────────────────────────────────┐
│ FPS: 15.2          Multi-Hazard Detection [V+A]           │
│   ┌──────────────────────────────────────────────────────┐ │
│   │                                                      │ │
│   │                  LIVE CAMERA FEED                    │ │
│   │                                                      │ │
│   └──────────────────────────────────────────────────────┘ │
│                                                            │
│   VIDEO: NORMAL              98%    AUDIO: Normal          │
│   ┌──────────── Audio Level Meter ──────────────────────┐  │
│   │████░░░░░░░░░░░░░░░░░░░░░░   Level: 1.2x  Freq: 0Hz │  │
│   └─────────────────────────────────────────────────────┘  │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│   │Normal│ │Water │ │ Oil  │ │Steam │ │ Fire │ │Sound │  │
│   └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
└────────────────────────────────────────────────────────────┘
```

### Alert Colors
| Detection | Color | Border |
|-----------|-------|--------|
| Normal | Green | Green border |
| Water Leak | Blue | Blue border |
| Oil Leak | Brown/Orange | Orange border |
| Steam Leak | Gray/White | Gray border |
| Fire | Red | **Flashing red border** |
| Sound (abnormal) | Yellow/Red | Flashing if critical |

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

### Phase 6: Audio Detection ✅
1. Create `detect_sound.py` — USB camera microphone based
2. Baseline calibration (5 sec quiet environment)
3. Frequency-band analysis for categorized alerts
4. Auto-detection of sample rate and channels per device

### Phase 7: Combined Detection ✅
1. Create `detect_combined.py` — video + audio in single display
2. Audio runs in background thread with retry logic
3. Unified UI with audio level meter and sound category indicators

---

## 🍓 Raspberry Pi Deployment

> See [RASPBERRY_PI_SETUP.md](raspberry_pi/RASPBERRY_PI_SETUP.md) for the full step-by-step guide.

### Automated Setup
```bash
# Transfer and run setup script (from Windows PowerShell)
scp "raspberry_pi/setup_pi.sh" test@<PI_IP>:~/Downloads/

# On the Pi
cd ~/Downloads
sed -i 's/\r$//' setup_pi.sh    # Fix Windows line endings
./setup_pi.sh                    # Installs everything (~5-10 min)
```

### Copy Model Files
```bash
cp ~/Downloads/hazard_detector.onnx ~/multi_hazard_detection/models/
cp ~/Downloads/hazard_detector.onnx.data ~/multi_hazard_detection/models/
```

### Run Detection
```bash
cd ~/multi_hazard_detection

# Find audio device index
./run_sound.sh --list-devices

# Combined video + audio (recommended)
./run_combined.sh --camera 0 --audio-device <MIC_INDEX> --skip 3

# Combined fullscreen (for demo/expo)
./run_combined.sh --camera 0 --audio-device <MIC_INDEX> --fullscreen

# Video only
./run_detection.sh --camera 0

# Audio only
./run_sound.sh --device <MIC_INDEX>
```

---

## 🔮 Future Enhancements

- Alert logging to file/database
- MQTT/HTTP alerts for IoT integration
- Web dashboard for remote monitoring
- Multiple camera support
- ML-based audio classification (replacing threshold-based)
- ESP32 NeoPixel alert indicators

---

## ⏱️ Estimated Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Setup & Environment | 30 min |
| 2 | Video Recording (all 5 classes) | 1-2 hours |
| 3 | Frame Extraction | 30 min |
| 4 | Model Training | 1-2 hours |
| 5 | Testing & Deployment | 1 hour |
| 6 | Audio Detection Module | 1-2 hours |
| 7 | Combined V+A Integration | 1 hour |
| **Total** | | **6-9 hours** |

---

## ✅ Success Criteria

- [x] Model achieves >95% accuracy on test set
- [x] Real-time inference at 10+ FPS on Raspberry Pi
- [x] Correctly distinguishes all 5 visual classes
- [x] Clear visual differentiation between water (clear) and oil (brown)
- [x] Audio abnormal sound detection with categorization
- [x] Combined video + audio display with unified UI
- [x] Automated Raspberry Pi setup script
- [x] USB device retry logic for reliable audio
- [ ] Responsive demo for GMR Expo

---

## 📞 Quick Start Commands

### Windows (Training & Testing)
```powershell
cd MULTI_HAZARD_DETECTION
.\hazard_env\Scripts\Activate.ps1

python scripts/extract_frames.py          # Video → Frames
python scripts/train_model.py             # Train model
python scripts/export_model.py            # Export to ONNX
python scripts/detect_combined.py --camera 0 --audio-device 1  # Test locally
```

### Raspberry Pi (Deployment)
```bash
cd ~/multi_hazard_detection
./run_combined.sh --camera 0 --audio-device 2 --skip 3           # Combined
./run_combined.sh --camera 0 --audio-device 2 --fullscreen       # Fullscreen
./run_detection.sh --camera 0                                     # Video only
./run_sound.sh --list-devices                                     # Find mic
```

---

**Created:** February 2026  
**Project:** GMR 2025 Expo - Video Analytics Demo  
**Author:** AI-Assisted Development
