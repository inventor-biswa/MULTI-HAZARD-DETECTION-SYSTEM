# Multi-Hazard Video + Audio Analytics System
## Client Deliverable Document

**Project:** GMR 2025 Expo - Industrial Hazard Detection Demo  
**Date:** February 10, 2026  
**Prepared by:** Thynx Development Team

---

## 📋 Executive Summary

We have developed a **real-time video + audio analytics system** for industrial hazard detection, designed for deployment on edge devices (Raspberry Pi 5). The system detects 5 visual hazard classes from a live camera feed and 5 categories of abnormal sounds from the USB camera microphone, providing immediate visual and audio alerts in a unified display.

---

## ✅ Deliverables Completed

### 1. Machine Learning Model

| Component | Status | Details |
|-----------|--------|---------|
| Model Architecture | ✅ Complete | MobileNetV2 (pretrained on ImageNet) |
| Training Dataset | ✅ Complete | 5 classes, extracted from demo videos |
| PyTorch Model | ✅ Complete | `hazard_detector.pth` (25.5 MB) |
| ONNX Model | ✅ Complete | `hazard_detector.onnx` (8.9 MB) - optimized for edge deployment |
| Class Labels | ✅ Complete | `class_labels.json` |

### 2. Detection Capabilities

| Hazard Type | Detection Accuracy | Visual Alert |
|-------------|-------------------|--------------|
| 🔥 **Fire** | High | Flashing red border |
| 💧 **Water Leakage** | High | Blue border |
| 🛢️ **Oil Leakage** | High | Orange border |
| 💨 **Steam Leakage** | High | Gray border |
| ✓ **Normal (No Hazard)** | High | Green border |

#### 🔊 Sound Detection (via USB Camera Microphone)

| Sound Type | Detection Method | Alert |
|------------|-----------------|-------|
| 🔊 **Loud Bang / Impact** | Energy spike (8x baseline) | Critical alert |
| ⚙️ **Grinding / Mechanical** | Mid-frequency energy (500-8kHz) | Warning alert |
| 💨 **Hissing / Gas Leak** | High-frequency energy (2-16kHz) | Warning alert |
| 🚨 **Alarm / Siren** | Tonal detection (800-4kHz) | Critical alert |
| ⚠️ **Abnormal Sound** | General energy spike (2x baseline) | Warning alert |

### 3. Software Components

| Script | Purpose | Status |
|--------|---------|--------|
| `extract_frames.py` | Extracts training frames from video files | ✅ Complete |
| `train_model.py` | Trains the classification model | ✅ Complete |
| `export_model.py` | Exports PyTorch model to ONNX format | ✅ Complete |
| `detect_hazard.py` | Real-time hazard detection with visual output | ✅ Complete |
| `detect_sound.py` | Abnormal sound detection via USB mic | ✅ Complete |
| `detect_combined.py` | Combined video + audio detection | ✅ Complete |

### 4. Raspberry Pi 5 Deployment Package

| Component | Status |
|-----------|--------|
| Automated setup script (`setup_pi.sh`) | ✅ Complete |
| Combined launcher (`run_combined.sh`) | ✅ Complete |
| Video-only launcher (`run_detection.sh`) | ✅ Complete |
| Audio-only launcher (`run_sound.sh`) | ✅ Complete |
| Desktop shortcut & auto-start on boot | ✅ Complete |
| Detailed setup documentation | ✅ Complete |
| Headless mode support (SSH) | ✅ Complete |
| Fullscreen display mode | ✅ Complete |
| USB audio retry logic (device contention fix) | ✅ Complete |

---

## 🎯 Key Features Implemented

### Real-Time Detection
- **Video Input:** USB camera or video file
- **Audio Input:** USB camera built-in microphone (auto-detected sample rate & channels)
- **Processing:** Frame-by-frame classification + continuous audio spectral analysis
- **Output:** Unified live display with color-coded hazard alerts + audio level meter
- **Expected FPS:** 10-15 FPS on Raspberry Pi 5

### Visual Feedback System
- Color-coded borders based on detection class
- Status bar showing current detection and confidence level
- Audio level meter with energy ratio and dominant frequency
- Sound category indicators (bang, grinding, hissing, alarm)
- FPS counter for performance monitoring
- Fullscreen mode for demo presentations
- Keyboard controls (`Q` to quit, `F` for fullscreen)

### Edge Deployment Ready
- Optimized ONNX model for efficient inference
- Complete installation and setup scripts
- Works with standard USB cameras
- Low resource footprint

---

## 📁 Project Deliverables

```
MULTI_HAZARD_DETECTION/
├── models/
│   ├── hazard_detector.pth          # PyTorch checkpoint
│   ├── hazard_detector.onnx         # ONNX model for deployment
│   ├── hazard_detector.onnx.data    # Model weights data
│   └── class_labels.json            # Class mapping
├── scripts/
│   ├── extract_frames.py            # Data preparation
│   ├── train_model.py               # Model training
│   ├── export_model.py              # ONNX export
│   ├── detect_hazard.py             # Visual hazard inference
│   ├── detect_sound.py              # Audio anomaly detection
│   └── detect_combined.py           # Combined video + audio
├── raspberry_pi/
│   ├── setup_pi.sh                  # Automated Pi setup
│   └── RASPBERRY_PI_SETUP.md        # Deployment guide
├── dataset/                          # Training dataset
├── README.md                         # Project documentation
└── MULTI_HAZARD_IMPLEMENTATION_PLAN.md
```

---

## 🔧 Technical Specifications

| Specification | Value |
|--------------|-------|
| **Model Architecture** | MobileNetV2 |
| **Input Resolution** | 224×224 pixels |
| **Number of Classes** | 5 |
| **Training Framework** | PyTorch |
| **Deployment Format** | ONNX Runtime |
| **Target Hardware** | Raspberry Pi 5 (4GB+) |
| **Camera Support** | USB Camera (any V4L2 compatible) |
| **Audio Input** | USB Camera built-in microphone |
| **Audio Detection** | Energy + spectral analysis (baseline-relative) |
| **Audio Dependencies** | PyAudio, SciPy |

---

## 🖥️ Deployment Instructions

### For Raspberry Pi 5

1. **Transfer setup script** to Pi via SCP
2. **Fix line endings** (`sed -i 's/\r$//' setup_pi.sh`)
3. **Run setup script** (`./setup_pi.sh`) — installs all dependencies (~5-10 min)
4. **Copy model files** (`hazard_detector.onnx` and `.onnx.data`)
5. **Find audio device** (`./run_sound.sh --list-devices`)
6. **Start detection** (`./run_combined.sh --camera 0 --audio-device <MIC_INDEX> --skip 3`)
7. **Optional:** Create desktop shortcut or auto-start on boot (see setup guide)

Complete step-by-step instructions are provided in `raspberry_pi/RASPBERRY_PI_SETUP.md`.

---

## 🔮 Future Enhancement Opportunities

| Enhancement | Description |
|-------------|-------------|
| ~~Audio Detection~~ | ~~Add abnormal sound detection module~~ ✅ **DONE** |
| ML Audio Model | Train custom ML model for specific industrial sounds |
| Alert Logging | Save detection events to database |
| MQTT Integration | Real-time IoT alerts |
| Web Dashboard | Remote monitoring interface |
| Multi-Camera | Support for multiple camera feeds |

---

## 📞 Support & Maintenance

All source code, trained models, and documentation are included in this deliverable. The system is designed for:

- **Easy retraining** with new data using provided scripts
- **Simple deployment** via automated setup scripts
- **Extensibility** for adding new hazard classes

---

**Document Version:** 1.1  
**Delivery Date:** February 11, 2026
