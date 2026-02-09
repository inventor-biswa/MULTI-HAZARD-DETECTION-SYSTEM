# Multi-Hazard Video Analytics System
## Client Deliverable Document

**Project:** GMR 2025 Expo - Industrial Hazard Detection Demo  
**Date:** February 10, 2026  
**Prepared by:** Thynx Development Team

---

## 📋 Executive Summary

We have developed a **real-time video analytics system** for industrial hazard detection, designed for deployment on edge devices (Raspberry Pi 5). The system can detect 5 different hazard classes from a live camera feed and provides immediate visual alerts.

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

### 3. Software Components

| Script | Purpose | Status |
|--------|---------|--------|
| `extract_frames.py` | Extracts training frames from video files | ✅ Complete |
| `train_model.py` | Trains the classification model | ✅ Complete |
| `export_model.py` | Exports PyTorch model to ONNX format | ✅ Complete |
| `detect_hazard.py` | Real-time hazard detection with visual output | ✅ Complete |

### 4. Raspberry Pi 5 Deployment Package

| Component | Status |
|-----------|--------|
| Automated setup script (`setup_pi.sh`) | ✅ Complete |
| Quick-run script (`run_detection.sh`) | ✅ Complete |
| Detailed setup documentation | ✅ Complete |
| Headless mode support (SSH) | ✅ Complete |
| Fullscreen display mode | ✅ Complete |

---

## 🎯 Key Features Implemented

### Real-Time Detection
- **Input:** USB camera or video file
- **Processing:** Frame-by-frame classification
- **Output:** Live display with color-coded hazard alerts
- **Expected FPS:** 10-15 FPS on Raspberry Pi 5

### Visual Feedback System
- Color-coded borders based on detection class
- Status bar showing current detection and confidence level
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
│   └── detect_hazard.py             # Real-time inference
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

---

## 🖥️ Deployment Instructions

### For Raspberry Pi 5

1. **Transfer setup script** to Pi via SCP
2. **Run setup script** (`./setup_pi.sh`)
3. **Copy model files** (`hazard_detector.onnx` and `.onnx.data`)
4. **Start detection** (`./run_detection.sh --camera 0`)

Complete step-by-step instructions are provided in `raspberry_pi/RASPBERRY_PI_SETUP.md`.

---

## 🔮 Future Enhancement Opportunities

| Enhancement | Description |
|-------------|-------------|
| Audio Detection | Add abnormal sound detection module |
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

**Document Version:** 1.0  
**Delivery Date:** February 10, 2026
