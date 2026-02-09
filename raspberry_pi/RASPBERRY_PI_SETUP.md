# 🍓 Raspberry Pi 5 - Multi-Hazard Detection Setup Guide

Complete guide for setting up the Multi-Hazard Detection System on a **fresh Raspberry Pi 5**.

---

## 📋 Prerequisites

- **Raspberry Pi 5** with Raspberry Pi OS (64-bit recommended)
- **USB Camera** connected to Pi
- **Monitor + Keyboard/Mouse** OR SSH access
- **Network connection** (for package installation)
- **Windows PC** with trained model files

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Transfer Setup Script to Pi

From your **Windows PowerShell**:

```powershell
# Replace <PI_IP> with your Pi's IP address (e.g., 192.168.1.100)
# Replace <PI_USER> with your Pi username (e.g., pi or test)

scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\raspberry_pi\setup_pi.sh" <PI_USER>@<PI_IP>:~/
```

### Step 2: Run Setup Script on Pi

SSH into your Raspberry Pi:

```bash
ssh <PI_USER>@<PI_IP>
```

Then run:

```bash
chmod +x ~/setup_pi.sh
~/setup_pi.sh
```

> ⏱️ **This takes 5-10 minutes** to install dependencies and set up the environment.

### Step 3: Copy Model Files

From **Windows PowerShell**:

```powershell
scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\models\hazard_detector.onnx" <PI_USER>@<PI_IP>:~/multi_hazard_detection/models/
scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\models\hazard_detector.onnx.data" <PI_USER>@<PI_IP>:~/multi_hazard_detection/models/
```

### Step 4: Run Detection

On the Raspberry Pi (with monitor connected):

```bash
cd ~/multi_hazard_detection
./run_detection.sh --camera 0
```

**Fullscreen mode:**
```bash
./run_detection.sh --camera 0 --fullscreen
```

---

## 🖥️ Display Modes

### With Monitor Connected
```bash
./run_detection.sh --camera 0
```
- Shows real-time video with detection overlays
- Color-coded status bar
- Press `q` to quit, `f` for fullscreen

### Headless Mode (SSH Only)
```bash
./run_detection.sh --camera 0 --headless
```
- Console output only
- Shows detection status every second
- Press `Ctrl+C` to quit

---

## ⚙️ Command Options

| Option | Description | Example |
|--------|-------------|---------|
| `--camera N` | USB camera index | `--camera 0` |
| `--video FILE` | Video file path | `--video test.mp4` |
| `--skip N` | Process every Nth frame | `--skip 2` (default) |
| `--fullscreen` | Start in fullscreen | `--fullscreen` |
| `--headless` | Console output only | `--headless` |

---

## 🔍 Troubleshooting

### Camera Not Found
```bash
# Check connected cameras
ls /dev/video*
v4l2-ctl --list-devices
```

### Model Not Found Error
Ensure you copied both files:
- `hazard_detector.onnx`
- `hazard_detector.onnx.data`

### Display Error (GTK/OpenCV)
If you get a GTK error when running with display:
```bash
sudo apt install -y libgtk-3-0 libgtk-3-dev
cd ~/multi_hazard_detection
source hazard_env/bin/activate
pip uninstall opencv-python-headless -y
pip install opencv-python
```

---

## 🎯 Detection Classes

| Class | Alert Level | Visual Indicator |
|-------|-------------|------------------|
| 🔥 Fire | **Critical** | Flashing red border |
| ✓ Normal | OK | Green border |
| 🛢️ Oil Leak | Warning | Orange border |
| 💨 Steam Leak | Warning | Gray border |
| 💧 Water Leak | Warning | Blue border |

---

## 📁 Project Structure on Pi

```
~/multi_hazard_detection/
├── hazard_env/              # Python virtual environment
├── models/
│   ├── hazard_detector.onnx      # Model file
│   ├── hazard_detector.onnx.data # Model weights
│   └── class_labels.json         # Class mappings
├── scripts/
│   └── detect_hazard.py     # Detection script
├── run_detection.sh         # Quick-run script
└── README.md                # Quick reference
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `f` | Toggle fullscreen |

---

## 🔄 Re-running Setup

If you need to start fresh:

```bash
rm -rf ~/multi_hazard_detection
~/setup_pi.sh
```

---

**Created:** February 2026  
**Project:** GMR 2025 Expo - Multi-Hazard Detection Demo
