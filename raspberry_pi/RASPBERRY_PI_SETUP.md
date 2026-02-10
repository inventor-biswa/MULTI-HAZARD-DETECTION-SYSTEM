# 🍓 Raspberry Pi - Multi-Hazard Detection Setup Guide

Complete guide for setting up the **Multi-Hazard Detection System** (Video + Audio) on a Raspberry Pi.

---

## 📋 Prerequisites

- **Raspberry Pi 4/5** with Raspberry Pi OS (64-bit recommended)
- **USB Camera** (e.g., Logitech C270 HD) — used for both video AND microphone
- **Monitor + Keyboard/Mouse** OR SSH access
- **Network connection** (for package installation)
- **Windows PC** with trained model files

---

## 🚀 Setup (Step-by-Step)

### Step 1: Transfer Files to Pi

From your **Windows PowerShell** (replace `<PI_IP>` with your Pi's IP):

```powershell
# Transfer setup script
scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\raspberry_pi\setup_pi.sh" test@<PI_IP>:~/Downloads/

# Transfer model files
scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\models\hazard_detector.onnx" test@<PI_IP>:~/Downloads/
scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\models\hazard_detector.onnx.data" test@<PI_IP>:~/Downloads/
```

### Step 2: SSH into Pi

```bash
ssh test@<PI_IP>
```

### Step 3: Run Setup Script

```bash
cd ~/Downloads

# Fix Windows line endings (IMPORTANT — without this you get "required file not found")
sed -i 's/\r$//' setup_pi.sh

# Make executable and run
chmod +x setup_pi.sh
./setup_pi.sh
```

> ⏱️ **This takes 5-10 minutes** to install dependencies and set up the environment.

### Step 4: Copy Model Files to Project

```bash
cp ~/Downloads/hazard_detector.onnx ~/multi_hazard_detection/models/
cp ~/Downloads/hazard_detector.onnx.data ~/multi_hazard_detection/models/
```

### Step 5: Find Your Audio Device Index

```bash
cd ~/multi_hazard_detection
./run_sound.sh --list-devices
```

Example output:
```
Available Audio Input Devices:
  [2] C270 HD WEBCAM: USB Audio (hw:3,0) (ch:1)
```

> 📝 Note the **number in brackets** — this is your `--audio-device` index (e.g., `2`)

### Step 6: Run Combined Detection 🚀

```bash
cd ~/multi_hazard_detection
./run_combined.sh --camera 0 --audio-device 2
```

> Replace `2` with the device index from Step 5.

**Fullscreen mode (for demo/expo):**
```bash
./run_combined.sh --camera 0 --audio-device 2 --fullscreen
```

---

## � All Run Commands

| What | Command |
|------|---------|
| **Combined (Video + Audio)** | `./run_combined.sh --camera 0 --audio-device 2` |
| **Combined Fullscreen** | `./run_combined.sh --camera 0 --audio-device 2 --fullscreen` |
| **Combined + Skip Frames** | `./run_combined.sh --camera 0 --audio-device 2 --skip 3` |
| **Video Only** | `./run_detection.sh --camera 0` |
| **Video Fullscreen** | `./run_detection.sh --camera 0 --fullscreen` |
| **Video Headless (SSH)** | `./run_detection.sh --camera 0 --headless` |
| **Audio Only** | `./run_sound.sh --device 2` |
| **List Audio Devices** | `./run_sound.sh --list-devices` |
| **Video from File** | `./run_detection.sh --video path/to/video.mp4` |

---

## ⚙️ Command Options

### Combined Detection (`run_combined.sh`)

| Option | Description | Default |
|--------|-------------|---------|
| `--camera N` | Camera index | `0` |
| `--video FILE` | Use video file instead of camera | — |
| `--audio-device N` | Microphone device index (use `--list-devices` to find) | auto |
| `--audio-threshold N` | Audio sensitivity (lower = more sensitive) | `2.0` |
| `--skip N` | Process every Nth frame (higher = faster but less smooth) | `2` |
| `--fullscreen` | Start in fullscreen mode | off |
| `--delay N` | Display delay in ms | `1` |

### Video Only (`run_detection.sh`)

| Option | Description | Default |
|--------|-------------|---------|
| `--camera N` | Camera index | `0` |
| `--video FILE` | Use video file | — |
| `--skip N` | Process every Nth frame | `2` |
| `--fullscreen` | Fullscreen mode | off |
| `--headless` | No display, console output only (for SSH) | off |

### Audio Only (`run_sound.sh`)

| Option | Description | Default |
|--------|-------------|---------|
| `--list-devices` | List all audio input devices | — |
| `--device N` | Audio device index | auto |
| `--threshold N` | Sensitivity multiplier | `2.0` |
| `--no-calibrate` | Skip baseline calibration | off |

---

## ⌨️ Keyboard Shortcuts (When Display is Active)

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `f` | Toggle fullscreen |
| `Ctrl+C` | Force quit |

---

## 🎯 Detection Classes

### Video Detection

| Class | Alert Level | Visual Indicator |
|-------|-------------|------------------|
| 🔥 Fire | **Critical** | Flashing red border |
| ✓ Normal | OK | Green border |
| 🛢️ Oil Leak | Warning | Orange border |
| 💨 Steam Leak | Warning | Gray border |
| 💧 Water Leak | Warning | Blue border |

### Audio Detection

| Sound Type | Alert Level |
|------------|-------------|
| 💥 Loud Bang / Impact | **Critical** |
| 🔔 Alarm / Siren | **Critical** |
| ⚙️ Grinding / Mechanical | Warning |
| 💨 Hissing / Gas Leak | Warning |
| 🔊 Abnormal Sound | Warning |

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
│   ├── detect_hazard.py     # Video detection script
│   ├── detect_sound.py      # Audio detection script
│   └── detect_combined.py   # Combined video + audio script
├── run_detection.sh         # Video only launcher
├── run_sound.sh             # Audio only launcher
├── run_combined.sh          # Combined launcher
└── README.md
```

---

## 🔍 Troubleshooting

### "required file not found" when running setup_pi.sh

The script has Windows line endings. Fix with:
```bash
sed -i 's/\r$//' setup_pi.sh
```

### "Package libatlas-base-dev has no installation candidate"

This package was removed in newer Raspberry Pi OS. Update `setup_pi.sh` to remove `libatlas-base-dev` from the apt install list — `liblapack-dev` and `libblas-dev` (already included) provide the same functionality.

### "Invalid sample rate" / Audio won't open

Your USB camera mic may not support 44100Hz. The updated `setup_pi.sh` auto-detects the device's native sample rate (e.g., C270 uses **48000Hz**). Re-run setup to get the fix.

To check your device's native rate:
```bash
cd ~/multi_hazard_detection
source hazard_env/bin/activate
python -c "
import pyaudio
p = pyaudio.PyAudio()
info = p.get_device_info_by_index(2)
print(f'Device: {info[\"name\"]}')
print(f'Channels: {info[\"maxInputChannels\"]}')
print(f'Native Sample Rate: {int(info[\"defaultSampleRate\"])}Hz')
p.terminate()
"
```

### Camera Not Found

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

### ALSA Warnings (Long list of "Unknown PCM" messages)

**These are harmless!** PyAudio scans all audio backends during initialization. The warnings don't affect functionality. Ignore them.

### Qt Font Warnings

If you see `QFontDatabase: Cannot find font directory` warnings — these are cosmetic and won't stop the detection from working.

### Model Not Found Error

Ensure you copied both files to `~/multi_hazard_detection/models/`:
- `hazard_detector.onnx`
- `hazard_detector.onnx.data`

### Display Error (GTK/OpenCV)

```bash
sudo apt install -y libgtk-3-0 libgtk-3-dev
cd ~/multi_hazard_detection
source hazard_env/bin/activate
pip uninstall opencv-python-headless -y
pip install opencv-python
```

### Audio Notes

- **Calibration takes ~5 seconds** — keep environment quiet during "Calibrating audio..." phase
- If audio fails, video-only mode continues automatically
- Audio threshold: lower value = more sensitive (default: 2.0)

---

## 🔄 Re-running Setup

If you need to start fresh:

```bash
rm -rf ~/multi_hazard_detection
cd ~/Downloads
./setup_pi.sh
# Then copy model files again (Step 4)
```

To update scripts only (without full reinstall):

```bash
# Re-transfer setup_pi.sh from Windows, then:
cd ~/Downloads
sed -i 's/\r$//' setup_pi.sh
./setup_pi.sh
cp ~/Downloads/hazard_detector.onnx ~/multi_hazard_detection/models/
cp ~/Downloads/hazard_detector.onnx.data ~/multi_hazard_detection/models/
```

---

**Created:** February 2026
**Project:** GMR 2025 Expo - Multi-Hazard Detection Demo



**for Desk top application**
cat > ~/Desktop/HazardDetection.desktop << 'EOF'
[Desktop Entry]
Name=Multi-Hazard Detection
Comment=Launch Video + Audio Hazard Detection
Exec=bash -c "cd ~/multi_hazard_detection && ./run_combined.sh --camera 0 --audio-device 2 --skip 3 --fullscreen"
Icon=camera-video
Terminal=true
Type=Application
Categories=Utility;
EOF
chmod +x ~/Desktop/HazardDetection.desktop


**Auto-Start on Boot (No Click Needed)**

mkdir -p ~/.config/autostart
cat > ~/.config/autostart/hazard_detection.desktop << 'EOF'
[Desktop Entry]
Name=Multi-Hazard Detection
Exec=bash -c "sleep 10 && cd ~/multi_hazard_detection && ./run_combined.sh --camera 0 --audio-device 2 --skip 3 --fullscreen"
Terminal=true
Type=Application
X-GNOME-Autostart-enabled=true
EOF