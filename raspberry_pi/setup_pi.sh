#!/bin/bash
# ============================================================
# Multi-Hazard Detection System - Raspberry Pi 5 Setup Script
# ============================================================
# Run this script after SSHing into your Raspberry Pi 5
# Usage: chmod +x setup_pi.sh && ./setup_pi.sh
# ============================================================

set -e  # Exit on any error

echo "============================================================"
echo "🍓 Multi-Hazard Detection - Raspberry Pi 5 Setup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$HOME/multi_hazard_detection"

# Step 1: System Update
echo -e "${YELLOW}[1/10] Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Step 2: Install system dependencies (including GTK for OpenCV GUI)
echo -e "${YELLOW}[2/10] Installing system dependencies...${NC}"
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    libopencv-dev \
    libhdf5-dev \
    libharfbuzz-dev \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    gfortran \
    cmake \
    v4l-utils \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgtk-3-dev \
    portaudio19-dev \
    libasound2-dev

# Step 3: Create project directory
echo -e "${YELLOW}[3/10] Creating project directory...${NC}"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/scripts"
cd "$PROJECT_DIR"

# Step 4: Create and activate virtual environment
echo -e "${YELLOW}[4/10] Setting up Python virtual environment...${NC}"
python3 -m venv hazard_env
source hazard_env/bin/activate

# Step 5: Install Python packages (full OpenCV with GUI support)
echo -e "${YELLOW}[5/10] Installing Python packages (this may take a while)...${NC}"
pip install --upgrade pip
pip install \
    numpy \
    opencv-python \
    onnxruntime \
    pyaudio \
    scipy

# Step 6: Create the detection script
echo -e "${YELLOW}[6/10] Creating detection script...${NC}"
cat > "$PROJECT_DIR/scripts/detect_hazard.py" << 'DETECT_SCRIPT'
"""
Multi-Hazard Detection - Real-Time Inference Script for Raspberry Pi 5

Runs real-time hazard detection from USB camera or video file.
Displays detection status with color-coded alerts for each class.

Usage:
    python detect_hazard.py --camera 0           # USB camera
    python detect_hazard.py --video path/to.mp4  # Video file
    python detect_hazard.py --camera 0 --skip 2  # Skip frames for better FPS
"""

import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from collections import deque

try:
    import onnxruntime as ort
except ImportError:
    print("❌ Please install onnxruntime: pip install onnxruntime")
    exit(1)


# Detection classes configuration
CLASS_CONFIG = {
    "fire": {
        "display": "FIRE",
        "emoji": "🔥",
        "color_bgr": (0, 0, 255),      # Red
        "alert_level": "critical"
    },
    "normal": {
        "display": "Normal",
        "emoji": "✓",
        "color_bgr": (0, 200, 0),      # Green
        "alert_level": "ok"
    },
    "oil_leak": {
        "display": "Oil Leak",
        "emoji": "🛢️",
        "color_bgr": (0, 100, 200),    # Orange/Brown
        "alert_level": "warning"
    },
    "steam_leak": {
        "display": "Steam Leak",
        "emoji": "💨",
        "color_bgr": (200, 200, 200),  # Gray
        "alert_level": "warning"
    },
    "water_leak": {
        "display": "Water Leak",
        "emoji": "💧",
        "color_bgr": (255, 150, 0),    # Blue
        "alert_level": "warning"
    }
}


class HazardDetector:
    """Real-time hazard detection using ONNX model."""
    
    def __init__(self, model_path: str, labels_path: str = None):
        """Initialize the detector with ONNX model."""
        print(f"🔄 Loading model: {model_path}")
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Load class labels
        if labels_path and Path(labels_path).exists():
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
            self.class_names = labels_data.get('classes', list(CLASS_CONFIG.keys()))
            self.idx_to_class = {int(k): v for k, v in labels_data.get('idx_to_class', {}).items()}
        else:
            self.class_names = list(CLASS_CONFIG.keys())
            self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}
        
        print(f"✅ Model loaded! Classes: {self.class_names}")
        
        # Normalization parameters (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Confidence smoothing
        self.confidence_history = deque(maxlen=5)
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input."""
        # Resize to 224x224
        img = cv2.resize(frame, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        img = (img - self.mean) / self.std
        
        # Transpose to NCHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, frame: np.ndarray) -> tuple:
        """
        Run inference on a frame.
        
        Returns:
            (class_name, confidence, all_probabilities)
        """
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0][0]
        
        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = self.idx_to_class.get(pred_idx, f"class_{pred_idx}")
        confidence = probs[pred_idx]
        
        # Smooth confidence (optional)
        self.confidence_history.append(probs)
        if len(self.confidence_history) > 1:
            avg_probs = np.mean(self.confidence_history, axis=0)
            pred_idx = np.argmax(avg_probs)
            pred_class = self.idx_to_class.get(pred_idx, f"class_{pred_idx}")
            confidence = avg_probs[pred_idx]
        
        return pred_class, confidence, probs


class DisplayRenderer:
    """Render detection results on frame."""
    
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.flash_state = False
        self.last_flash_time = time.time()
        
    def render(self, frame: np.ndarray, class_name: str, confidence: float, 
               all_probs: np.ndarray, fps: float, idx_to_class: dict) -> np.ndarray:
        """Render detection overlay on frame."""
        h, w = frame.shape[:2]
        
        # Get class config
        config = CLASS_CONFIG.get(class_name, CLASS_CONFIG["normal"])
        color = config["color_bgr"]
        alert_level = config["alert_level"]
        display_name = config["display"]
        
        # Flash effect for critical alerts
        if alert_level == "critical":
            if time.time() - self.last_flash_time > 0.3:
                self.flash_state = not self.flash_state
                self.last_flash_time = time.time()
            
            if self.flash_state:
                # Add red tint to frame
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        
        # Draw border based on alert level
        border_thickness = 8 if alert_level == "critical" else 4
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, border_thickness)
        
        # Status bar background (two sections: status text + class indicators)
        status_bar_h = 110
        cv2.rectangle(frame, (0, h - status_bar_h), (w, h), (30, 30, 30), -1)
        
        # Status text (top section of status bar) - left side
        status_text = f"STATUS: {display_name.upper()}"
        if alert_level != "ok":
            status_text += "!"
        
        cv2.putText(frame, status_text, (20, h - status_bar_h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence (top right) - positioned with enough space
        conf_text = f"{confidence*100:.1f}%"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, conf_text, (w - text_size[0] - 20, h - status_bar_h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Separator line
        cv2.line(frame, (10, h - status_bar_h + 50), (w - 10, h - status_bar_h + 50), (60, 60, 60), 1)
        
        # Class indicators bar (bottom section)
        indicator_y = h - 20
        indicator_h = 35
        indicator_w = (w - 40) // len(self.class_names)
        
        for i, cls in enumerate(self.class_names):
            x = 20 + i * indicator_w
            cls_config = CLASS_CONFIG.get(cls, CLASS_CONFIG["normal"])
            cls_color = cls_config["color_bgr"]
            
            # Highlight active class
            if cls == class_name:
                cv2.rectangle(frame, (x, indicator_y - indicator_h), (x + indicator_w - 8, indicator_y), 
                             cls_color, -1)
                text_color = (0, 0, 0) if sum(cls_color) > 400 else (255, 255, 255)
            else:
                cv2.rectangle(frame, (x, indicator_y - indicator_h), (x + indicator_w - 8, indicator_y), 
                             (60, 60, 60), -1)
                cv2.rectangle(frame, (x, indicator_y - indicator_h), (x + indicator_w - 8, indicator_y), 
                             (80, 80, 80), 1)  # Border
                text_color = (150, 150, 150)
            
            # Class name (centered in button)
            text = cls_config["display"]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (indicator_w - 8 - text_size[0]) // 2
            text_y = indicator_y - (indicator_h - text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "Multi-Hazard Detection", (w//2 - 130, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame


def main():
    parser = argparse.ArgumentParser(description="Multi-Hazard Detection")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g., 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--labels", type=str, default=None, help="Path to class labels JSON")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame (default: 2)")
    parser.add_argument("--delay", type=int, default=1, help="Display delay (ms)")
    parser.add_argument("--fullscreen", action="store_true", help="Fullscreen mode")
    parser.add_argument("--headless", action="store_true", help="Run without display (print to console)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 Multi-Hazard Detection System - Raspberry Pi 5")
    print("="*60 + "\n")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    model_path = args.model or str(models_dir / "hazard_detector.onnx")
    labels_path = args.labels or str(models_dir / "class_labels.json")
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please copy the model files to the models/ directory")
        print("   Required files:")
        print("     - hazard_detector.onnx")
        print("     - hazard_detector.onnx.data")
        print("     - class_labels.json")
        return
    
    # Initialize detector
    detector = HazardDetector(model_path, labels_path)
    renderer = DisplayRenderer(detector.class_names)
    
    # Open video source
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        # Set camera properties for better performance on Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        source_name = f"Camera {args.camera}"
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = Path(args.video).name
    else:
        # Default to camera 0
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        source_name = "Camera 0 (default)"
    
    if not cap.isOpened():
        print(f"❌ Could not open video source: {source_name}")
        print("   Check if camera is connected: ls /dev/video*")
        return
    
    print(f"📹 Source: {source_name}")
    print(f"⚙️ Frame skip: {args.skip}")
    if not args.headless:
        print(f"⌨️ Press 'q' to quit, 'f' for fullscreen toggle")
    else:
        print(f"⌨️ Press Ctrl+C to quit")
    print()
    
    # Create window (if not headless)
    window_name = "Multi-Hazard Detection"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    frame_count = 0
    fps_start = time.time()
    fps = 0.0
    last_prediction = ("normal", 0.0, np.zeros(len(detector.class_names)))
    last_print_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                break
            
            frame_count += 1
            
            # Run inference on selected frames
            if frame_count % args.skip == 0:
                class_name, confidence, all_probs = detector.predict(frame)
                last_prediction = (class_name, confidence, all_probs)
            else:
                class_name, confidence, all_probs = last_prediction
            
            # Calculate FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Headless mode: print to console
            if args.headless:
                if time.time() - last_print_time > 1.0:  # Print every second
                    config = CLASS_CONFIG.get(class_name, CLASS_CONFIG["normal"])
                    alert = "⚠️ " if config["alert_level"] != "ok" else "✓ "
                    print(f"{alert}{class_name.upper()}: {confidence*100:.1f}% | FPS: {fps:.1f}")
                    last_print_time = time.time()
            else:
                # Render overlay
                display_frame = renderer.render(
                    frame.copy(), 
                    class_name, 
                    confidence, 
                    all_probs, 
                    fps,
                    detector.idx_to_class
                )
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Handle key presses
                key = cv2.waitKey(args.delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    # Toggle fullscreen
                    current = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        window_name, 
                        cv2.WND_PROP_FULLSCREEN, 
                        cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                    )
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
    
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("✅ Detection stopped")


if __name__ == "__main__":
    main()
DETECT_SCRIPT

# Step 7: Create sound detection script
echo -e "${YELLOW}[7/10] Creating sound detection script...${NC}"
cp /dev/stdin "$PROJECT_DIR/scripts/detect_sound.py" << 'SOUND_SCRIPT'
"""
Abnormal Sound Detection Module for Raspberry Pi
Detects abnormal sounds using USB camera microphone.
Usage:
    python detect_sound.py --list-devices
    python detect_sound.py --device 1
"""
import numpy as np
import json
import time
import argparse
import threading
import sys
import os
from pathlib import Path
from collections import deque
from datetime import datetime

try:
    import pyaudio
except ImportError:
    print("Install pyaudio: pip install pyaudio")
    sys.exit(1)

try:
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SAMPLE_RATE = 44100
CHUNK_SIZE = 4096
CHANNELS = 1
FORMAT = pyaudio.paInt16
ENERGY_WINDOW_SEC = 0.5
BASELINE_DURATION_SEC = 5
ALERT_COOLDOWN_SEC = 2.0

SOUND_CATEGORIES = {
    "loud_bang": {"display": "LOUD BANG / IMPACT", "energy_multiplier": 8.0, "min_freq": 50, "max_freq": 2000, "alert_level": "critical"},
    "grinding": {"display": "GRINDING / MECHANICAL", "energy_multiplier": 3.0, "min_freq": 500, "max_freq": 8000, "alert_level": "warning"},
    "hissing": {"display": "HISSING / GAS LEAK", "energy_multiplier": 2.5, "min_freq": 2000, "max_freq": 16000, "alert_level": "warning"},
    "alarm": {"display": "ALARM / SIREN", "energy_multiplier": 4.0, "min_freq": 800, "max_freq": 4000, "alert_level": "critical"},
    "abnormal": {"display": "ABNORMAL SOUND", "energy_multiplier": 2.0, "min_freq": 50, "max_freq": 16000, "alert_level": "warning"},
}

class SoundAnalyzer:
    def __init__(self, sample_rate=SAMPLE_RATE, threshold_multiplier=2.0):
        self.sample_rate = sample_rate
        self.threshold_multiplier = threshold_multiplier
        self.baseline_energy = None
        self.baseline_spectrum = None
        self.is_calibrated = False
        buffer_size = int(ENERGY_WINDOW_SEC * sample_rate / CHUNK_SIZE) + 1
        self.energy_history = deque(maxlen=buffer_size)
        self.spectrum_history = deque(maxlen=buffer_size)
        self.last_alert_time = 0
        self.alert_history = deque(maxlen=100)

    def audio_to_float(self, audio_data):
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        return samples / 32768.0

    def compute_energy(self, samples):
        return float(np.sqrt(np.mean(samples ** 2)))

    def compute_spectrum(self, samples):
        if HAS_SCIPY:
            freqs = rfftfreq(len(samples), 1.0 / self.sample_rate)
            spectrum = np.abs(rfft(samples))
        else:
            spectrum = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1.0 / self.sample_rate)
        return freqs, spectrum

    def get_band_energy(self, freqs, spectrum, low_freq, high_freq):
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        if not np.any(mask): return 0.0
        return float(np.mean(spectrum[mask] ** 2))

    def calibrate_baseline(self, audio_chunks):
        all_energies = []
        all_spectra = []
        for chunk in audio_chunks:
            samples = self.audio_to_float(chunk)
            all_energies.append(self.compute_energy(samples))
            freqs, spectrum = self.compute_spectrum(samples)
            all_spectra.append(spectrum)
        self.baseline_energy = max(np.mean(all_energies), 0.001)
        self.baseline_spectrum = np.mean(all_spectra, axis=0)
        self.baseline_freqs = freqs
        self.is_calibrated = True
        return self.baseline_energy

    def analyze(self, audio_data):
        samples = self.audio_to_float(audio_data)
        energy = self.compute_energy(samples)
        freqs, spectrum = self.compute_spectrum(samples)
        self.energy_history.append(energy)
        result = {"is_abnormal": False, "category": "normal", "display_text": "Normal",
                  "energy": energy, "energy_ratio": 1.0, "dominant_freq": 0.0,
                  "alert_level": "ok", "confidence": 0.0, "timestamp": datetime.now().isoformat()}
        if not self.is_calibrated: return result
        energy_ratio = energy / self.baseline_energy
        result["energy_ratio"] = energy_ratio
        if len(spectrum) > 0:
            peak_idx = np.argmax(spectrum[1:]) + 1
            if peak_idx < len(freqs): result["dominant_freq"] = float(freqs[peak_idx])
        now = time.time()
        if now - self.last_alert_time < ALERT_COOLDOWN_SEC: return result
        detected_category = None
        max_confidence = 0.0
        for cat_name, cat_config in SOUND_CATEGORIES.items():
            if cat_name == "abnormal": continue
            required = cat_config["energy_multiplier"] * self.threshold_multiplier / 2.0
            band_energy = self.get_band_energy(freqs, spectrum, cat_config["min_freq"], cat_config["max_freq"])
            baseline_band = max(self.get_band_energy(self.baseline_freqs, self.baseline_spectrum, cat_config["min_freq"], cat_config["max_freq"]), 0.0001)
            band_ratio = band_energy / baseline_band
            if energy_ratio >= required and band_ratio >= required:
                confidence = min(1.0, (energy_ratio / required) * 0.5 + (band_ratio / required) * 0.5)
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_category = cat_name
        if detected_category is None:
            required = SOUND_CATEGORIES["abnormal"]["energy_multiplier"] * self.threshold_multiplier / 2.0
            if energy_ratio >= required:
                detected_category = "abnormal"
                max_confidence = min(1.0, energy_ratio / required)
        if detected_category:
            cat = SOUND_CATEGORIES[detected_category]
            result.update({"is_abnormal": True, "category": detected_category,
                          "display_text": cat["display"], "alert_level": cat["alert_level"],
                          "confidence": max_confidence})
            self.last_alert_time = now
            self.alert_history.append({"timestamp": result["timestamp"], "category": detected_category,
                                       "energy_ratio": energy_ratio, "confidence": max_confidence})
        return result

class AudioCapture:
    def __init__(self, device_index=None, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, channels=CHANNELS):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def list_devices(self):
        print("\nAvailable Audio Input Devices:")
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} (ch:{info['maxInputChannels']})")

    def open_stream(self):
        # Auto-detect channels and sample rate from device
        actual_channels = self.channels
        rates_to_try = [self.sample_rate, 48000, 44100, 32000, 16000]
        if self.device_index is not None:
            try:
                info = self.pa.get_device_info_by_index(self.device_index)
                max_ch = int(info['maxInputChannels'])
                if max_ch > 0:
                    actual_channels = min(max_ch, 2)
                native_rate = int(info['defaultSampleRate'])
                if native_rate not in rates_to_try:
                    rates_to_try.insert(0, native_rate)
                else:
                    rates_to_try.remove(native_rate)
                    rates_to_try.insert(0, native_rate)
                print(f"Device [{self.device_index}] {info['name']} ch:{max_ch} native:{native_rate}Hz")
            except Exception:
                pass
        for rate in rates_to_try:
            for ch in [actual_channels, 1]:
                try:
                    kwargs = {'format': FORMAT, 'channels': ch, 'rate': rate,
                              'input': True, 'frames_per_buffer': self.chunk_size}
                    if self.device_index is not None: kwargs['input_device_index'] = self.device_index
                    self.stream = self.pa.open(**kwargs)
                    self.actual_channels = ch
                    self.sample_rate = rate
                    print(f"Audio opened: {rate}Hz, {ch}ch")
                    return True
                except Exception as e:
                    last_error = e
                    continue
        print(f"Failed to open audio: {last_error}")
        return False

    def read_chunk(self):
        if self.stream and self.stream.is_active():
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                if hasattr(self, 'actual_channels') and self.actual_channels > 1:
                    samples = np.frombuffer(data, dtype=np.int16).reshape(-1, self.actual_channels)
                    return samples.mean(axis=1).astype(np.int16).tobytes()
                return data
            except: return None
        return None

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

def main():
    parser = argparse.ArgumentParser(description="Sound Detection")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        c = AudioCapture(); c.list_devices(); c.close(); return
    capture = AudioCapture(device_index=args.device)
    if not capture.open_stream(): capture.close(); return
    analyzer = SoundAnalyzer(threshold_multiplier=args.threshold)
    if not args.no_calibrate:
        print("Calibrating baseline (5 sec)...")
        chunks = []
        for _ in range(int(5 * SAMPLE_RATE / CHUNK_SIZE)):
            c = capture.read_chunk()
            if c: chunks.append(c)
        analyzer.calibrate_baseline(chunks)
        print(f"Baseline: {analyzer.baseline_energy:.6f}")
    else:
        analyzer.baseline_energy = 0.01
        analyzer.baseline_spectrum = np.ones(CHUNK_SIZE // 2 + 1) * 0.001
        analyzer.baseline_freqs = np.linspace(0, SAMPLE_RATE / 2, CHUNK_SIZE // 2 + 1)
        analyzer.is_calibrated = True
    print("Listening...")
    try:
        while True:
            chunk = capture.read_chunk()
            if chunk is None: continue
            result = analyzer.analyze(chunk)
            if result["is_abnormal"]:
                print(f"  {result['energy_ratio']:.1f}x | {result['display_text']} (conf: {result['confidence']:.0%})")
    except KeyboardInterrupt:
        print(f"\nAlerts: {len(analyzer.alert_history)}")
    finally:
        capture.close()

if __name__ == "__main__": main()
SOUND_SCRIPT

# Step 8: Create combined detection script
echo -e "${YELLOW}[8/10] Creating combined video+audio detection script...${NC}"
cat > "$PROJECT_DIR/scripts/detect_combined.py" << 'COMBINED_SCRIPT'
"""
Combined Video + Audio Hazard Detection for Raspberry Pi
Usage: python detect_combined.py --camera 0 --audio-device 1
"""
import cv2
import numpy as np
import json
import time
import argparse
import threading
import sys
from pathlib import Path
from collections import deque

try:
    import onnxruntime as ort
except ImportError:
    print("Install onnxruntime"); sys.exit(1)

try:
    import pyaudio
except ImportError:
    print("Install pyaudio"); sys.exit(1)

from detect_hazard import HazardDetector, CLASS_CONFIG
from detect_sound import SoundAnalyzer, AudioCapture, SAMPLE_RATE, CHUNK_SIZE, SOUND_CATEGORIES

class AudioDetectionThread:
    def __init__(self, device_index=None, threshold=2.0):
        self.device_index = device_index
        self.threshold = threshold
        self.running = False
        self.thread = None
        self.current_result = {"is_abnormal": False, "category": "normal",
                               "alert_level": "ok", "confidence": 0.0,
                               "energy_ratio": 0.0, "dominant_freq": 0.0, "display_text": "Normal"}
        self.is_calibrated = False
        self.error = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=3)

    def _run(self):
        import time as _time
        try:
            # Retry logic: USB device may be busy if camera is opening simultaneously
            capture = None
            for attempt in range(3):
                _time.sleep(2)  # Wait for camera to finish grabbing USB device
                capture = AudioCapture(device_index=self.device_index)
                if capture.open_stream():
                    break
                capture.close()
                capture = None
                print(f"Audio retry {attempt+1}/3...")
            if capture is None or capture.stream is None:
                self.error = "Failed to open audio after 3 retries"
                return
            analyzer = SoundAnalyzer(threshold_multiplier=self.threshold)
            print("Calibrating audio (5 sec)...")
            chunks = []
            cal_needed = int(5 * capture.sample_rate / CHUNK_SIZE)
            for _ in range(cal_needed):
                if not self.running: capture.close(); return
                c = capture.read_chunk()
                if c: chunks.append(c)
            analyzer.calibrate_baseline(chunks)
            self.is_calibrated = True
            print("Audio calibrated!")
            while self.running:
                chunk = capture.read_chunk()
                if chunk is None: continue
                result = analyzer.analyze(chunk)
                self.current_result = {
                    "is_abnormal": result["is_abnormal"], "category": result["category"],
                    "alert_level": result["alert_level"], "confidence": result["confidence"],
                    "energy_ratio": result["energy_ratio"], "dominant_freq": result["dominant_freq"],
                    "display_text": result.get("display_text", "Normal")}
            capture.close()
        except Exception as e:
            self.error = str(e)

class CombinedRenderer:
    def __init__(self, class_names):
        self.class_names = class_names
        self.flash_state = False
        self.last_flash_time = time.time()

    def render(self, frame, video_class, video_confidence, all_probs, fps, idx_to_class, audio_result):
        h, w = frame.shape[:2]
        config = CLASS_CONFIG.get(video_class, CLASS_CONFIG["normal"])
        color = config["color_bgr"]
        alert_level = config["alert_level"]
        display_name = config["display"]
        audio_abnormal = audio_result.get("is_abnormal", False)
        audio_alert = audio_result.get("alert_level", "ok")
        is_critical = (alert_level == "critical") or (audio_alert == "critical")
        if is_critical:
            if time.time() - self.last_flash_time > 0.3:
                self.flash_state = not self.flash_state
                self.last_flash_time = time.time()
            if self.flash_state:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        border_thickness = 8 if is_critical else 4
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, border_thickness)
        bar_h = 140
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f"VIDEO: {display_name.upper()}", (20, h - bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"{video_confidence*100:.1f}%", (w//2 - 60, h - bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        audio_color = (0, 0, 255) if audio_alert == "critical" else (0, 200, 255) if audio_abnormal else (0, 200, 0)
        audio_text = audio_result.get("display_text", "Normal") if audio_abnormal else "Normal"
        cv2.putText(frame, f"AUDIO: {audio_text}", (w//2 + 10, h - bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
        energy = audio_result.get("energy_ratio", 0)
        fill = int(min(energy / 10.0, 1.0) * (w - 40))
        cv2.rectangle(frame, (20, h - bar_h + 40), (w - 20, h - bar_h + 52), (60, 60, 60), -1)
        if fill > 0:
            mc = (0, 200, 0) if energy < 2 else (0, 200, 255) if energy < 5 else (0, 0, 255)
            cv2.rectangle(frame, (20, h - bar_h + 40), (20 + fill, h - bar_h + 52), mc, -1)
        cv2.putText(frame, f"Level: {energy:.1f}x  Freq: {audio_result.get('dominant_freq', 0):.0f}Hz",
                    (20, h - bar_h + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.line(frame, (10, h - bar_h + 78), (w - 10, h - bar_h + 78), (60, 60, 60), 1)
        all_cls = self.class_names + ["sound"]
        iw = (w - 40) // len(all_cls)
        for i, cls in enumerate(all_cls):
            x = 20 + i * iw
            if cls == "sound":
                active = audio_abnormal; cc = audio_color; text = "Sound"
            else:
                active = (cls == video_class)
                cc = CLASS_CONFIG.get(cls, CLASS_CONFIG["normal"])["color_bgr"]
                text = CLASS_CONFIG.get(cls, CLASS_CONFIG["normal"])["display"]
            if active:
                cv2.rectangle(frame, (x, h - 45), (x + iw - 6, h - 15), cc, -1)
                tc = (0,0,0) if sum(cc) > 400 else (255,255,255)
            else:
                cv2.rectangle(frame, (x, h - 45), (x + iw - 6, h - 15), (60, 60, 60), -1)
                tc = (150, 150, 150)
            ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(frame, text, (x + (iw - 6 - ts[0])//2, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, tc, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Multi-Hazard [V+A]", (w//2 - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

def main():
    parser = argparse.ArgumentParser(description="Combined Detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--audio-device", type=int, default=None)
    parser.add_argument("--audio-threshold", type=float, default=2.0)
    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent
    model_path = args.model or str(base_dir / "models" / "hazard_detector.onnx")
    labels_path = args.labels or str(base_dir / "models" / "class_labels.json")
    if not Path(model_path).exists(): print(f"Model not found: {model_path}"); return
    detector = HazardDetector(model_path, labels_path)
    renderer = CombinedRenderer(detector.class_names)
    audio_thread = AudioDetectionThread(device_index=args.audio_device, threshold=args.audio_threshold)
    audio_thread.start()
    timeout = 10
    while not audio_thread.is_calibrated and timeout > 0:
        time.sleep(0.5); timeout -= 0.5
        if audio_thread.error: print(f"Audio error: {audio_thread.error}"); break
    cap = cv2.VideoCapture(args.camera if args.video is None else args.video)
    if not cap.isOpened(): print("Cannot open video"); audio_thread.stop(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    wn = "Multi-Hazard [V+A]"
    cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
    if args.fullscreen: cv2.setWindowProperty(wn, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fc = 0; fps_start = time.time(); fps = 0.0
    last_pred = ("normal", 0.0, np.zeros(len(detector.class_names)))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break
            fc += 1
            if fc % args.skip == 0:
                cn, conf, ap = detector.predict(frame); last_pred = (cn, conf, ap)
            else:
                cn, conf, ap = last_pred
            if fc % 30 == 0: fps = 30 / (time.time() - fps_start); fps_start = time.time()
            df = renderer.render(frame.copy(), cn, conf, ap, fps, detector.idx_to_class, audio_thread.current_result)
            cv2.imshow(wn, df)
            key = cv2.waitKey(args.delay) & 0xFF
            if key == ord('q'): break
            elif key == ord('f'):
                cur = cv2.getWindowProperty(wn, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(wn, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if cur == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
    except KeyboardInterrupt: pass
    finally:
        audio_thread.stop(); cap.release(); cv2.destroyAllWindows()
        print("Detection stopped")

if __name__ == "__main__": main()
COMBINED_SCRIPT

# Step 9: Create class labels file
echo -e "${YELLOW}[9/10] Creating class labels file...${NC}"
cat > "$PROJECT_DIR/models/class_labels.json" << 'LABELS_JSON'
{
    "classes": ["fire", "normal", "oil_leak", "steam_leak", "water_leak"],
    "class_to_idx": {
        "fire": 0,
        "normal": 1,
        "oil_leak": 2,
        "steam_leak": 3,
        "water_leak": 4
    },
    "idx_to_class": {
        "0": "fire",
        "1": "normal",
        "2": "oil_leak",
        "3": "steam_leak",
        "4": "water_leak"
    },
    "display_names": {
        "fire": "🔥 FIRE",
        "normal": "✓ Normal",
        "oil_leak": "🛢️ Oil Leak",
        "steam_leak": "💨 Steam Leak",
        "water_leak": "💧 Water Leak"
    },
    "colors_bgr": {
        "fire": [0, 0, 255],
        "normal": [0, 255, 0],
        "oil_leak": [0, 140, 255],
        "steam_leak": [180, 180, 180],
        "water_leak": [255, 191, 0]
    }
}
LABELS_JSON

# Step 10: Create helper scripts and README
echo -e "${YELLOW}[10/10] Creating helper scripts...${NC}"

# Run script
cat > "$PROJECT_DIR/run_detection.sh" << 'RUN_SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"
source hazard_env/bin/activate
python scripts/detect_hazard.py "$@"
RUN_SCRIPT
chmod +x "$PROJECT_DIR/run_detection.sh"

# Sound detection run script
cat > "$PROJECT_DIR/run_sound.sh" << 'RUN_SOUND'
#!/bin/bash
cd "$(dirname "$0")"
source hazard_env/bin/activate
python scripts/detect_sound.py "$@"
RUN_SOUND
chmod +x "$PROJECT_DIR/run_sound.sh"

# Combined detection run script
cat > "$PROJECT_DIR/run_combined.sh" << 'RUN_COMBINED'
#!/bin/bash
cd "$(dirname "$0")"
source hazard_env/bin/activate
python scripts/detect_combined.py "$@"
RUN_COMBINED
chmod +x "$PROJECT_DIR/run_combined.sh"
chmod +x "$PROJECT_DIR/run_detection.sh"

# README
cat > "$PROJECT_DIR/README.md" << 'README'
# Multi-Hazard Detection - Raspberry Pi 5

## Quick Start

### 1. Copy Model Files
Copy these files from your Windows PC to `~/multi_hazard_detection/models/`:
- `hazard_detector.onnx`
- `hazard_detector.onnx.data`

Use SCP from Windows PowerShell:
```powershell
scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\models\hazard_detector.onnx" pi@<PI_IP>:~/multi_hazard_detection/models/
scp "D:\Thynx\gmr 2025 expo\MULTI_HAZARD_DETECTION\models\hazard_detector.onnx.data" pi@<PI_IP>:~/multi_hazard_detection/models/
```

### 2. Run Detection

With display (monitor connected):
```bash
./run_detection.sh --camera 0
```

Fullscreen mode:
```bash
./run_detection.sh --camera 0 --fullscreen
```

Without display (SSH/headless):
```bash
./run_detection.sh --camera 0 --headless
```

With video file:
```bash
./run_detection.sh --video path/to/video.mp4
```

### Options
- `--camera 0` : Use USB camera (index 0)
- `--video FILE` : Use video file
- `--skip N` : Process every Nth frame (default: 2)
- `--headless` : No display, print to console
- `--fullscreen` : Start in fullscreen
- `-h` : Show help

### Keyboard Shortcuts (with display)
- `q` : Quit
- `f` : Toggle fullscreen

### Check Camera
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

README

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "📁 Project directory: $PROJECT_DIR"
echo ""
echo -e "${YELLOW}⚠️  NEXT STEPS:${NC}"
echo ""
echo "1. Copy model files from your Windows PC:"
echo "   From Windows PowerShell, run:"
echo ""
echo -e "   ${GREEN}scp \"D:\\Thynx\\gmr 2025 expo\\MULTI_HAZARD_DETECTION\\models\\hazard_detector.onnx\" pi@<PI_IP>:~/multi_hazard_detection/models/${NC}"
echo -e "   ${GREEN}scp \"D:\\Thynx\\gmr 2025 expo\\MULTI_HAZARD_DETECTION\\models\\hazard_detector.onnx.data\" pi@<PI_IP>:~/multi_hazard_detection/models/${NC}"
echo ""
echo "2. Run video-only detection:"
echo -e "   ${GREEN}cd ~/multi_hazard_detection${NC}"
echo -e "   ${GREEN}./run_detection.sh --camera 0${NC}"
echo ""
echo "3. Find your USB cam mic device index:"
echo -e "   ${GREEN}./run_sound.sh --list-devices${NC}"
echo ""
echo "4. Run combined video + audio detection:"
echo -e "   ${GREEN}./run_combined.sh --camera 0 --audio-device <MIC_INDEX>${NC}"
echo ""
echo "   Or fullscreen for demo:"
echo -e "   ${GREEN}./run_combined.sh --camera 0 --audio-device <MIC_INDEX> --fullscreen${NC}"
echo ""
