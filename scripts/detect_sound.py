"""
Abnormal Sound Detection Module
================================
Detects abnormal sounds (loud bangs, grinding, hissing, alarms, etc.)
using the USB camera's built-in microphone.

This module can run standalone or alongside the visual hazard detector.

Dependencies:
    pip install pyaudio numpy scipy

Usage:
    # List available audio devices (find your USB cam mic)
    python detect_sound.py --list-devices

    # Run with default microphone
    python detect_sound.py

    # Run with specific device index (from --list-devices)
    python detect_sound.py --device 1

    # Run with custom thresholds
    python detect_sound.py --device 1 --threshold 0.05 --calibrate

    # Run alongside visual detection (separate terminal)
    python detect_sound.py --device 1 --alert-file ../models/sound_alert.json
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
    print("❌ PyAudio not found. Install it:")
    print("   pip install pyaudio")
    print("   (On Windows, you may need: pip install pipwin && pipwin install pyaudio)")
    sys.exit(1)

try:
    from scipy import signal as scipy_signal
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    print("⚠️  scipy not found. Using basic detection only.")
    print("   For better detection: pip install scipy")
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Audio capture settings
SAMPLE_RATE = 44100          # Hz - standard audio sample rate
CHUNK_SIZE = 4096            # Samples per buffer (~93ms at 44100Hz)
CHANNELS = 1                 # Mono (most USB cam mics are mono)
FORMAT = pyaudio.paInt16     # 16-bit audio

# Detection settings
ENERGY_WINDOW_SEC = 0.5      # Sliding window for energy calculation (seconds)
BASELINE_DURATION_SEC = 5    # Duration to capture baseline ambient noise
ALERT_COOLDOWN_SEC = 2.0     # Minimum time between alerts

# Sound classification thresholds (relative to baseline)
SOUND_CATEGORIES = {
    "loud_bang": {
        "display": "🔊 LOUD BANG / IMPACT",
        "energy_multiplier": 8.0,       # 8x above baseline
        "min_freq": 50,                 # Hz
        "max_freq": 2000,               # Hz
        "alert_level": "critical",
        "color": "\033[91m",            # Red
    },
    "grinding": {
        "display": "⚙️  GRINDING / MECHANICAL",
        "energy_multiplier": 3.0,
        "min_freq": 500,
        "max_freq": 8000,
        "alert_level": "warning",
        "color": "\033[93m",            # Yellow
    },
    "hissing": {
        "display": "💨 HISSING / GAS LEAK",
        "energy_multiplier": 2.5,
        "min_freq": 2000,
        "max_freq": 16000,
        "alert_level": "warning",
        "color": "\033[96m",            # Cyan
    },
    "alarm": {
        "display": "🚨 ALARM / SIREN",
        "energy_multiplier": 4.0,
        "min_freq": 800,
        "max_freq": 4000,
        "alert_level": "critical",
        "color": "\033[95m",            # Magenta
    },
    "abnormal": {
        "display": "⚠️  ABNORMAL SOUND",
        "energy_multiplier": 2.0,
        "min_freq": 50,
        "max_freq": 16000,
        "alert_level": "warning",
        "color": "\033[93m",            # Yellow
    },
}

RESET_COLOR = "\033[0m"
GREEN = "\033[92m"
DIM = "\033[90m"


# ═══════════════════════════════════════════════════════════════
# SOUND ANALYZER
# ═══════════════════════════════════════════════════════════════

class SoundAnalyzer:
    """Analyzes audio data for abnormal sound patterns."""

    def __init__(self, sample_rate=SAMPLE_RATE, threshold_multiplier=2.0):
        self.sample_rate = sample_rate
        self.threshold_multiplier = threshold_multiplier

        # Baseline (ambient) noise profile
        self.baseline_energy = None
        self.baseline_spectrum = None
        self.is_calibrated = False

        # Rolling buffers
        buffer_size = int(ENERGY_WINDOW_SEC * sample_rate / CHUNK_SIZE) + 1
        self.energy_history = deque(maxlen=buffer_size)
        self.spectrum_history = deque(maxlen=buffer_size)

        # Alert management
        self.last_alert_time = 0
        self.alert_history = deque(maxlen=100)

    def audio_to_float(self, audio_data: bytes) -> np.ndarray:
        """Convert raw audio bytes to normalized float array."""
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        # Normalize to [-1, 1]
        samples = samples / 32768.0
        return samples

    def compute_energy(self, samples: np.ndarray) -> float:
        """Compute RMS energy of audio samples."""
        return float(np.sqrt(np.mean(samples ** 2)))

    def compute_spectrum(self, samples: np.ndarray) -> tuple:
        """Compute frequency spectrum using FFT."""
        if HAS_SCIPY:
            # Use scipy for better FFT
            freqs = rfftfreq(len(samples), 1.0 / self.sample_rate)
            spectrum = np.abs(rfft(samples))
        else:
            # Fallback to numpy
            spectrum = np.abs(np.fft.rfft(samples))
            freqs = np.fft.rfftfreq(len(samples), 1.0 / self.sample_rate)

        return freqs, spectrum

    def get_band_energy(self, freqs: np.ndarray, spectrum: np.ndarray,
                        low_freq: float, high_freq: float) -> float:
        """Get energy in a specific frequency band."""
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        if not np.any(mask):
            return 0.0
        return float(np.mean(spectrum[mask] ** 2))

    def calibrate_baseline(self, audio_chunks: list):
        """Calibrate the baseline noise profile from ambient recording."""
        all_energies = []
        all_spectra = []

        for chunk in audio_chunks:
            samples = self.audio_to_float(chunk)
            energy = self.compute_energy(samples)
            all_energies.append(energy)

            freqs, spectrum = self.compute_spectrum(samples)
            all_spectra.append(spectrum)

        self.baseline_energy = np.mean(all_energies)
        self.baseline_spectrum = np.mean(all_spectra, axis=0)
        self.baseline_freqs = freqs
        self.is_calibrated = True

        # Set minimum baseline to avoid division by zero
        if self.baseline_energy < 0.001:
            self.baseline_energy = 0.001

        return self.baseline_energy

    def analyze(self, audio_data: bytes) -> dict:
        """
        Analyze an audio chunk for abnormal sounds.

        Returns:
            dict with keys:
                - is_abnormal: bool
                - category: str (sound category name)
                - display: str (display text)
                - energy: float (current energy)
                - energy_ratio: float (ratio to baseline)
                - dominant_freq: float (dominant frequency in Hz)
                - alert_level: str ("ok", "warning", "critical")
                - confidence: float (0-1)
        """
        samples = self.audio_to_float(audio_data)
        energy = self.compute_energy(samples)
        freqs, spectrum = self.compute_spectrum(samples)

        self.energy_history.append(energy)
        self.spectrum_history.append(spectrum)

        result = {
            "is_abnormal": False,
            "category": "normal",
            "display": f"{GREEN}✓ Normal{RESET_COLOR}",
            "energy": energy,
            "energy_ratio": 1.0,
            "dominant_freq": 0.0,
            "alert_level": "ok",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

        if not self.is_calibrated:
            return result

        # Calculate energy ratio to baseline
        energy_ratio = energy / self.baseline_energy
        result["energy_ratio"] = energy_ratio

        # Find dominant frequency
        if len(spectrum) > 0:
            peak_idx = np.argmax(spectrum[1:]) + 1  # Skip DC component
            if peak_idx < len(freqs):
                result["dominant_freq"] = float(freqs[peak_idx])

        # Check cooldown
        now = time.time()
        if now - self.last_alert_time < ALERT_COOLDOWN_SEC:
            # Still in cooldown, but update energy info
            return result

        # ─── Classification Logic ───────────────────────────
        # Check each sound category from most specific to least
        detected_category = None
        max_confidence = 0.0

        for cat_name, cat_config in SOUND_CATEGORIES.items():
            if cat_name == "abnormal":
                continue  # Check this last as fallback

            required_multiplier = cat_config["energy_multiplier"] * self.threshold_multiplier / 2.0
            band_energy = self.get_band_energy(
                freqs, spectrum,
                cat_config["min_freq"],
                cat_config["max_freq"]
            )

            # Compare band energy to baseline band energy
            baseline_band = self.get_band_energy(
                self.baseline_freqs, self.baseline_spectrum,
                cat_config["min_freq"],
                cat_config["max_freq"]
            )

            if baseline_band < 0.0001:
                baseline_band = 0.0001

            band_ratio = band_energy / baseline_band

            if energy_ratio >= required_multiplier and band_ratio >= required_multiplier:
                confidence = min(1.0, (energy_ratio / required_multiplier) * 0.5 +
                                 (band_ratio / required_multiplier) * 0.5)
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_category = cat_name

        # Fallback: general abnormal detection
        if detected_category is None:
            abnormal_config = SOUND_CATEGORIES["abnormal"]
            required = abnormal_config["energy_multiplier"] * self.threshold_multiplier / 2.0
            if energy_ratio >= required:
                detected_category = "abnormal"
                max_confidence = min(1.0, energy_ratio / required)

        # Set result
        if detected_category:
            cat = SOUND_CATEGORIES[detected_category]
            result["is_abnormal"] = True
            result["category"] = detected_category
            result["display"] = f"{cat['color']}{cat['display']}{RESET_COLOR}"
            result["alert_level"] = cat["alert_level"]
            result["confidence"] = max_confidence
            self.last_alert_time = now

            # Log to history
            self.alert_history.append({
                "timestamp": result["timestamp"],
                "category": detected_category,
                "energy_ratio": energy_ratio,
                "confidence": max_confidence,
                "dominant_freq": result["dominant_freq"],
            })

        return result


# ═══════════════════════════════════════════════════════════════
# AUDIO CAPTURE
# ═══════════════════════════════════════════════════════════════

class AudioCapture:
    """Captures audio from a microphone device."""

    def __init__(self, device_index=None, sample_rate=SAMPLE_RATE,
                 chunk_size=CHUNK_SIZE, channels=CHANNELS):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.pa = pyaudio.PyAudio()
        self.stream = None

    def list_devices(self):
        """List all available audio input devices."""
        print("\n" + "=" * 60)
        print("🎤 Available Audio Input Devices")
        print("=" * 60)

        device_count = self.pa.get_device_count()
        input_devices = []

        for i in range(device_count):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                input_devices.append((i, info))
                is_default = " ⭐ DEFAULT" if i == self.pa.get_default_input_device_info()['index'] else ""
                print(f"\n  [{i}] {info['name']}{is_default}")
                print(f"      Channels: {info['maxInputChannels']}")
                print(f"      Sample Rate: {int(info['defaultSampleRate'])} Hz")

        if not input_devices:
            print("\n  ❌ No audio input devices found!")
            print("     Make sure your USB camera is connected and has a microphone.")

        print("\n" + "-" * 60)
        print("💡 Tip: Use --device <index> to select a specific device")
        print("   Example: python detect_sound.py --device 1")
        print("=" * 60 + "\n")

        return input_devices

    def open_stream(self):
        """Open the audio capture stream. Auto-detects channel count."""
        # Auto-detect channels from device if needed
        actual_channels = self.channels
        if self.device_index is not None:
            try:
                info = self.pa.get_device_info_by_index(self.device_index)
                max_ch = int(info['maxInputChannels'])
                if max_ch > 0:
                    # Use device's channel count (but cap at 2 for sanity)
                    actual_channels = min(max_ch, 2)
                    print(f"🎤 Device [{self.device_index}] {info['name']} - using {actual_channels} channel(s)")
            except Exception:
                pass

        # Try opening with detected channels, then fallback
        for ch in [actual_channels, 2, 1]:
            try:
                kwargs = {
                    'format': FORMAT,
                    'channels': ch,
                    'rate': self.sample_rate,
                    'input': True,
                    'frames_per_buffer': self.chunk_size,
                }
                if self.device_index is not None:
                    kwargs['input_device_index'] = self.device_index

                self.stream = self.pa.open(**kwargs)
                self.actual_channels = ch
                if ch != actual_channels:
                    print(f"   ℹ️  Opened with {ch} channel(s) instead")
                return True
            except Exception as e:
                last_error = e
                continue

        print(f"❌ Failed to open audio stream: {last_error}")
        print("   Try a different device with --device <index>")
        print("   Use --list-devices to see available devices")
        return False

    def read_chunk(self) -> bytes:
        """Read one chunk of audio data. Downmixes to mono if needed."""
        if self.stream and self.stream.is_active():
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                # Downmix to mono if multi-channel
                if hasattr(self, 'actual_channels') and self.actual_channels > 1:
                    samples = np.frombuffer(data, dtype=np.int16)
                    # Reshape to (num_frames, num_channels) and average
                    samples = samples.reshape(-1, self.actual_channels)
                    mono = samples.mean(axis=1).astype(np.int16)
                    return mono.tobytes()
                return data
            except Exception as e:
                print(f"⚠️  Audio read error: {e}")
                return None
        return None

    def close(self):
        """Close the audio stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


# ═══════════════════════════════════════════════════════════════
# VISUAL METER (Console-based)
# ═══════════════════════════════════════════════════════════════

def render_meter(energy_ratio: float, max_ratio: float = 10.0, width: int = 40) -> str:
    """Render a visual audio level meter."""
    filled = int(min(energy_ratio / max_ratio, 1.0) * width)

    bar = ""
    for i in range(width):
        if i < filled:
            if i < width * 0.6:
                bar += "\033[92m█\033[0m"  # Green
            elif i < width * 0.8:
                bar += "\033[93m█\033[0m"  # Yellow
            else:
                bar += "\033[91m█\033[0m"  # Red
        else:
            bar += "\033[90m░\033[0m"

    return bar


def print_header():
    """Print the detection header."""
    print("\n" + "=" * 60)
    print("🔊 Abnormal Sound Detection System")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# ALERT FILE (for integration with visual detection)
# ═══════════════════════════════════════════════════════════════

class AlertFileWriter:
    """Writes sound alerts to a JSON file for cross-process integration."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lock = threading.Lock()

    def write_alert(self, result: dict):
        """Write current sound status to alert file."""
        alert_data = {
            "timestamp": result["timestamp"],
            "is_abnormal": result["is_abnormal"],
            "category": result["category"],
            "alert_level": result["alert_level"],
            "confidence": result["confidence"],
            "energy_ratio": result["energy_ratio"],
            "dominant_freq": result["dominant_freq"],
        }

        with self.lock:
            try:
                with open(self.filepath, 'w') as f:
                    json.dump(alert_data, f, indent=2)
            except Exception as e:
                pass  # Silently fail for non-critical file writes


# ═══════════════════════════════════════════════════════════════
# MAIN DETECTION LOOP
# ═══════════════════════════════════════════════════════════════

def run_detection(args):
    """Main detection loop."""
    print_header()

    # Initialize audio capture
    capture = AudioCapture(
        device_index=args.device,
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_SIZE,
    )

    if not capture.open_stream():
        capture.close()
        return

    device_name = "Default"
    if args.device is not None:
        info = capture.pa.get_device_info_by_index(args.device)
        device_name = info['name']
    else:
        info = capture.pa.get_default_input_device_info()
        device_name = info['name']

    print(f"\n🎤 Device: {device_name}")
    print(f"📊 Sample Rate: {SAMPLE_RATE} Hz")
    print(f"📦 Chunk Size: {CHUNK_SIZE} samples")
    print(f"🎚️  Threshold Multiplier: {args.threshold}x")

    # Initialize analyzer
    analyzer = SoundAnalyzer(
        sample_rate=SAMPLE_RATE,
        threshold_multiplier=args.threshold,
    )

    # Optional alert file writer
    alert_writer = None
    if args.alert_file:
        alert_writer = AlertFileWriter(args.alert_file)
        print(f"📄 Alert file: {args.alert_file}")

    # ─── Calibration Phase ──────────────────────────────────
    if args.calibrate or not args.no_calibrate:
        print(f"\n🔄 Calibrating baseline ambient noise...")
        print(f"   Please keep the environment at normal noise level")
        print(f"   for {BASELINE_DURATION_SEC} seconds...")
        print()

        calibration_chunks = []
        chunks_needed = int(BASELINE_DURATION_SEC * SAMPLE_RATE / CHUNK_SIZE)

        for i in range(chunks_needed):
            chunk = capture.read_chunk()
            if chunk:
                calibration_chunks.append(chunk)

            # Progress bar
            progress = int((i + 1) / chunks_needed * 30)
            bar = "█" * progress + "░" * (30 - progress)
            pct = (i + 1) / chunks_needed * 100
            sys.stdout.write(f"\r   [{bar}] {pct:.0f}%")
            sys.stdout.flush()

        baseline = analyzer.calibrate_baseline(calibration_chunks)
        print(f"\n\n✅ Baseline calibrated! Ambient energy: {baseline:.6f}")
        print(f"   Sounds {args.threshold * 2:.1f}x louder will trigger alerts")
    else:
        # Use a default baseline
        analyzer.baseline_energy = 0.01
        analyzer.baseline_spectrum = np.ones(CHUNK_SIZE // 2 + 1) * 0.001
        analyzer.baseline_freqs = np.linspace(0, SAMPLE_RATE / 2, CHUNK_SIZE // 2 + 1)
        analyzer.is_calibrated = True
        print("\n⚠️  Using default baseline (no calibration)")

    # ─── Detection Phase ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"🟢 Listening for abnormal sounds... (Press Ctrl+C to stop)")
    print(f"{'─' * 60}\n")

    try:
        frame_count = 0
        while True:
            chunk = capture.read_chunk()
            if chunk is None:
                continue

            frame_count += 1
            result = analyzer.analyze(chunk)

            # Write to alert file (for integration)
            if alert_writer:
                alert_writer.write_alert(result)

            # Console display
            meter = render_meter(result["energy_ratio"])
            ratio_text = f"{result['energy_ratio']:6.2f}x"

            if result["is_abnormal"]:
                # Alert display
                print(f"\r  {meter} {ratio_text} │ {result['display']}"
                      f" (conf: {result['confidence']:.0%},"
                      f" freq: {result['dominant_freq']:.0f}Hz)")
            else:
                # Normal display (update in place)
                sys.stdout.write(f"\r  {meter} {ratio_text} │ {GREEN}✓ Normal{RESET_COLOR}"
                                f" │ freq: {result['dominant_freq']:5.0f}Hz  ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n\n{'─' * 60}")
        print("⏹️  Detection stopped\n")

        # Print summary
        if analyzer.alert_history:
            print(f"📊 Detection Summary:")
            print(f"   Total alerts: {len(analyzer.alert_history)}")

            # Count by category
            categories = {}
            for alert in analyzer.alert_history:
                cat = alert['category']
                categories[cat] = categories.get(cat, 0) + 1

            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                cat_display = SOUND_CATEGORIES[cat]["display"]
                print(f"   • {cat_display}: {count} times")
        else:
            print("📊 No abnormal sounds detected during this session")

    finally:
        # Clean up alert file
        if alert_writer and os.path.exists(args.alert_file):
            try:
                os.remove(args.alert_file)
            except:
                pass

        capture.close()
        print("✅ Audio capture stopped\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Abnormal Sound Detection using USB Camera Microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_sound.py --list-devices          # Find your USB cam mic
  python detect_sound.py --device 1              # Use device index 1
  python detect_sound.py --device 1 --calibrate  # With calibration
  python detect_sound.py --threshold 3.0         # Higher threshold (less sensitive)
  python detect_sound.py --alert-file ../models/sound_alert.json  # For integration
        """
    )

    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (use --list-devices to find)")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Sensitivity threshold multiplier (default: 2.0, higher = less sensitive)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Force calibration of ambient noise baseline")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip calibration, use default baseline")
    parser.add_argument("--alert-file", type=str, default=None,
                        help="Path to write alert JSON (for integration with visual detection)")

    args = parser.parse_args()

    if args.list_devices:
        capture = AudioCapture()
        capture.list_devices()
        capture.close()
        return

    run_detection(args)


if __name__ == "__main__":
    main()
