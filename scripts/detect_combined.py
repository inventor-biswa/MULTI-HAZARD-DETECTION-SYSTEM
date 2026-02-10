"""
Combined Multi-Hazard Detection - Video + Audio
=================================================
Runs BOTH visual hazard detection (camera) and abnormal sound detection
(USB camera microphone) simultaneously in a single unified display.

Usage:
    python detect_combined.py --camera 0 --audio-device 1
    python detect_combined.py --camera 0 --audio-device 1 --fullscreen
    python detect_combined.py --video path/to.mp4 --audio-device 1
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
from datetime import datetime

try:
    import onnxruntime as ort
except ImportError:
    print("❌ Please install onnxruntime: pip install onnxruntime")
    sys.exit(1)

try:
    import pyaudio
except ImportError:
    print("❌ Please install pyaudio: pip install pyaudio")
    sys.exit(1)

try:
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import from sibling modules
from detect_hazard import HazardDetector, CLASS_CONFIG
from detect_sound import SoundAnalyzer, AudioCapture, SAMPLE_RATE, CHUNK_SIZE, SOUND_CATEGORIES


# ═══════════════════════════════════════════════════════════════
# AUDIO THREAD
# ═══════════════════════════════════════════════════════════════

class AudioDetectionThread:
    """Runs sound detection in a background thread."""

    def __init__(self, device_index=None, threshold=2.0):
        self.device_index = device_index
        self.threshold = threshold
        self.running = False
        self.thread = None

        # Shared state (thread-safe via GIL for simple reads)
        self.current_result = {
            "is_abnormal": False,
            "category": "normal",
            "alert_level": "ok",
            "confidence": 0.0,
            "energy_ratio": 0.0,
            "dominant_freq": 0.0,
            "display_text": "Normal",
        }
        self.is_calibrated = False
        self.error = None

    def start(self):
        """Start the audio detection thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the audio detection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)

    def _run(self):
        """Background audio detection loop."""
        try:
            # Retry logic: USB device may be busy if camera is opening simultaneously
            capture = None
            for attempt in range(3):
                time.sleep(2)  # Wait for camera to finish grabbing USB device
                capture = AudioCapture(
                    device_index=self.device_index,
                    sample_rate=SAMPLE_RATE,
                    chunk_size=CHUNK_SIZE,
                )
                if capture.open_stream():
                    break
                capture.close()
                capture = None
                print(f"🔄 Audio retry {attempt+1}/3...")

            if capture is None or capture.stream is None:
                self.error = "Failed to open audio stream after 3 retries"
                return

            # Use the actual sample rate that was negotiated with the device
            actual_rate = getattr(capture, 'sample_rate', SAMPLE_RATE)

            analyzer = SoundAnalyzer(
                sample_rate=actual_rate,
                threshold_multiplier=self.threshold,
            )

            # Calibration
            print("🔄 Calibrating audio baseline (5 sec)...")
            calibration_chunks = []
            chunks_needed = int(5 * actual_rate / CHUNK_SIZE)

            for i in range(chunks_needed):
                if not self.running:
                    capture.close()
                    return
                chunk = capture.read_chunk()
                if chunk:
                    calibration_chunks.append(chunk)

            analyzer.calibrate_baseline(calibration_chunks)
            self.is_calibrated = True
            print("✅ Audio baseline calibrated!")

            # Detection loop
            while self.running:
                chunk = capture.read_chunk()
                if chunk is None:
                    continue

                result = analyzer.analyze(chunk)

                # Update shared state
                cat_config = SOUND_CATEGORIES.get(result["category"], {})
                self.current_result = {
                    "is_abnormal": result["is_abnormal"],
                    "category": result["category"],
                    "alert_level": result["alert_level"],
                    "confidence": result["confidence"],
                    "energy_ratio": result["energy_ratio"],
                    "dominant_freq": result["dominant_freq"],
                    "display_text": cat_config.get("display", "Normal").replace("\033[91m", "").replace("\033[93m", "").replace("\033[96m", "").replace("\033[95m", "").replace("\033[0m", ""),
                }

            capture.close()

        except Exception as e:
            self.error = str(e)
            print(f"❌ Audio thread error: {e}")


# ═══════════════════════════════════════════════════════════════
# COMBINED DISPLAY RENDERER
# ═══════════════════════════════════════════════════════════════

class CombinedRenderer:
    """Renders both video and audio detection results on frame."""

    def __init__(self, class_names: list):
        self.class_names = class_names
        self.flash_state = False
        self.last_flash_time = time.time()
        self.sound_flash = False
        self.sound_flash_time = time.time()

    def render(self, frame: np.ndarray, video_class: str, video_confidence: float,
               all_probs: np.ndarray, fps: float, idx_to_class: dict,
               audio_result: dict) -> np.ndarray:
        """Render combined detection overlay on frame."""
        h, w = frame.shape[:2]

        # Get video class config
        config = CLASS_CONFIG.get(video_class, CLASS_CONFIG["normal"])
        color = config["color_bgr"]
        alert_level = config["alert_level"]
        display_name = config["display"]

        # Check audio alert
        audio_abnormal = audio_result.get("is_abnormal", False)
        audio_alert = audio_result.get("alert_level", "ok")
        audio_text = audio_result.get("display_text", "Normal")

        # Determine overall alert
        is_critical = (alert_level == "critical") or (audio_alert == "critical")

        # Flash effect for critical alerts (video or audio)
        if is_critical:
            if time.time() - self.last_flash_time > 0.3:
                self.flash_state = not self.flash_state
                self.last_flash_time = time.time()

            if self.flash_state:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 150), -1)
                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        # Draw border
        border_thickness = 8 if is_critical else 4
        border_color = (0, 0, 255) if audio_abnormal and audio_alert == "critical" else color
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, border_thickness)

        # ─── Status Bar (bottom) ────────────────────────────
        status_bar_h = 140
        cv2.rectangle(frame, (0, h - status_bar_h), (w, h), (30, 30, 30), -1)

        # Row 1: Video detection status
        video_icon = "🎥" if alert_level == "ok" else "⚠"
        status_text = f"VIDEO: {display_name.upper()}"
        if alert_level != "ok":
            status_text += "!"

        cv2.putText(frame, status_text, (20, h - status_bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        conf_text = f"{video_confidence*100:.1f}%"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, conf_text, (w//2 - text_size[0] - 10, h - status_bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Row 2: Audio detection status
        audio_color = (0, 200, 0)  # Green = normal
        if audio_abnormal:
            if audio_alert == "critical":
                audio_color = (0, 0, 255)  # Red
            else:
                audio_color = (0, 200, 255)  # Yellow

        # Clean display text (remove ANSI codes)
        clean_audio = audio_text
        for code in ["\033[91m", "\033[93m", "\033[96m", "\033[95m", "\033[92m", "\033[0m"]:
            clean_audio = clean_audio.replace(code, "")

        audio_status = f"AUDIO: {clean_audio if audio_abnormal else 'Normal'}"
        cv2.putText(frame, audio_status, (w//2 + 10, h - status_bar_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)

        # Audio energy meter bar
        energy_ratio = audio_result.get("energy_ratio", 0)
        meter_x = 20
        meter_y = h - status_bar_h + 40
        meter_w = w - 40
        meter_h = 12

        # Background
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h),
                      (60, 60, 60), -1)

        # Filled portion
        fill_w = int(min(energy_ratio / 10.0, 1.0) * meter_w)
        if fill_w > 0:
            # Green to yellow to red gradient
            if energy_ratio < 2:
                meter_color = (0, 200, 0)
            elif energy_ratio < 5:
                meter_color = (0, 200, 255)
            else:
                meter_color = (0, 0, 255)
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h),
                          meter_color, -1)

        # Threshold markers
        for mult in [2, 5, 8]:
            marker_x = meter_x + int((mult / 10.0) * meter_w)
            cv2.line(frame, (marker_x, meter_y), (marker_x, meter_y + meter_h), (255, 255, 255), 1)

        # Audio level text
        cv2.putText(frame, f"Level: {energy_ratio:.1f}x", (20, h - status_bar_h + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        freq_text = f"Freq: {audio_result.get('dominant_freq', 0):.0f}Hz"
        cv2.putText(frame, freq_text, (180, h - status_bar_h + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Separator
        cv2.line(frame, (10, h - status_bar_h + 78), (w - 10, h - status_bar_h + 78), (60, 60, 60), 1)

        # Row 3: Class indicator buttons
        indicator_y = h - 15
        indicator_h = 30

        # Visual classes
        all_classes = self.class_names + ["sound"]
        indicator_w = (w - 40) // len(all_classes)

        for i, cls in enumerate(all_classes):
            x = 20 + i * indicator_w

            if cls == "sound":
                # Sound indicator
                is_active = audio_abnormal
                cls_color = audio_color
                text = "Sound"
            else:
                is_active = (cls == video_class)
                cls_config = CLASS_CONFIG.get(cls, CLASS_CONFIG["normal"])
                cls_color = cls_config["color_bgr"]
                text = cls_config["display"]

            if is_active:
                cv2.rectangle(frame, (x, indicator_y - indicator_h),
                              (x + indicator_w - 6, indicator_y), cls_color, -1)
                text_color = (0, 0, 0) if sum(cls_color) > 400 else (255, 255, 255)
            else:
                cv2.rectangle(frame, (x, indicator_y - indicator_h),
                              (x + indicator_w - 6, indicator_y), (60, 60, 60), -1)
                cv2.rectangle(frame, (x, indicator_y - indicator_h),
                              (x + indicator_w - 6, indicator_y), (80, 80, 80), 1)
                text_color = (150, 150, 150)

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x + (indicator_w - 6 - text_size[0]) // 2
            text_y = indicator_y - (indicator_h - text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        # ─── Top Bar ────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "Multi-Hazard Detection [V+A]", (w//2 - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Combined Video + Audio Hazard Detection")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g., 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--labels", type=str, default=None, help="Path to class labels JSON")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--delay", type=int, default=1, help="Display delay (ms)")
    parser.add_argument("--fullscreen", action="store_true", help="Fullscreen mode")
    parser.add_argument("--audio-device", type=int, default=None, help="Audio device index for microphone")
    parser.add_argument("--audio-threshold", type=float, default=2.0, help="Audio sensitivity (default: 2.0)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🎯 Multi-Hazard Detection System [VIDEO + AUDIO]")
    print("=" * 60 + "\n")

    # Setup paths
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"

    model_path = args.model or str(models_dir / "hazard_detector.onnx")
    labels_path = args.labels or str(models_dir / "class_labels.json")

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Initialize video detector
    detector = HazardDetector(model_path, labels_path)
    renderer = CombinedRenderer(detector.class_names)

    # Start audio detection thread
    audio_thread = AudioDetectionThread(
        device_index=args.audio_device,
        threshold=args.audio_threshold,
    )
    audio_thread.start()
    print(f"🎤 Audio detection starting on device {args.audio_device}...")

    # Wait for audio calibration
    timeout = 10
    while not audio_thread.is_calibrated and timeout > 0:
        time.sleep(0.5)
        timeout -= 0.5
        if audio_thread.error:
            print(f"⚠️  Audio failed: {audio_thread.error}")
            print("   Continuing with video-only mode")
            break

    # Open video source
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = Path(args.video).name
    else:
        print("❌ Specify --camera or --video")
        audio_thread.stop()
        return

    if not cap.isOpened():
        print(f"❌ Could not open video source: {source_name}")
        audio_thread.stop()
        return

    print(f"📹 Video: {source_name}")
    print(f"⌨️  Press 'q' to quit, 'f' for fullscreen\n")

    # Create window
    window_name = "Multi-Hazard Detection [V+A]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0
    fps_start = time.time()
    fps = 0.0
    last_prediction = ("normal", 0.0, np.zeros(len(detector.class_names)))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1

            # Video inference
            if frame_count % args.skip == 0:
                class_name, confidence, all_probs = detector.predict(frame)
                last_prediction = (class_name, confidence, all_probs)
            else:
                class_name, confidence, all_probs = last_prediction

            # FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()

            # Get current audio state
            audio_result = audio_thread.current_result

            # Render combined display
            display_frame = renderer.render(
                frame.copy(),
                class_name,
                confidence,
                all_probs,
                fps,
                detector.idx_to_class,
                audio_result,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(args.delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                current = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                )

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")

    finally:
        audio_thread.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Combined detection stopped")


if __name__ == "__main__":
    main()
