"""
Multi-Hazard Detection - Real-Time Inference Script

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
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--delay", type=int, default=1, help="Display delay (ms)")
    parser.add_argument("--fullscreen", action="store_true", help="Fullscreen mode")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎯 Multi-Hazard Detection System")
    print("="*60 + "\n")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    model_path = args.model or str(models_dir / "hazard_detector.onnx")
    labels_path = args.labels or str(models_dir / "class_labels.json")
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run train_model.py and export_model.py first!")
        return
    
    # Initialize detector
    detector = HazardDetector(model_path, labels_path)
    renderer = DisplayRenderer(detector.class_names)
    
    # Open video source
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera {args.camera}"
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = Path(args.video).name
    else:
        print("❌ Specify --camera or --video")
        return
    
    if not cap.isOpened():
        print(f"❌ Could not open video source: {source_name}")
        return
    
    print(f"📹 Source: {source_name}")
    print(f"⚙️ Frame skip: {args.skip}")
    print(f"⌨️ Press 'q' to quit, 'f' for fullscreen toggle")
    print()
    
    # Create window
    window_name = "Multi-Hazard Detection"
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
        cv2.destroyAllWindows()
        print("✅ Detection stopped")


if __name__ == "__main__":
    main()
