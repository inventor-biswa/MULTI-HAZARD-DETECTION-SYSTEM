"""
Multi-Hazard Detection - Model Export Script

Exports the trained PyTorch model to ONNX format for Raspberry Pi deployment.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import argparse


def create_model(num_classes: int):
    """Recreate the MobileNetV2 model architecture."""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, num_classes)
    )
    return model


def export_to_onnx(model, save_path: Path, input_size: int = 224):
    """Export PyTorch model to ONNX format."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model exported: {save_path}")


def verify_onnx_model(onnx_path: Path):
    """Verify the exported ONNX model."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("✅ ONNX model verification passed!")
        
        # Print model info
        print(f"   Inputs: {[i.name for i in model.graph.input]}")
        print(f"   Outputs: {[o.name for o in model.graph.output]}")
        
        return True
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False


def test_onnx_inference(onnx_path: Path, num_classes: int):
    """Test ONNX inference with dummy input."""
    try:
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        # Create dummy input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"✅ ONNX inference test passed!")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Predicted class: {np.argmax(outputs[0])}")
        
        return True
    except Exception as e:
        print(f"❌ ONNX inference test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth model file")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📦 Multi-Hazard Detection - ONNX Export")
    print("="*60 + "\n")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    
    # Find model file
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = models_dir / "hazard_detector.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run train_model.py first!")
        return
    
    print(f"📁 Loading model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    class_names = checkpoint.get('class_names', ['fire', 'normal', 'oil_leak', 'steam_leak', 'water_leak'])
    num_classes = len(class_names)
    
    print(f"📊 Classes ({num_classes}): {class_names}")
    print(f"📊 Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.1f}%")
    
    # Create and load model
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    onnx_path = models_dir / "hazard_detector.onnx"
    print(f"\n🔄 Exporting to ONNX...")
    export_to_onnx(model, onnx_path)
    
    # Get file size
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"   Model size: {onnx_size:.1f} MB")
    
    # Verify ONNX model
    print(f"\n🔍 Verifying ONNX model...")
    verify_onnx_model(onnx_path)
    
    # Test inference
    print(f"\n🧪 Testing ONNX inference...")
    test_onnx_inference(onnx_path, num_classes)
    
    print("\n" + "="*60)
    print("✅ Export Complete!")
    print("="*60)
    print(f"\n📁 Files for Raspberry Pi deployment:")
    print(f"   - {onnx_path}")
    print(f"   - {models_dir / 'class_labels.json'}")
    print(f"   - {base_dir / 'scripts' / 'detect_hazard.py'}")
    print(f"\n💡 Run on Raspberry Pi:")
    print(f"   python detect_hazard.py --camera 0")


if __name__ == "__main__":
    main()
