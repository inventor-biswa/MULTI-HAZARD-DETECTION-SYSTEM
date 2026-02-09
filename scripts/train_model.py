"""
Multi-Hazard Detection - Model Training Script

Trains a MobileNetV2-based classifier for 5-class hazard detection:
- fire
- normal (no leakage)
- oil_leak
- steam_leak
- water_leak
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Configuration
DEFAULT_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "num_classes": 5,
    "input_size": 224,
    "patience": 10,  # Early stopping patience
}


def get_data_transforms():
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(num_classes: int, pretrained: bool = True):
    """Create MobileNetV2 model with custom classifier."""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Freeze early layers (optional, can be tuned)
    for param in model.features[:10].parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, num_classes)
    )
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def evaluate_test_set(model, dataloader, device, class_names):
    """Evaluate on test set with per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    print("\n" + "="*50)
    print("📊 Test Set Results")
    print("="*50)
    
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    
    for idx in range(len(class_names)):
        if class_total[idx] > 0:
            acc = 100. * class_correct[idx] / class_total[idx]
            print(f"  {class_names[idx]:15s}: {class_correct[idx]:4d}/{class_total[idx]:4d} ({acc:.1f}%)")
    
    overall_acc = 100. * total_correct / total_samples
    print("-"*50)
    print(f"  {'Overall':15s}: {total_correct:4d}/{total_samples:4d} ({overall_acc:.1f}%)")
    
    return overall_acc


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Training plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Hazard Detection Model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 Multi-Hazard Detection - Model Training")
    print("="*60 + "\n")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "dataset"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Check dataset exists
    if not (dataset_dir / "train").exists():
        print("❌ Dataset not found! Run extract_frames.py first.")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # Data transforms and loaders
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = datasets.ImageFolder(dataset_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(dataset_dir / "val", transform=val_transform)
    test_dataset = datasets.ImageFolder(dataset_dir / "test", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    class_names = train_dataset.classes
    print(f"📦 Classes: {class_names}")
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Update class labels JSON
    labels_path = models_dir / "class_labels.json"
    labels_data = {
        "classes": class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": {v: k for k, v in train_dataset.class_to_idx.items()},
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
    with open(labels_path, 'w') as f:
        json.dump(labels_data, f, indent=4)
    print(f"📝 Saved class labels: {labels_path}")
    
    # Create model
    model = create_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n⚙️ Training Config:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print()
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            model_path = models_dir / "hazard_detector.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, model_path)
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_CONFIG["patience"]:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Plot training history
    plot_path = models_dir / "training_history.png"
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Load best model for testing
    checkpoint = torch.load(models_dir / "hazard_detector.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_acc = evaluate_test_set(model, test_loader, device, class_names)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"📊 Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"📊 Test Accuracy: {test_acc:.1f}%")
    print(f"📁 Model saved: {models_dir / 'hazard_detector.pth'}")
    print("\n💡 Next step: Run export_model.py to convert to ONNX")


if __name__ == "__main__":
    main()
