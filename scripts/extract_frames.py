"""
Multi-Hazard Detection - Frame Extraction Script

Extracts frames from training videos and splits them into train/val/test folders.
Videos:
- FIRE.mp4 -> fire
- NO_LEAKAGE.mp4 -> normal  
- OIL_LEAKAGE.mp4 -> oil_leak
- STEAM_LEAKAGE.mp4 -> steam_leak
- WATER_LEAKAGE.mp4 -> water_leak
"""

import cv2
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
VIDEO_MAPPING = {
    "FIRE.mp4": "fire",
    "NO_LEAKAGE.mp4": "normal",
    "OIL_LEAKAGE.mp4": "oil_leak",
    "STEAM_LEAKAGE.mp4": "steam_leak",
    "WATER_LEAKAGE.mp4": "water_leak"
}

FRAME_SKIP = 15  # Extract every 15th frame
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def extract_frames_from_video(video_path: Path, output_dir: Path, class_name: str, frame_skip: int = 15):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Base output directory (dataset folder)
        class_name: Name of the class (folder name)
        frame_skip: Extract every nth frame
    
    Returns:
        List of extracted frame paths
    """
    if not video_path.exists():
        print(f"  ⚠️ Video not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Could not open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  📊 Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"  📊 Expected extracted frames: ~{total_frames // frame_skip}")
    
    # Create temporary folder for extracted frames
    temp_dir = output_dir / "temp" / class_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_paths = []
    frame_count = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames, desc=f"    Extracting", unit="frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Resize frame to 224x224 for training
            frame_resized = cv2.resize(frame, (224, 224))
            
            # Save frame
            frame_path = temp_dir / f"{class_name}_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_paths.append(frame_path)
            saved_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"  ✅ Extracted {saved_count} frames")
    return extracted_paths


def split_and_move_frames(frame_paths: list, output_dir: Path, class_name: str,
                          train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Split frames into train/val/test and move to respective folders.
    """
    if not frame_paths:
        return
    
    # Shuffle frames
    random.shuffle(frame_paths)
    
    n_total = len(frame_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_frames = frame_paths[:n_train]
    val_frames = frame_paths[n_train:n_train + n_val]
    test_frames = frame_paths[n_train + n_val:]
    
    # Create class folders
    train_dir = output_dir / "train" / class_name
    val_dir = output_dir / "val" / class_name
    test_dir = output_dir / "test" / class_name
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Move frames
    for i, frame_path in enumerate(train_frames):
        dest = train_dir / f"{class_name}_train_{i:05d}.jpg"
        shutil.move(str(frame_path), str(dest))
    
    for i, frame_path in enumerate(val_frames):
        dest = val_dir / f"{class_name}_val_{i:05d}.jpg"
        shutil.move(str(frame_path), str(dest))
    
    for i, frame_path in enumerate(test_frames):
        dest = test_dir / f"{class_name}_test_{i:05d}.jpg"
        shutil.move(str(frame_path), str(dest))
    
    print(f"  📁 Split: Train={len(train_frames)}, Val={len(val_frames)}, Test={len(test_frames)}")


def main():
    print("\n" + "="*60)
    print("🎬 Multi-Hazard Detection - Frame Extraction")
    print("="*60 + "\n")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    video_dir = base_dir
    dataset_dir = base_dir / "dataset"
    
    # Clean existing dataset if any
    if dataset_dir.exists():
        print("🗑️ Cleaning existing dataset folder...")
        shutil.rmtree(dataset_dir)
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Video directory: {video_dir}")
    print(f"📂 Dataset directory: {dataset_dir}")
    print(f"⚙️ Frame skip: {FRAME_SKIP}")
    print(f"⚙️ Split ratio: Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")
    print()
    
    # Process each video
    for video_name, class_name in VIDEO_MAPPING.items():
        video_path = video_dir / video_name
        
        print(f"\n📹 Processing: {video_name} -> '{class_name}'")
        print("-" * 40)
        
        # Extract frames
        frame_paths = extract_frames_from_video(
            video_path, 
            dataset_dir, 
            class_name, 
            FRAME_SKIP
        )
        
        # Split into train/val/test
        if frame_paths:
            split_and_move_frames(
                frame_paths, 
                dataset_dir, 
                class_name,
                TRAIN_RATIO, 
                VAL_RATIO
            )
    
    # Cleanup temp folder
    temp_dir = dataset_dir / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Dataset Summary")
    print("="*60)
    
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if split_dir.exists():
            total = 0
            print(f"\n{split.upper()}:")
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    total += count
                    print(f"  - {class_dir.name}: {count} images")
            print(f"  Total: {total} images")
    
    print("\n✅ Frame extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
