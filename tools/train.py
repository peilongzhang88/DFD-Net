import sys
import os
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_dfd_net():
    # Load DFD-Net configuration
    model = YOLO("configs/yolo11-DFD-Net.yaml")

    # Training settings strictly following Paper Section 3.1
    model.train(
        data="data/tooth_crack.yaml",
        epochs=300,
        imgsz=640,
        batch=32,
        optimizer='AdamW',
        lr0=0.01,
        weight_decay=0.0005,
        # Task-specific augmentation (Table 1)
        mosaic=0.0,    # Disable Mosaic for sub-pixel continuity
        mixup=0.0,     # Disable Mixup to preserve faint signatures
        hsv_h=0.0,     # Prevent color distortion
        flipud=0.5,    # Vertical flip for camera orientation
        name="DFD_Net_Final_Version",
        device=0,      # Change to 'cpu' if no GPU available
    )

if __name__ == "__main__":
    train_dfd_net()