import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_dfd_net():
    # Load trained weights
    model = YOLO("weights/best.pt")

    # Run validation on the expert-annotated test set
    metrics = model.val(
        data="data/tooth_crack.yaml",
        imgsz=640,
        conf=0.25,      # Early-warning confidence threshold
        iou=0.5,        # Standard IoU for crack detection
        split='test',   # Use dedicated test split
        save_json=True, # For COCO-style metric analysis
        plots=True      # Generate P-R curves and Confusion Matrix
    )
    
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.p[0]:.4f}")
    print(f"Recall: {metrics.box.r[0]:.4f}")

if __name__ == "__main__":
    validate_dfd_net()