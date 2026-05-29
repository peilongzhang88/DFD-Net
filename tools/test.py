import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_inference(source_path):
    # Load model
    model = YOLO("weights/best.pt")

    # Run inference for clinical or industrial scenarios
    results = model.predict(
        source=source_path,
        imgsz=640,
        conf=0.25,
        save=True,          # Save visualized results
        save_txt=True,      # Save detection coordinates
        line_width=2,       # Thin lines for sub-pixel cracks
        project="inference_results",
        name="tooth_crack_samples"
    )

    print(f"Inference complete. Results saved in: inference_results/")

if __name__ == "__main__":
    # Test on dental clinical samples or industrial PVEL-AD samples
    test_path = "data/samples/" 
    if os.path.exists(test_path):
        run_inference(test_path)
    else:
        print(f"Path not found: {test_path}. Please check your sample data directory.")