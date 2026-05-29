# 🦷 DFD-Net: Direction-Aware and Frequency-Aligned Network for Precise Tooth Crack Detection

<div align="center">

[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/Base-YOLOv11-00FFFF?style=flat-square&logo=gitl)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Academic](https://img.shields.io/badge/Status-Top_Tier_Journal-red?style=flat-square)]()

**Beyond the limits of human visual resolution: A specialized detector for sub-pixel linear manifolds.**

</div>

---

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Complete Dataset** | 753 expert-annotated intraoral images (YOLO format) | [Download](https://doi.org/10.5281/zenodo.20439973) |
| **Pre-trained Weights** | Optimized `.pt` files for DFD-Net (Ours) | [Download](https://doi.org/10.5281/zenodo.20439973) |
| **Validation Samples** | Zero-shot industrial test samples (PVEL-AD) | [Download](https://doi.org/10.5281/zenodo.20439973) |


## 💡 Motivation & Highlights
Tooth cracks are the "silent killers" of natural dentition. DFD-Net is engineered to capture 1D-like pathological features while isolating complex clinical noise.

*   **🟢 State-of-the-Art Precision**: Achieves a superior **0.925 mAP50** on clinical benchmarks.
*   **🔵 Architectural Innovation**: Introduces `DEGConv`, `FAAFusion`, and `DSAM` for direction-aware encoding.
*   **🟠 Edge-AI Optimized**: Real-time inference (>45 FPS) on **NVIDIA Jetson Orin Nano**.
*   **🔴 Robust Generalization**: Only **2.92%** performance decay in zero-shot industrial transfer.

---

## 🏗️ Overall Architecture (Strategic Innovation)
DFD-Net fundamentally re-engineers the isotropic convolution paradigm to focus on directional micro-textures.

![Figure 2](assets/fig2.png)
> **Figure 2.** Overall Architecture of the DFD-Net. It features a **Direction-Aware Backbone** for manifold encoding, a **Frequency-Aligned Neck** for orientation calibration, and a **High-Resolution Head** for texture refinement.

---

## 🧪 Experimental Benchmarking
DFD-Net exhibits remarkable stability and high sensitivity to micro-textures smaller than 10 pixels.

### 📊 Metric Curves & Convergence
![Figure 5](assets/fig5.png)
> **Figure 5.** Training loss and metric curves. The model reaches peak precision rapidly with minimal oscillation, validating the efficacy of the **Slide Weight Function (SWF)**.

| Method | Precision | Recall | mAP50 |
| :--- | :---: | :---: | :---: | :---: |
| YOLOv11n (Baseline) | 0.872 | 0.803 | 0.872 |
| **DFD-Net (Ours)** | **0.928** | **0.865** | **0.925** |

---

## 🏥 Clinical & Edge Deployment
Designed for real-world intraoral scanning, DFD-Net maintains high sensitivity under saliva-induced specularity.

![Figure 7](assets/fig7.png)
> **Figure 7.** Real-time tooth crack detection on the **NVIDIA Jetson Orin Nano** terminal. Left: Baseline (False Negatives); Right: DFD-Net (Precise localization).

---

## 🌍 Zero-Shot Cross-Domain Generalization
Testing the "Universal Geometric Primitives": Moving from biological enamel to industrial silicon without any fine-tuning.

![Figure 9](assets/fig9.png)
> **Figure 9. Qualitative 4×3 comparison of zero-shot generalization results.**
> - **Col I**: Raw infrared images (PVEL-AD).
> - **Col II**: Baseline detections (Fragmented/Missing).
> - **Col III**: **DFD-Net results (Continuous/Precise)**.
> - *Rows (a)–(d) highlight resilience against intense busbar interference.*

---

## 🛠️ Quick Start & Installation

### ⚙️ Environment Setup
```bash
# Clone the repository
git clone https://github.com/YourUsername/DFD-Net.git && cd DFD-Net

# One-click installation
pip install -r requirements.txt


🏎️ Inference & Training

# Run Clinical Inference
python tools/test.py --weights weights/best.pt --source data/samples/

# Reproduce Results
python tools/train.py --config configs/yolo11-DFD-Net.yaml


