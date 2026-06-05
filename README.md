# Shahed Interceptor - AI Drone Detection & Tracking

[![CI](https://github.com/rami-work/Shahed-Interceptor-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/rami-work/Shahed-Interceptor-AI/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time detection and tracking of Shahed drones using YOLO11 with a two-stage interception guidance system (YOLO scan → CSRT lock-on → visual servoing).

> **Topics:** `drone-detection` `yolo` `computer-vision` `object-detection` `shahed` `interceptor`

## Demo

| Detection | Lock-On Tracking |
|-----------|-----------------|
| ![Detection](assets/detect1.gif) | ![Lock-On](assets/lockin_final1.gif) |

## Results

| Metric | Value |
|--------|-------|
| mAP50 | **98%** |
| Precision | **97.5%** |
| Recall | **94.9%** |
| mAP50-95 | **78%** |
| Model | YOLO11-Small |
| Resolution | 928px |
| Dataset | Custom Shahed dataset (1,582 images) |
| Hardware | NVIDIA RTX 2070 SUPER |

## How It Works

```
Stage 1: YOLO Detection
    ↓ (target visible for >1s)
Stage 2: CSRT Blind Tracking
    ↓ (tracker locked)
Stage 3: Visual Servoing Guidance
    → YAW command (left/right)
    → ALT command (up/down)
```

## Project Structure

```
Shahed-Interceptor-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── detect.py          # simple detection
│   ├── run_video.py       # full interception system
│   └── train.ipynb        # training notebook
├── assets/
│   ├── detect1.gif
│   └── lockin_final1.gif
├── models/
│   └── best.pt
└── data/
    └── sample.mp4
```

## Installation

```bash
git clone https://github.com/rami-work/Shahed-Interceptor-AI.git
cd Shahed-Interceptor-AI
pip install -r requirements.txt
```

## Usage

### Simple Detection
```bash
python src/detect.py
```

### Full Interception System
```bash
python src/run_video.py
```

### Training
Open `src/train.ipynb` and update the paths:
```python
DATA_YAML = 'path/to/your/data.yaml'
WEIGHTS   = 'path/to/last.pt'
RUN_NAME  = 'shahed_v1'
```

## Tips to Improve the Model

### 1. Use Multi-Class Training
The current model is trained on a single class (`shahed`). Train on a multi-class dataset to reduce false positives:

```python
# use a dataset with: bird, not, shahed
# then filter only shahed class in inference
for r in results:
    for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
        if cls == 2:  # shahed class only
            # draw box
```

### 2. Add More Training Data
- Collect videos from different angles and lighting conditions
- Include night vision and thermal footage
- Add different drone models (not just Shahed-136)
- Use Roboflow for auto-labeling: [roboflow.com](https://roboflow.com)

### 3. Fine-Tune Hyperparameters
```python
model.train(
    data='data.yaml',
    epochs=100,        # more epochs
    imgsz=1280,        # higher resolution
    batch=4,           # smaller batch for better convergence
    lr0=0.001,         # lower learning rate
    mosaic=1.0,        # data augmentation
    mixup=0.1,         # more augmentation
)
```

### 4. Improve Tracking
- Adjust `LOCK_TIME` in `run_video.py` (currently 1s)
- Tune P-controller gains (`kp=0.05`)
- Try different trackers: CSRT, KCF, MOSSE

### 5. Export for Deployment
```python
# export to ONNX for faster inference
model.export(format='onnx')

# export to TensorRT for GPU acceleration
model.export(format='engine', half=True)
```

## Tech Stack

- **Python 3.12**
- **YOLO11** (Ultralytics)
- **OpenCV** (tracking + visualization)
- **PyTorch** (backend)
- **CSRT Tracker** (object tracking)
- **imageio** (video output)

## Dataset

Custom Shahed drone detection dataset created on Roboflow:
- [Roboflow Universe](https://universe.roboflow.com/ramialthobait-gmail-com/shahed-detect)
- 1,582 images with bounding box annotations
- Single class: `shahed`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rami Althobait** — AI/ML Engineer
- LinkedIn: [linkedin.com/in/ramial](https://linkedin.com/in/ramial)
- GitHub: [github.com/rami-work](https://github.com/rami-work)
