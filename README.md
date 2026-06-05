# Shahed Interceptor - AI Drone Detection & Tracking

Real-time detection and lock-on tracking of Shahed drones using YOLO11 with a two-stage pipeline: YOLO scanning -> CSRT lock-on -> visual servoing guidance.

Training on 4 classes (bird, plane, pole, shahed) to reduce false positives. The tracker only locks onto shahed and ignores everything else.

## Demo

| Detection | Lock-On Tracking |
|-----------|-----------------|
| ![Detection](assets/detect1.gif) | ![Lock-On](assets/lockin_final1.gif) |

## Results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| shahed | 348 | 739 | 96.8% | 95.5% | **97.7%** | 76.3% |
| plane | 85 | 89 | 96.3% | 95.5% | **95.3%** | 74.1% |
| pole | 121 | 152 | 91.7% | 94.7% | **96.8%** | 68.0% |
| bird | 34 | 4830 | 83.1% | 80.5% | **86.7%** | 51.6% |
| **Overall** | 626 | 5810 | 92.0% | 91.6% | **94.1%** | 67.5% |

- Model: YOLO11s (19.2MB)
- Resolution: 800px
- Dataset: ~4,000 images, 4 classes

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | bird | Birds in the sky |
| 1 | plane | Military aircraft |
| 2 | pole | Utility poles and towers |
| 3 | shahed | Shahed drone (target) |

The tracker in `run_video.py` ignores classes 0-2 and only locks onto class 3 (shahed).

## Limitations (honest)

- Bird class is weak (86.7% mAP) due to limited varied bird images in training
- Model trained at 800px, small objects may be missed
- CSRT tracker drifts on fast-moving targets or occlusions
- P-controller is basic, no PID tuning
- Tested only on daylight footage, no night/thermal

## Project Structure

```
Shahed-Interceptor-AI/
├── src/
│   ├── detect.py        simple detection only
│   ├── run_video.py      full interception system
│   └── train.ipynb       training notebook
├── assets/
│   ├── detect1.gif
│   └── lockin_final1.gif
├── models/
│   └── best.pt
├── data/
│   └── test1.mp4
├── README.md
├── requirements.txt
└── .gitignore
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

Lock-on logic:
1. YOLO scans for shahed only (ignores bird/plane/pole)
2. Target must be visible for 1 second
3. CSRT tracker takes over for blind tracking
4. P-controller computes yaw/alt commands toward target

### Training

Open `src/train.ipynb` in Colab and set paths to your Roboflow dataset:
```python
DATA_YAML = 'path/to/data.yaml'
WEIGHTS = 'yolo11s.pt'
```

## Author

**Rami Althobait** — ramialthobait@gmail.com
