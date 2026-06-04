import cv2
from ultralytics import YOLO
import os
import imageio

# --- paths ---
# put your model in models/best.pt
# put your video in data/sample.mp4
MODEL_PATH = 'models/best.pt'
VIDEO_PATH = 'data/sample.mp4'

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"Video not found: {VIDEO_PATH}")
    exit()

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = 'results/output.mp4'
os.makedirs('results', exist_ok=True)
writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Output: {output_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.6, imgsz=1280)
    annotated_frame = results[0].plot()

    writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

cap.release()
writer.close()
print(f"Saved to: {output_path}")
