import cv2, os, imageio
from ultralytics import YOLO

MODEL_PATH = 'models/best.pt'
VIDEO_PATH = 'data/test3.mp4'

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"Video not found: {VIDEO_PATH}")
    exit()

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
os.makedirs('results', exist_ok=True)
out_name = f'results/detect_{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}.mp4'
writer = imageio.get_writer(out_name, fps=fps, codec='libx264', quality=8)

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Output: {out_name}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = model(frame, conf=0.7, imgsz=1280)
    writer.append_data(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))

cap.release()
writer.close()
print(f"Saved to: {out_name}")
