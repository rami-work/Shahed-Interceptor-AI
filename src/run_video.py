import cv2
from ultralytics import YOLO
import time
import os
import imageio


class PController:
    """quick and dirty p-controller for guidance"""
    def __init__(self, kp):
        self.kp = kp

    def compute(self, error):
        return self.kp * error


pid_yaw = PController(kp=0.05)
pid_alt = PController(kp=0.05)

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

# track how long each target has been visible
appearance_timers = {}
opencv_tracker = None
locked_on = False
locked_bbox = None

# must see target for 1s before locking
LOCK_TIME = 1

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Output: {output_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    fh, fw = frame.shape[:2]

    results = model.track(frame, persist=True, conf=0.5, imgsz=1280,
                          stream=True, tracker="bytetrack.yaml")
    annotated_frame = frame.copy()

    if not locked_on:
        # YOLO scanning for targets
        for r in results:
            if r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()

            for box, tid in zip(boxes, track_ids):
                now = time.time()
                if tid not in appearance_timers:
                    appearance_timers[tid] = now

                elapsed = now - appearance_timers[tid]
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"SCANNING {elapsed:.1f}s [ID:{tid}]",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if elapsed >= LOCK_TIME and not locked_on:
                    print(f"LOCKED ON target {tid}, starting CSRT tracker")

                    try:
                        opencv_tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        opencv_tracker = cv2.legacy.TrackerCSRT_create()

                    w, h = x2 - x1, y2 - y1
                    pad = 15
                    locked_bbox = (
                        max(0, x1 - pad),
                        max(0, y1 - pad),
                        w + pad * 2,
                        h + pad * 2
                    )
                    opencv_tracker.init(frame, locked_bbox)
                    locked_on = True
                    break
    else:
        # csrt tracking + visual servoing
        ok, locked_bbox = opencv_tracker.update(frame)

        if ok:
            x, y, w, h = map(int, locked_bbox)

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(annotated_frame, "LOCKED ON", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

            cx_frame, cy_frame = fw // 2, fh // 2
            cx_target = x + w // 2
            cy_target = y + h // 2

            err_x = cx_target - cx_frame
            err_y = cy_target - cy_frame

            yaw_vel = pid_yaw.compute(err_x)
            alt_vel = pid_alt.compute(err_y)

            # crosshair + error line
            cv2.drawMarker(annotated_frame, (cx_frame, cy_frame),
                           (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(annotated_frame, (cx_target, cy_target), 5, (0, 0, 255), -1)
            cv2.line(annotated_frame, (cx_frame, cy_frame),
                     (cx_target, cy_target), (255, 0, 0), 2)

            hud = f"YAW:{yaw_vel:.2f} ALT:{alt_vel:.2f}"
            cv2.putText(annotated_frame, hud, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print("Lost target, back to scanning")
            locked_on = False
            opencv_tracker = None
            appearance_timers = {}

    writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

cap.release()
writer.close()
print(f"Saved to: {output_path}")
