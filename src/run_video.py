import cv2, os, time, imageio
from ultralytics import YOLO


class PController:
    def __init__(self, kp):
        self.kp = kp
    def compute(self, error):
        return self.kp * error


pid_yaw = PController(kp=0.05)
pid_alt = PController(kp=0.05)

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

SHAHED_CLASS_NAME = 'shahed'
SHAHED_CLASS_ID = None
for idx, name in model.names.items():
    if name == SHAHED_CLASS_NAME:
        SHAHED_CLASS_ID = idx
        break

if SHAHED_CLASS_ID is None:
    print(f"ERROR: '{SHAHED_CLASS_NAME}' class not found in model!")
    print(f"Available classes: {model.names}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
os.makedirs('results', exist_ok=True)
out_name = f'results/lock_{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}.mp4'
writer = imageio.get_writer(out_name, fps=fps, codec='libx264', quality=8)

appearance_timers = {}
opencv_tracker = None
locked_on = False
locked_bbox = None
LOCK_TIME = 1

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Output: {out_name}")
print(f"Locking only on: {SHAHED_CLASS_NAME}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    fh, fw = frame.shape[:2]
    results = model.track(frame, persist=True, conf=0.5, imgsz=1280,
                          stream=True, tracker="bytetrack.yaml")
    annotated_frame = frame.copy()

    if not locked_on:
        for r in results:
            if r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().tolist()
            classes = r.boxes.cls.int().cpu().tolist()

            for box, tid, cls in zip(boxes, track_ids, classes):
                if cls != SHAHED_CLASS_ID:
                    continue

                now = time.time()
                if tid not in appearance_timers:
                    appearance_timers[tid] = now

                elapsed = now - appearance_timers[tid]
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"SCANNING SHAHED {elapsed:.1f}s [ID:{tid}]",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if elapsed >= LOCK_TIME and not locked_on:
                    print(f"LOCKED ON shahed {tid}, starting CSRT tracker")

                    try:
                        opencv_tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        opencv_tracker = cv2.legacy.TrackerCSRT_create()

                    w, h = x2 - x1, y2 - y1
                    locked_bbox = (max(0, x1 - 15), max(0, y1 - 15), w + 30, h + 30)
                    opencv_tracker.init(frame, locked_bbox)
                    locked_on = True
                    break
    else:
        ok, locked_bbox = opencv_tracker.update(frame)

        if ok:
            x, y, w, h = map(int, locked_bbox)

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.putText(annotated_frame, "LOCKED ON SHAHED", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

            cx_frame, cy_frame = fw // 2, fh // 2
            cx_target = x + w // 2
            cy_target = y + h // 2

            err_x = cx_target - cx_frame
            err_y = cy_target - cy_frame

            yaw_vel = pid_yaw.compute(err_x)
            alt_vel = pid_alt.compute(err_y)

            cv2.drawMarker(annotated_frame, (cx_frame, cy_frame),
                           (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(annotated_frame, (cx_target, cy_target), 5, (0, 0, 255), -1)
            cv2.line(annotated_frame, (cx_frame, cy_frame),
                     (cx_target, cy_target), (255, 0, 0), 2)

            hud = f"YAW:{yaw_vel:.2f} ALT:{alt_vel:.2f}"
            cv2.putText(annotated_frame, hud, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print("Lost shahed, back to scanning")
            locked_on = False
            opencv_tracker = None
            appearance_timers = {}

    writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

cap.release()
writer.close()
print(f"Saved to: {out_name}")
